"""Time RL rollouts (`trainer/rollout.py`) of the 124M decoder on one H100.

Run from training/:
    modal run -m kernels.modal_rollout::main [--batch 256] [--prompt 200] [--new 384] \
        [--output artifacts/kernels/rollout.json]

Random weights, or `--checkpoint` (an SFT or RL checkpoint on the run volume, which
only our own jobs write), and no EOS, so every engine decodes exactly `new` tokens per
row. Reports
milliseconds per decode step for each engine, the top CUDA kernels of the eager engine,
and for the custom engine the greedy agreement and logit error against eager decoding.
"""

import json
import time
from pathlib import Path

import modal

from kernels.image import LAUNCH_LIMIT_SECONDS, image, limits

app = modal.App("rollout-profile")
image = image.add_local_dir("trainer", "/project/trainer", ignore=["__pycache__", "kernels"])


@app.function(
    image=image, volumes={"/runs": modal.Volume.from_name("lm-exploration-training")}, **limits
)
def profile(deadline, batch, prompt, new, checkpoint):
    import torch
    from model_100m.model import GPT, ModelConfig
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as torch_profile

    from trainer import rollout

    if time.time() > deadline:
        raise RuntimeError("Launch deadline passed; refusing work")
    torch.manual_seed(0)
    if checkpoint:
        saved = torch.load(Path("/runs") / checkpoint, map_location="cpu", weights_only=False)
        model = GPT(ModelConfig(**saved["model_config"]))
        model.load_state_dict(saved["model"])
        model = model.cuda().eval()
    else:
        model = GPT(ModelConfig(vocab_size=32768, block_size=2048)).cuda().eval()
    lengths = torch.randint(prompt // 2, prompt + 1, (batch,)).tolist()
    prompts = [torch.randint(4, 32768, (n,)).tolist() for n in lengths]
    results = {"device": torch.cuda.get_device_name(), "batch": batch, "prompt": prompt}
    results["checkpoint"] = checkpoint or "random"
    results["new_tokens"] = new

    def timed(engine, **kwargs):
        """Milliseconds of the second identical call (the first captures graphs)."""
        for _ in range(2):
            generator = torch.Generator(device="cuda").manual_seed(0)
            torch.cuda.synchronize()
            start = time.perf_counter()
            rollout.sample(model, prompts, new, -1, generator=generator, engine=engine, **kwargs)
            torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000

    for engine in rollout.ENGINES:
        for top_k in (0, 32):
            ms = timed(engine, top_k=top_k)
            results[f"{engine}_top{top_k}_ms_per_token"] = ms / new
    with torch_profile(activities=[ProfilerActivity.CUDA]) as trace:
        rollout.sample(model, prompts, 32, -1, engine="eager")
    events = sorted(trace.key_averages(), key=lambda e: -e.device_time_total)[:12]
    results["eager_top_kernels"] = [
        {"name": e.key[:80], "ms": e.device_time_total / 1000, "calls": e.count} for e in events
    ]
    with torch_profile(activities=[ProfilerActivity.CUDA]) as trace:
        rollout.sample(model, prompts, 32, -1, engine="graph")
    events = sorted(trace.key_averages(), key=lambda e: -e.device_time_total)[:12]
    results["graph_top_kernels"] = [
        {"name": e.key[:80], "ms": e.device_time_total / 1000, "calls": e.count} for e in events
    ]
    # Agreement of the custom engine with eager decoding, greedy.
    eager = rollout.sample(model, prompts, 64, -1, temperature=0, engine="eager")
    custom = rollout.sample(model, prompts, 64, -1, temperature=0, engine="graph")
    same = [a == b for a, b in zip(eager.completions, custom.completions, strict=True)]
    first = [
        next((i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y), len(a))
        for a, b in zip(eager.completions, custom.completions, strict=True)
    ]
    results["greedy_identical_rows"] = sum(same) / len(same)
    results["greedy_mean_first_divergence"] = sum(first) / len(first)
    results["logit_error"] = rollout.compare_engines(model, prompts[:32])
    return results


@app.function(
    image=image, volumes={"/runs": modal.Volume.from_name("lm-exploration-training")}, **limits
)
def diagnose(deadline, checkpoint, batch=32, prompt=200):
    """Error of one decode step's log-probabilities against the learner's full forward,
    with the custom kernels swapped one at a time for PyTorch references."""
    import torch
    import torch.nn.functional as F
    from model_100m.model import GPT, ModelConfig

    from kernels.decode import GraphDecoder
    from trainer import rollout

    if time.time() > deadline:
        raise RuntimeError("Launch deadline passed; refusing work")
    saved = torch.load(Path("/runs") / checkpoint, map_location="cpu", weights_only=False)
    model = GPT(ModelConfig(**saved["model_config"]))
    model.load_state_dict(saved["model"])
    model = model.cuda().eval()
    torch.manual_seed(0)
    lengths = torch.randint(prompt // 2, prompt + 1, (batch,)).tolist()
    prompts = [torch.randint(4, 32768, (n,)).tolist() for n in lengths]

    class TorchRope(GraphDecoder):
        def rope_append(self, qkv, index):
            q, k, v = (t.view(self.batch, self.heads, 1, 64) for t in qkv.chunk(3, -1))
            positions = self.positions.long()[:, None]
            q, k = rollout.rotate(q, positions), rollout.rotate(k, positions)
            slot = int(self.slot)
            self.keys[index][:, :, slot] = k[:, :, 0]
            self.values[index][:, :, slot] = v[:, :, 0]
            return q[:, :, 0].contiguous()

    class TorchAttention(GraphDecoder):
        def attend(self, q, index):
            end = int(self.slot) + 1
            slots = torch.arange(end, device=q.device)
            mask = (slots[None, :] >= self.start[:, None])[:, None, None, :]
            out = F.scaled_dot_product_attention(
                q[:, :, None],
                self.keys[index][:, :, :end],
                self.values[index][:, :, :end],
                attn_mask=mask,
            )
            return out[:, :, 0]

    class TorchBoth(TorchRope, TorchAttention):
        pass

    with torch.no_grad():
        base = GraphDecoder(model, batch, 512, 0, True)
        first = base.prefill(prompts).argmax(-1)
        keys = [k.clone() for k in base.keys]
        values = [v.clone() for v in base.values]
        learner = []
        for row, tokens in enumerate(prompts):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                learner.append(
                    model(torch.tensor([tokens + [int(first[row])]], device="cuda"))[0, -1]
                )
        learner = torch.log_softmax(torch.stack(learner).float(), -1)
        results = {}
        for name, kind in (
            ("custom", GraphDecoder),
            ("torch_rope", TorchRope),
            ("torch_attention", TorchAttention),
            ("torch_both", TorchBoth),
        ):
            decoder = kind(model, batch, 512, 0, True)
            decoder.prefill(prompts)
            for target, source in zip(decoder.keys + decoder.values, keys + values, strict=True):
                target.copy_(source)
            decoder.token.copy_(first)
            logp = torch.log_softmax(decoder.logits(), -1)
            difference = (logp - learner).abs()
            results[name] = {
                "expected_abs": (learner.exp() * difference).sum(-1).mean().item(),
                "max_abs_top": difference.gather(1, learner.argmax(-1, keepdim=True)).max().item(),
            }
        valid = torch.zeros((batch, 512), dtype=torch.bool, device="cuda")
        width = max(lengths)
        for row, n in enumerate(lengths):
            valid[row, width - n : width + 1] = True
        cache = list(zip(keys, values, strict=True))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            eager = rollout.forward(
                model,
                first[:, None],
                torch.tensor(lengths, device="cuda")[:, None],
                cache,
                width,
                valid,
            )
        difference = (torch.log_softmax(eager, -1) - learner).abs()
        results["eager"] = {
            "expected_abs": (learner.exp() * difference).sum(-1).mean().item(),
            "max_abs_top": difference.gather(1, learner.argmax(-1, keepdim=True)).max().item(),
        }
    return results


@app.function(
    image=image, volumes={"/runs": modal.Volume.from_name("lm-exploration-training")}, **limits
)
def replay_check(deadline, checkpoint, batch=32, prompt=200, new=32):
    """Graph-sampled tokens re-scored step by step without the graph and by the learner:
    separates errors of graph replay and of the sampling kernel."""
    import torch
    from model_100m.model import GPT, ModelConfig

    from kernels.decode import GraphDecoder
    from trainer import rollout

    if time.time() > deadline:
        raise RuntimeError("Launch deadline passed; refusing work")
    saved = torch.load(Path("/runs") / checkpoint, map_location="cpu", weights_only=False)
    model = GPT(ModelConfig(**saved["model_config"]))
    model.load_state_dict(saved["model"])
    model = model.cuda().eval()
    torch.manual_seed(0)
    lengths = torch.randint(prompt // 2, prompt + 1, (batch,)).tolist()
    prompts = [torch.randint(4, 32768, (n,)).tolist() for n in lengths]
    generator = torch.Generator(device="cuda").manual_seed(0)
    samples = rollout.sample(model, prompts, new, -1, 1.0, generator, engine="graph")
    tokens = torch.tensor(samples.completions, device="cuda")
    with torch.no_grad():
        direct = GraphDecoder(model, batch, 512, 0, True)
        logits = direct.prefill(prompts)
        stepwise = []
        for step in range(new):
            stepwise.append(torch.log_softmax(logits, -1).gather(1, tokens[:, step, None])[:, 0])
            direct.token.copy_(tokens[:, step])
            logits = direct.logits()
            direct.positions += 1
            direct.slot += 1
        stepwise = torch.stack(stepwise, 1)
        learner = []
        for row, tokens_row in enumerate(prompts):
            sequence = tokens_row + samples.completions[row]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(torch.tensor([sequence], device="cuda"))[0].float()
            scores = torch.log_softmax(out[len(tokens_row) - 1 : -1], -1)
            learner.append(scores.gather(1, tokens[row, :, None])[:, 0])
        learner = torch.stack(learner)
    recorded = samples.logprobs
    return {
        "recorded_vs_stepwise": (recorded - stepwise).abs().mean().item(),
        "stepwise_vs_learner": (stepwise - learner).abs().mean().item(),
        "recorded_vs_learner": (recorded - learner).abs().mean().item(),
        "recorded_vs_stepwise_by_step": (recorded - stepwise).abs().mean(0).tolist(),
    }


@app.local_entrypoint()
def check(checkpoint: str):
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    print(json.dumps(diagnose.remote(deadline, checkpoint), indent=2))
    print(json.dumps(replay_check.remote(deadline, checkpoint), indent=2))


@app.local_entrypoint()
def main(
    batch: int = 256, prompt: int = 200, new: int = 384, checkpoint: str = "", output: str = ""
):
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    result = profile.remote(deadline, batch, prompt, new, checkpoint)
    path = Path(output or "artifacts/kernels/rollout.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
