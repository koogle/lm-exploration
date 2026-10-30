"""The hybrid family for trainer.loop, and the evaluation and export helpers it uses.

`HybridFamily` adapts a run config (hybrid/configs/*.json) to the common training
loop: the hybrid model with role-declared Muon/AdamW, prepared-shard data
(`data.PreparedMixture`; budget: every window token), the objective mean
next-token loss plus the decaying MTP weight times mean MTP loss (each over its
global target count), held-out loss, bits per byte, gate and sink statistics,
fixed-prompt samples, and bf16 exports to the run's HF repo every
`hf_every_tokens` and at the end.

    torchrun --nproc_per_node=8 -m model_hybrid.train --config <run.json> \\
        --run-dir <dir> --data-root <dir> --stop-unix <time> [--max-tokens N] \\
        [--max-steps N] [--resume <checkpoint>] [--stop-file <path>]

A run directory resumes from `--resume` or else its latest checkpoint;
`workflows.modal_train` runs sessions this way for every family.
"""

import argparse
import dataclasses
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import documents as doc_utils
from . import model as hybrid
from . import tokenizer as glm

H100_BF16_FLOPS = 989e12
SAMPLE_PROMPTS = (
    "The history of the printing press",
    "def fibonacci(n):",
    "Question: A train travels 120 km in 2 hours. What is its average speed?\nAnswer:",
    "In mathematics, a prime number is",
)


@dataclasses.dataclass(frozen=True)
class RunConfig:
    recipe: str
    model: dict
    target_tokens: int
    global_rows: int = 256
    microbatch: int = 4
    warmup_tokens: int = 500_000_000
    decay_fraction: float = 0.15
    optimizer: dict = dataclasses.field(default_factory=dict)
    gradient_clip: float = 1.0
    compile: bool = True
    checkpoint_seconds: int = 5400
    keep_checkpoints: int = 2
    eval_every_steps: int = 500
    eval_documents: dict = dataclasses.field(
        default_factory=lambda: {"climbmix": 1000, "code": 300, "math": 300}
    )
    sample_every_steps: int = 2000
    log_every_steps: int = 10
    hf_repo: str = ""
    hf_every_tokens: int = 5_000_000_000
    seed: int = 1234
    # An export folder (config.json, model.safetensors) whose trunk weights start the
    # run when it has no checkpoint yet, e.g. a pretrained model for mid-training.
    init_from: str = ""

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def learning_rate_factor(config, tokens):
    """Linear warmup, constant, then linear decay to 0 over the last decay_fraction."""
    if tokens < config.warmup_tokens:
        return tokens / config.warmup_tokens
    decay_start = (1 - config.decay_fraction) * config.target_tokens
    if tokens < decay_start:
        return 1.0
    return max(0.0, (config.target_tokens - tokens) / (config.target_tokens - decay_start))


def flops_per_token(model_config, lm, attended_keys):
    """Training FLOPs per token: 6 x matmul weights (incl. both output-layer uses) + attention."""
    matmul = sum(p.numel() for n, p in lm.named_parameters() if p.ndim == 2)
    heads = 1 if model_config.mtp else 0
    matmul += heads * lm.output_weight.numel()  # MTP reuses the output layer.
    full_layers = model_config.layer_kinds.count("full") + (
        1 if model_config.mtp and model_config.mtp_layer_kind == "full" else 0
    )
    attention = 12 * full_layers * model_config.n_head * model_config.head_dim * attended_keys
    return 6 * matmul + attention


def mean_attended_keys(positions):
    return float((positions.float() + 1).mean())


def model_config_of(config):
    model_config = hybrid.HybridConfig(
        **{k: tuple(v) if isinstance(v, list) else v for k, v in config.model.items()}
    )
    if model_config.eos_id != glm.EOS_ID:
        raise ValueError("model eos_id must be the GLM tokenizer's EOS")
    return model_config


# The launcher creates this file when a graceful stop is requested.
STOP_FILE = os.environ.get("HYBRID_STOP_FILE", "/tmp/hybrid-stop")


def gate_sink_statistics(lm):
    """Parameter statistics of every attention sink, gate bias and KDA decay."""
    stats = {}
    for name, module in lm.named_modules():
        if isinstance(module, hybrid.Attention):
            sink = module.sink.detach().float()
            bias = module.gate_bias.detach().float()
            stats[name] = {
                "sink_mean": sink.mean().item(),
                "sink_max": sink.max().item(),
                "gate_bias_open": torch.sigmoid(bias).mean().item(),
            }
        elif name.endswith("mixer") and hasattr(module, "A_log"):
            stats[name] = {"decay_rate_mean": module.A_log.detach().float().exp().mean().item()}
    return stats


@torch.no_grad()
def evaluate(lm, heldout, device, eos_id):
    """Held-out loss, bits per byte and MTP loss per source; gate/sink activations."""
    lm.eval()
    results = {}
    hybrid.Attention.record_statistics = True
    try:
        for name, (inputs, labels, total_bytes) in heldout.items():
            sums = {"ce": 0.0, "count": 0, "mtp_ce": 0.0, "mtp_count": 0}
            activations = []
            for start in range(0, len(inputs), 4):
                x = torch.from_numpy(inputs[start : start + 4]).to(device)
                y = torch.from_numpy(labels[start : start + 4]).to(device)
                targets = doc_utils.mtp_labels(x, y, eos_id)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    losses = lm(x, y, targets)
                for key in sums:
                    sums[key] += losses[key].item() if key.endswith("ce") else int(losses[key])
                activations.append(
                    [m.statistics for m in lm.modules() if isinstance(m, hybrid.Attention)]
                )
            layers = list(zip(*activations, strict=True))
            results[name] = {
                "loss": sums["ce"] / sums["count"],
                "bits_per_byte": sums["ce"] / math.log(2) / total_bytes,
                "mtp_loss": sums["mtp_ce"] / max(sums["mtp_count"], 1),
                "tokens": sums["count"],
                "gate_open": [float(np.mean([s["gate_open"] for s in layer])) for layer in layers],
                "sink_weight": [
                    float(np.mean([s["sink_weight"] for s in layer])) for layer in layers
                ],
            }
    finally:
        hybrid.Attention.record_statistics = False
        lm.train()
    return results


@torch.no_grad()
def samples(lm, tokenizer, device, new_tokens=48):
    """Greedy continuations of fixed prompts (full recompute per token)."""
    lm.eval()
    outputs = []
    for prompt in SAMPLE_PROMPTS:
        ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        for _ in range(new_tokens):
            x = torch.tensor([ids], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ids.append(int(lm(x)[0, -1].argmax()))
        outputs.append({"prompt": prompt, "completion": tokenizer.decode(ids)})
    lm.train()
    return outputs


def upload_export(lm, model_config, run_dir, progress, repo):
    """bf16 trunk weights and config under `<run_dir>/export/`, and to the HF model repo
    with the run's metrics in a background thread."""
    import threading

    from safetensors.torch import save_file

    folder = Path(run_dir, "export", f"tokens-{progress['tokens']:012d}")
    folder.mkdir(parents=True, exist_ok=True)
    weights = {
        k: v.detach().to("cpu", torch.bfloat16).contiguous()
        for k, v in lm.state_dict().items()
        if not k.startswith("mtp.")
    }
    save_file(weights, folder / "model.safetensors")
    config = dataclasses.asdict(model_config) | {
        "tokenizer": glm.SOURCE,
        "tokenizer_sha256": glm.SHA256,
    }
    (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (folder / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    logs = [Path(run_dir, name) for name in ("metrics.jsonl", "validation.jsonl")]

    def push():
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo, private=True, exist_ok=True)
        api.upload_folder(repo_id=repo, folder_path=folder, path_in_repo=folder.name)
        for log in logs:
            if log.exists():
                api.upload_file(path_or_fileobj=log, path_in_repo=log.name, repo_id=repo)

    thread = threading.Thread(target=push, daemon=False)
    thread.start()
    return thread


def recipe_data(config, data_root):
    """(identity, mixture factory, held-out factory(mixture, context)) for a run config's
    prepared-shard recipe; held-out windows wait for each source's held-out shard."""
    from .data import PreparedMixture, heldout_index, heldout_windows, shard_paths

    recipe = json.loads(Path(config.recipe).read_text())
    root = Path(data_root) / recipe["name"]

    def heldout(mixture, context):
        for name, source in recipe["sources"].items():
            meta = shard_paths(mixture.root, name, heldout_index(source))[2]
            waited = time.monotonic()
            while not meta.exists():
                if time.monotonic() - waited > mixture.wait_seconds:
                    raise TimeoutError(f"Held-out shard for {name} was not prepared")
                time.sleep(15)
        return {
            name: heldout_windows(recipe, mixture.root, name, context, count)
            for name, count in config.eval_documents.items()
        }

    identity = {"recipe": recipe, "tokenizer_sha256": glm.SHA256}
    return identity, lambda: PreparedMixture(recipe, root), heldout


class HybridData:
    """A recipe mixture for the loop: every window token counts toward the budget."""

    def __init__(self, identity, mixture, state):
        self.identity, self.mixture = identity, mixture
        if state is not None:
            mixture.load_state_dict(state)

    def next_batch(self, rows):
        inputs, labels, names = self.mixture.next_batch(rows)
        return inputs, labels, names, int(inputs.size)

    def state_dict(self):
        return self.mixture.state_dict()

    def load_state_dict(self, state):
        self.mixture.load_state_dict(state)

    def close(self):
        pass


class HybridFamily:
    """A hybrid run config for trainer.loop, training into `output` (its exports go to
    `output/export/`); `data` = (identity, mixture factory) overrides the recipe
    (synthetic benchmarks), and then nothing is held out."""

    def __init__(self, config, output, data_root=None, data=None):
        self.config, self.output = config, Path(output)
        self.settings = dataclasses.asdict(config)
        self.model_config = model_config_of(config)
        self.global_rows, self.microbatch = config.global_rows, config.microbatch
        self.target_tokens, self.gradient_clip = config.target_tokens, config.gradient_clip
        self.seed, self.log_every = config.seed, config.log_every_steps
        self.evaluate_every = config.eval_every_steps
        self.checkpoint_every, self.checkpoint_seconds = 0, config.checkpoint_seconds
        self.keep_checkpoints = config.keep_checkpoints
        self.collective_timeout_seconds = 3600
        if data is None:
            self.identity, self.factory, self.heldout_factory = recipe_data(config, data_root)
        else:
            (self.identity, self.factory), self.heldout_factory = data, None
        self.lm, self.uploads = None, []
        self.next_upload = None

    def learning_rate_factor(self, tokens):
        return learning_rate_factor(self.config, tokens)

    def build(self, device):
        torch.backends.cuda.matmul.allow_tf32 = True
        self.lm = hybrid.HybridLM(self.model_config).to(device)
        return self.lm

    def initialize(self, model):
        """An `init_from` export's trunk weights start a fresh run (mid-training);
        otherwise the model keeps its own initialization."""
        if self.config.init_from:
            from safetensors.torch import load_file

            path = Path(self.config.init_from) / "model.safetensors"
            model.load_state_dict(load_file(path, device="cpu"))

    def compile(self, model):
        return torch.compile(model) if self.config.compile else model

    def make_optimizer(self, model):
        from trainer.optimizers import OptimizerConfig, RoleOptimizer

        settings = {
            k: tuple(v) if isinstance(v, list) else v for k, v in self.config.optimizer.items()
        }
        return RoleOptimizer(model, OptimizerConfig(**settings), fused=True)

    def open_data(self, deadline, state):
        return HybridData(self.identity, self.factory(), state)

    def metadata(self):
        return {
            "architecture": {
                "module": hybrid.HybridConfig.__module__,
                "config": dataclasses.asdict(self.model_config),
            },
            "tokenizer": glm.SOURCE,
        }

    def source_files(self):
        from . import data, documents, fused_loss, kda, softmax_kernel
        from . import model as hybrid_model
        from . import tokenizer as hybrid_tokenizer

        modules = (data, documents, fused_loss, hybrid_model, kda, softmax_kernel, hybrid_tokenizer)
        files = {f"hybrid.{Path(m.__file__).stem}": Path(m.__file__) for m in modules}
        return {"family": Path(__file__), **files}

    def branchable(self, saved):
        return False

    def check_resume(self, saved):
        """Resume needs only the loop's checks."""

    def heldout(self, data):
        if self.heldout_factory is None:
            return None
        windows = self.heldout_factory(data.mixture, self.model_config.block_size)
        return {"windows": windows, "tokenizer": glm.load()}

    def evaluate(self, model, heldout, progress, final):
        if heldout is None:
            return {}
        device = next(model.parameters()).device
        result = {
            "heldout": evaluate(model, heldout["windows"], device, self.model_config.eos_id),
            "parameters": gate_sink_statistics(model),
        }
        if final or progress["step"] % self.config.sample_every_steps == 0:
            result["samples"] = samples(model, heldout["tokenizer"], device)
        return result

    def prepare(self, inputs, labels):
        counts = {"ntp": (labels != -100).sum()}
        if not self.model_config.mtp:
            return {}, counts
        targets = doc_utils.mtp_labels(inputs, labels, self.model_config.eos_id)
        return {"mtp_labels": targets}, counts | {"mtp": (targets != -100).sum()}

    def loss(self, model, inputs, labels, extras, counts, progress):
        losses = model(inputs, labels, extras.get("mtp_labels"))
        objective = losses["loss"] / counts["ntp"]
        sums = {"ce": losses["ce"], "z": losses["z"]}
        if self.model_config.mtp:
            mtp_count = counts["mtp"].clamp_min(1)
            weight = hybrid.mtp_weight(self.model_config, progress["tokens"] / self.target_tokens)
            objective = objective + weight * losses["mtp_loss"] / mtp_count
            sums["mtp_ce"] = losses["mtp_ce"]
        return objective, sums

    def log(self, sums, counts):
        row = {
            "loss": float(sums["ce"] / counts["ntp"]),
            "z": float(sums["z"] / counts["ntp"]),
        }
        if self.model_config.mtp:
            row["mtp_loss"] = float(sums["mtp_ce"] / counts["mtp"].clamp_min(1))
        return row

    def step_metrics(self, inputs, seconds, budget, world):
        documents = doc_utils.document_ids(inputs[:8], self.model_config.eos_id)
        keys = mean_attended_keys(doc_utils.document_positions(documents))
        flops = flops_per_token(self.model_config, self.lm, keys)
        return {"mfu": budget / seconds * flops / (world * H100_BF16_FLOPS)}

    def after_step(self, data, model, progress):
        """A bf16 export to the HF repo every `hf_every_tokens`."""
        config = self.config
        if not config.hf_repo:
            return
        if self.next_upload is None:
            done = progress["tokens"] - progress["tokens"] % config.hf_every_tokens
            self.next_upload = done + config.hf_every_tokens
        if progress["tokens"] >= self.next_upload:
            self.uploads.append(
                upload_export(model, self.model_config, self.output, progress, config.hf_repo)
            )
            self.next_upload += config.hf_every_tokens

    def finish(self, data, model, progress, reason):
        """The final export when the run reached its target; waits for every upload."""
        exported = None
        if self.config.hf_repo and reason == "token_limit":
            self.uploads.append(
                upload_export(model, self.model_config, self.output, progress, self.config.hf_repo)
            )
            exported = f"tokens-{progress['tokens']:012d}"
        for thread in self.uploads:
            thread.join()
        return {"export": exported}


def main():
    from distributed.config import DistributedRuntime
    from trainer import loop

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--stop-unix", type=float, required=True)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", help="A checkpoint to resume (default: the run's latest)")
    parser.add_argument("--stop-file", default=STOP_FILE)
    args = parser.parse_args()
    loop.train(
        HybridFamily(RunConfig.load(args.config), args.run_dir, args.data_root),
        args.run_dir,
        runtime=DistributedRuntime.from_torchrun(os.environ),
        resume=args.resume,
        stop_unix=args.stop_unix,
        max_steps=args.max_steps,
        stop_tokens=args.max_tokens,
        stop_file=args.stop_file,
    )


if __name__ == "__main__":
    main()
