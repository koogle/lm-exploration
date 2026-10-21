"""GPU correctness and batch-one measurements; results are never committed to Git."""

import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import atomic_json, summarize, validate
from megakernel.model import Decoder, compile_kernel, load_teacher


def errors(actual, reference):
    a, b = actual.float().flatten(), reference.float().flatten()
    return {
        "nrmse": (
            (a - b).square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-8)
        ).item(),
        "max_abs": (a - b).abs().max().item(),
        "cosine": torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
        "finite": bool(torch.isfinite(a).all()),
    }


@torch.inference_mode()
def run(root, attempt, manifest):
    import modal

    volume = modal.Volume.from_name("lm-exploration-teacher-rollouts")
    out = Path(root) / "attempts" / attempt
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(Path(root) / "manifest.json", manifest)
    started = time.monotonic()
    report = {
        "status": "starting",
        "actual_billed_usd": None,
        "backend": "megakernel",
        "comparison": "Transformers 5.4.0 eager BF16; not vLLM throughput",
        "correctness": [],
    }
    atomic_json(
        out / "runtime.json",
        {
            "packages": {
                d.metadata["Name"]: d.version
                for d in importlib.metadata.distributions()
            },
            "gpu": subprocess.check_output(["nvidia-smi"], text=True),
            "nvcc": subprocess.check_output(["nvcc", "--version"], text=True),
        },
    )

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(out / "report.json", report)
        volume.commit()

    try:
        lib = Path("/experiment/megakernel/kernel.so")
        if not lib.exists():
            t = time.monotonic()
            lib = compile_kernel(out / "kernel.so")
            report["compilation_seconds"] = time.monotonic() - t
        t = time.monotonic()
        model, head, tokenizer = load_teacher("/runs/teacher-hf-cache")
        report["model_load_and_download_seconds"] = time.monotonic() - t
        print("TEACHER_LOADED", report["model_load_and_download_seconds"], flush=True)
        decoder = Decoder(model, head, lib)
        report["cooperative_resident_block_limit"] = decoder.resident_blocks
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        save()
        if manifest["config"].get("targets_only"):
            prior = Path(
                "/runs/qwen35-targets-kernel-20261022-01/attempts/targets-kernel-01/report.json"
            )
            previous = json.loads(prior.read_text())
            if not previous.get("buffered_distillation", {}).get("correctness_passed"):
                raise ValueError("Prior sparse-target qualification has not passed")
            prior_manifest = json.loads(
                (prior.parents[2] / "manifest.json").read_text()
            )
            for key in (
                "megakernel/model.py",
                "megakernel/kernel.cu",
                "megakernel/targets.py",
            ):
                if (
                    prior_manifest["source_hashes"][key]
                    != manifest["source_hashes"][key]
                ):
                    raise ValueError(f"Qualified source changed: {key}")
            report["qualification_source"] = str(prior)
            from megakernel.throughput import measure

            measure(decoder, tokenizer, manifest, out, volume, report, save)
            report["status"] = "completed"
            return report
        # Capture every transformer layer without relying on hidden-state output conventions.
        reference_layers = {}
        hooks = [
            layer.register_forward_hook(
                lambda mod, args, value, i=i: reference_layers.__setitem__(
                    i, value.detach().clone()
                )
            )
            for i, layer in enumerate(model.layers)
        ]
        top_matches = []
        prompts = [
            "The capital of France is",
            "def clamp(x, lo, hi):",
            "Return JSON with the key count and integer value 3.",
        ]
        for prompt in prompts:
            decoder.reset()
            cache = None
            ids = tokenizer.encode(prompt, add_special_tokens=False)[:12]
            for position, token in enumerate(ids):
                reference = model(
                    input_ids=torch.tensor([[token]], device="cuda"),
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = reference.past_key_values
                ref_logits = torch.nn.functional.linear(
                    reference.last_hidden_state[0, -1], head
                ).float()
                logits = decoder.step(token, debug=True)
                torch.cuda.synchronize()
                row = {
                    "prompt": prompt,
                    "position": position,
                    "token": token,
                    "logits": errors(logits, ref_logits),
                    "top1_equal": int(logits.argmax()) == int(ref_logits.argmax()),
                    "kernel_top1": int(logits.argmax()),
                    "reference_top1": int(ref_logits.argmax()),
                    "layers": [
                        errors(decoder.debug[i + 1], reference_layers[i])
                        for i in range(32)
                    ],
                }
                linear_idx, full_idx = 0, 0
                row["states"], row["keys"] = [], []
                for i, layer_type in enumerate(model.config.layer_types):
                    if layer_type == "linear_attention":
                        row["states"].append(
                            errors(
                                decoder.state[linear_idx].transpose(-1, -2),
                                cache.recurrent_states[i][0],
                            )
                        )
                        linear_idx += 1
                    else:
                        row["keys"].append(
                            errors(
                                decoder.k[full_idx, :, : position + 1],
                                cache.key_cache[i][0],
                            )
                        )
                        full_idx += 1
                report["correctness"].append(row)
                top_matches.append(row["top1_equal"])
                print("CHECK", position, row["logits"], row["top1_equal"], flush=True)
            save()
        for hook in hooks:
            hook.remove()
        # Thresholds are predeclared diagnostics, not exact arithmetic equivalence.
        report["thresholds"] = {
            "max_logit_nrmse": 0.04,
            "min_logit_cosine": 0.999,
            "min_top1_fraction": 0.95,
            "max_recurrent_state_nrmse": 0.08,
        }
        report["top1_match_fraction"] = sum(top_matches) / len(top_matches)
        report["correctness_passed"] = (
            report["top1_match_fraction"] >= 0.95
            and all(
                r["logits"]["finite"]
                and r["logits"]["nrmse"] <= 0.04
                and r["logits"]["cosine"] >= 0.999
                for r in report["correctness"]
            )
            and all(
                s["nrmse"] <= 0.08 for r in report["correctness"] for s in r["states"]
            )
        )
        save()
        if not report["correctness_passed"]:
            report["status"] = "correctness_failed"
            return report
        # Reset reproducibility catches stale state and uninitialized scratch.
        ids = tokenizer.encode("Hello", add_special_tokens=False)
        first = decoder.generate(ids, 8)
        second = decoder.generate(ids, 8)
        report["reset_reproducible"] = first == second
        if not report["reset_reproducible"]:
            report["status"] = "reset_failed"
            return report
        from megakernel.buffer_probe import qualify

        report["buffered_distillation"] = qualify(decoder, tokenizer, out)
        save()
        if not report["buffered_distillation"]["correctness_passed"]:
            report["status"] = "buffered_correctness_failed"
            return report
        from megakernel.throughput import measure

        measure(decoder, tokenizer, manifest, out, volume, report, save)
        measurements = []
        for backend in ("megakernel", "transformers_eager"):
            rows = []
            generation_seconds = 0
            for prompt in manifest["prompts"]:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                ids = tokenizer.encode(text, add_special_tokens=False)
                torch.cuda.synchronize()
                t = time.perf_counter()
                if backend == "megakernel":
                    generated = decoder.generate(
                        ids, max_tokens=128, eos=tokenizer.eos_token_id
                    )
                else:
                    cache, generated = None, []
                    # Same sequential prefill protocol as the custom backend; no claim about optimized prefill.
                    for token in ids:
                        ref = model(
                            input_ids=torch.tensor([[token]], device="cuda"),
                            past_key_values=cache,
                            use_cache=True,
                        )
                        cache = ref.past_key_values
                    for step in range(128):
                        next_token = int(
                            torch.nn.functional.linear(
                                ref.last_hidden_state[0, -1], head
                            ).argmax()
                        )
                        if next_token == tokenizer.eos_token_id:
                            break
                        generated.append(next_token)
                        if step + 1 < 128:
                            ref = model(
                                input_ids=torch.tensor([[next_token]], device="cuda"),
                                past_key_values=cache,
                                use_cache=True,
                            )
                            cache = ref.past_key_values
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t
                generation_seconds += elapsed
                output = tokenizer.decode(generated, skip_special_tokens=True)
                finish = "length" if len(generated) == 128 else "stop"
                t = time.perf_counter()
                validation = validate(prompt, output, finish)
                row = {
                    "prompt_id": prompt["id"],
                    "output": output,
                    "output_ids": generated,
                    "prompt_tokens": len(ids),
                    "output_tokens": len(generated),
                    "finish_reason": finish,
                    "validation": validation,
                    "error": None,
                    "verification_seconds": time.perf_counter() - t,
                    "generation_seconds": elapsed,
                    "sampling": {"temperature": 0, "max_tokens": 128},
                    "synthetic": True,
                }
                rows.append(row)
                atomic_json(out / f"{backend}-{prompt['id']}.json", row)
                print(
                    "GENERATED",
                    backend,
                    prompt["id"],
                    len(generated),
                    elapsed,
                    validation,
                    flush=True,
                )
                volume.commit()
            summary = summarize(
                rows, generation_seconds, manifest["config"]["gpu_usd_per_hour"]
            )
            summary["backend"] = backend
            measurements.append(summary)
            report["measurements"] = measurements
            save()
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["peak_torch_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["estimated_gpu_usd"] = (
            (time.monotonic() - started) * manifest["config"]["gpu_usd_per_hour"] / 3600
        )
        save()
    return report


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], json.loads(Path(sys.argv[3]).read_text()))
