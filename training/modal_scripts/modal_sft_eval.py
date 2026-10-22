"""Greedy executable-repair diagnostic for SFT or pretraining checkpoints on Modal.

Run from training/:
    modal run -m modal_scripts.modal_sft_eval --dataset <curriculum.jsonl> \
        --checkpoints base=<run/step-N.pt>,sft=<run/step-N.pt> [--output <result.json>]

The dataset must carry executable `cases` and `family` fields, as written by
`training_data.repair_curriculum`. Each named checkpoint generates answers for two held-out
families per language on one L40S (15-minute timeout, no retries). Each answer runs in its own
network-blocked Modal Sandbox without volumes or secrets (90-second timeout); expected outputs
stay outside the sandbox. The launcher records a deadline 24 hours after launch and refuses new
generation calls or sandboxes after it. Results are structured JSON in the ignored artifacts/.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

from distributed.image import image

LAUNCH_LIMIT_SECONDS = 24 * 60 * 60
DECODING = "greedy BF16, 128 new tokens, PAD/BOS/UNK suppressed"

app = modal.App("simple-llm-sft-eval")
run_volume = modal.Volume.from_name("lm-exploration-training")
image = image.add_local_dir("modal_scripts", "/project/modal_scripts")
# Compilers and runtimes only; candidate code never runs in the GPU container.
runtime_image = modal.Image.debian_slim(python_version="3.12").apt_install(
    "gcc", "g++", "rustc", "golang-go", "default-jdk-headless", "nodejs", "node-typescript"
)


def check_deadline(deadline):
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing further evaluation work")


def select(rows, limit_per_language):
    """Held-out rows, one per family, up to `limit_per_language` per language."""
    counts, families, chosen = {}, set(), []
    for row in rows:
        lang = row["language"]
        if row["split"] != "validation" or counts.get(lang, 0) >= limit_per_language:
            continue
        if (lang, row["family"]) in families:
            continue
        families.add((lang, row["family"]))
        counts[lang] = counts.get(lang, 0) + 1
        chosen.append(row)
    return chosen


@app.function(
    image=image,
    gpu="L40S",
    cpu=4,
    memory=16384,
    timeout=900,
    retries=0,
    single_use_containers=True,
    volumes={"/runs": run_volume},
)
def generate(checkpoint, rows, deadline):
    import torch
    from model_100m.model import GPT, ModelConfig
    from training_data.repair_sft import prompt
    from training_data.subword import (
        BOS_TOKEN_ID,
        EOS_TOKEN_ID,
        PAD_TOKEN_ID,
        UNK_TOKEN_ID,
        load_frozen,
    )

    from trainer.sft_recipe import checkpoint as checked

    check_deadline(deadline)
    run_volume.reload()
    path = Path("/runs") / checked(checkpoint)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["format"] == "scholarly-ddp-v1":
        cfg = {k: saved["config"][k] for k in ("n_layer", "n_head", "n_embd")}
        cfg.update(vocab_size=32768, block_size=saved["config"]["context"])
    elif saved["format"] == "repair-sft-v2":
        cfg = saved["model_config"]
    else:
        raise ValueError("Unknown checkpoint format")
    model = GPT(ModelConfig(**cfg)).cuda().eval()
    model.load_state_dict(saved["model"])
    del saved
    tokenizer = load_frozen()
    results = []
    for row in rows:
        ids = [BOS_TOKEN_ID] + tokenizer.encode(prompt(row), add_special_tokens=False).ids
        generated, eos = [], False
        with torch.inference_mode():
            for _ in range(128):
                x = torch.tensor([ids + generated], device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)[0, -1].float()
                logits[[PAD_TOKEN_ID, BOS_TOKEN_ID, UNK_TOKEN_ID]] = -float("inf")
                token = int(logits.argmax())
                if token == EOS_TOKEN_ID:
                    eos = True
                    break
                generated.append(token)
        results.append(
            {
                "id": row["id"],
                "language": row["language"],
                "family": row["family"],
                "code": tokenizer.decode(generated),
                "tokens": len(generated),
                "eos": eos,
            }
        )
    return results


def observations(language, code, cases):
    """Program printing solve(a, b) per case. Expected answers stay outside the sandbox."""
    pairs = [(a, b) for a, b, _ in cases]
    if language == "python":
        calls = [f"print(solve({a}, {b}))" for a, b in pairs]
        return "\n".join([code, *calls])
    if language in ("javascript", "typescript"):
        calls = [f"console.log(solve({a}, {b}));" for a, b in pairs]
        return "\n".join([code, *calls])
    if language == "bash":
        return "\n".join(["set -eu", code, *(f"solve {a} {b}" for a, b in pairs)])
    if language in ("c", "cpp"):
        calls = [f'printf("%d\\n", solve({a}, {b}));' for a, b in pairs]
        return "\n".join(["#include <stdio.h>", code, "int main(void) {", *calls, "return 0; }"])
    if language == "rust":
        calls = [f'println!("{{}}", solve({a}, {b}));' for a, b in pairs]
        return "\n".join([code, "fn main() {", *calls, "}"])
    if language == "go":
        calls = [f"fmt.Println(solve({a}, {b}))" for a, b in pairs]
        return "\n".join(['package main\nimport "fmt"', code, "func main() {", *calls, "}"])
    if language == "java":
        calls = [f"System.out.println(solve({a}, {b}));" for a, b in pairs]
        return "\n".join(
            ["public class Main {", code, "public static void main(String[] args) {", *calls, "}}"]
        )
    raise ValueError(language)


def grade(row, code, deadline):
    from training_data.repair_curriculum import commands

    check_deadline(deadline)
    filename, build, run = commands(row["language"])
    sandbox = modal.Sandbox.create(
        app=app, image=runtime_image, timeout=90, cpu=1, memory=2048, block_network=True
    )
    try:
        source = observations(row["language"], code, row["cases"])
        writer = sandbox.exec(
            "python",
            "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2])",
            "/tmp/" + filename,
            source,
            timeout=10,
        )
        writer.wait()
        if writer.returncode:
            raise RuntimeError("Sandbox source upload failed")
        stdout = ""
        for command in ([build] if build else []) + [run]:
            process = sandbox.exec(*command, workdir="/tmp", timeout=30)
            stdout, stderr = process.stdout.read(), process.stderr.read()
            process.wait()
            if process.returncode:
                return {
                    "passed": False,
                    "stage": "build" if command == build else "run",
                    "diagnostic": stderr[-1000:],
                }
        expected = [str(n) for _, _, n in row["cases"]]
        return {
            "passed": stdout.splitlines() == expected,
            "stage": "outputs",
            "observed": stdout[-2000:],
            "expected": expected,
        }
    except (modal.exception.ExecTimeoutError, modal.exception.SandboxTimeoutError):
        return {"passed": False, "stage": "timeout"}
    finally:
        sandbox.terminate()


@app.local_entrypoint()
def main(dataset: str, checkpoints: str, output: str = "", limit_per_language: int = 2):
    from trainer.sft_recipe import parse_checkpoints

    named = parse_checkpoints(checkpoints)
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    rows = [json.loads(line) for line in Path(dataset).read_text().splitlines() if line.strip()]
    chosen = select(rows, limit_per_language)
    by_id = {row["id"]: row for row in chosen}
    calibration = {}
    for row in chosen:
        calibration.setdefault(row["language"], row)

    def calibrate(row):
        good, bad = grade(row, row["fixed"], deadline), grade(row, row["broken"], deadline)
        if not good["passed"] or bad["passed"]:
            raise ValueError(f"Grader calibration failed: {row['id']}: {good}, {bad}")
        return row["language"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        calibrated = list(pool.map(calibrate, calibration.values()))
    path = Path(output or f"artifacts/sft/evaluation-{'-'.join(named)}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": dataset,
        "decoding": DECODING,
        "calibrated_languages": calibrated,
        "models": {},
    }
    for name, checkpoint in named.items():
        samples = generate.remote(checkpoint, chosen, deadline)
        with ThreadPoolExecutor(max_workers=4) as pool:
            grades = list(pool.map(lambda s: grade(by_id[s["id"]], s["code"], deadline), samples))
        for sample, result in zip(samples, grades, strict=True):
            sample["grading"] = result
        report["models"][name] = {
            "checkpoint": checkpoint,
            "samples": samples,
            "passed": sum(g["passed"] for g in grades),
            "total": len(grades),
            "eos": sum(s["eos"] for s in samples),
        }
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"model": name, "passed": report["models"][name]["passed"]}), flush=True)
