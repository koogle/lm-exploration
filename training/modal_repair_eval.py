"""Small greedy base/SFT repair diagnostic, graded in disposable Modal Sandboxes."""

import json
from pathlib import Path

import modal

from modal_repair_data import runtime_image
from modal_repair_sft import PARENT, checkpoint_path, image, run_volume

image = image.add_local_file("modal_repair_sft.py", "/project/modal_repair_sft.py").add_local_file(
    "modal_repair_data.py", "/project/modal_repair_data.py"
)
app = modal.App("simple-llm-repair-eval")


@app.function(
    image=image,
    gpu="L40S",
    cpu=4,
    memory=16384,
    timeout=900,
    single_use_containers=True,
    volumes={"/runs": run_volume},
)
def generate(checkpoint, examples, limit_per_language=2):
    import torch
    from model_100m.model import GPT, ModelConfig
    from training_data.repair_sft import prompt
    from training_data.subword import load_frozen

    run_volume.reload()
    saved = torch.load(checkpoint_path(checkpoint), map_location="cpu", weights_only=False)
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
    results, counts = [], {}
    # Different families, not two parameterizations of the same family.
    families = set()
    for row in examples:
        lang = row["language"]
        if row["split"] != "validation" or counts.get(lang, 0) >= limit_per_language:
            continue
        if (lang, row["family"]) in families:
            continue
        families.add((lang, row["family"]))
        counts[lang] = counts.get(lang, 0) + 1
        ids = [1] + tokenizer.encode(prompt(row), add_special_tokens=False).ids
        generated = []
        eos = False
        with torch.inference_mode():
            for _ in range(128):
                x = torch.tensor([ids + generated], device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)[0, -1].float()
                logits[[0, 1, 3]] = -float("inf")
                token = int(logits.argmax())
                if token == 2:
                    eos = True
                    break
                generated.append(token)
        results.append(
            {
                "id": row["id"],
                "language": lang,
                "family": row["family"],
                "code": tokenizer.decode(generated),
                "tokens": len(generated),
                "eos": eos,
            }
        )
    return results


def observations(language, code, cases):
    """Print observations only. Expected answers stay outside the model sandbox."""
    pairs = [(a, b) for a, b, _ in cases]
    if language == "python":
        return code + "\n" + "\n".join(f"print(solve({a}, {b}))" for a, b in pairs)
    if language in ("javascript", "typescript"):
        return code + "\n" + "\n".join(f"console.log(solve({a}, {b}));" for a, b in pairs)
    if language == "bash":
        return "set -eu\n" + code + "\n" + "\n".join(f"solve {a} {b}" for a, b in pairs)
    if language in ("c", "cpp"):
        return (
            "#include <stdio.h>\n"
            + code
            + "\nint main(void) {\n"
            + "\n".join(f'printf("%d\\n", solve({a}, {b}));' for a, b in pairs)
            + "\nreturn 0; }\n"
        )
    if language == "rust":
        return (
            code
            + "\nfn main() {\n"
            + "\n".join(f'println!("{{}}", solve({a}, {b}));' for a, b in pairs)
            + "\n}\n"
        )
    if language == "go":
        return (
            'package main\nimport "fmt"\n'
            + code
            + "\nfunc main() {\n"
            + "\n".join(f"fmt.Println(solve({a}, {b}))" for a, b in pairs)
            + "\n}\n"
        )
    if language == "java":
        return (
            "public class Main {\n"
            + code
            + "\npublic static void main(String[] args) {\n"
            + "\n".join(f"System.out.println(solve({a}, {b}));" for a, b in pairs)
            + "\n}}\n"
        )
    raise ValueError(language)


def grade(row, code):
    from training_data.repair_curriculum import commands

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
def main(dataset: str, checkpoint: str, output: str):
    from concurrent.futures import ThreadPoolExecutor

    rows = [json.loads(line) for line in Path(dataset).read_text().splitlines()]
    by_id = {row["id"]: row for row in rows}
    calibration = {}
    for row in rows:
        lang = row["language"]
        if row["split"] == "validation" and lang not in calibration:
            calibration[lang] = row

    def calibrate(row):
        good, bad = grade(row, row["fixed"]), grade(row, row["broken"])
        if not good["passed"] or bad["passed"]:
            raise ValueError(f"Grader calibration failed: {row['id']}: {good}, {bad}")
        return row["language"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        calibrated = list(pool.map(calibrate, calibration.values()))
    print(json.dumps({"calibrated_languages": calibrated}), flush=True)
    report = {"decoding": "greedy BF16, 128 new tokens, PAD/BOS/UNK suppressed", "models": {}}
    for name, path in (("base", PARENT), ("sft", checkpoint)):
        samples = generate.remote(path, rows)
        with ThreadPoolExecutor(max_workers=4) as pool:
            grades = list(
                pool.map(lambda sample: grade(by_id[sample["id"]], sample["code"]), samples)
            )
        for sample, result in zip(samples, grades):
            sample["grading"] = result
        report["models"][name] = {
            "checkpoint": path,
            "samples": samples,
            "passed": sum(g["passed"] for g in grades),
            "total": len(grades),
            "eos": sum(s["eos"] for s in samples),
        }
        Path(output).write_text(json.dumps(report, indent=2))
        print(
            json.dumps(
                {"model": name, "passed": report["models"][name]["passed"], "total": len(grades)}
            ),
            flush=True,
        )
