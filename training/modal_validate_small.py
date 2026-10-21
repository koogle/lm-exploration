"""Evaluate the completed 9.85M checkpoint; no training updates."""

import json
from pathlib import Path

import modal

from distributed_training.image import image

app = modal.App("lm-exploration-validate-modern-10m")
data_volume = modal.Volume.from_name("lm-exploration-mixture-data")
run_volume = modal.Volume.from_name("lm-exploration-training")


@app.function(
    image=image,
    gpu="L40S",
    cpu=8,
    memory=16384,
    timeout=1800,
    retries=0,
    max_containers=1,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def validate(
    device="cuda",
    checkpoint_path="/runs/scholarly-modern-10m-5b-20261017-s02/step-000072805.pt",
    manifest_override="",
    database_override="",
    output_path="/runs/scholarly-modern-10m-5b-validation-20261018.json",
    local=False,
):
    import math
    import sqlite3
    import zlib

    import model_100m.model
    import torch

    torch.set_num_threads(4)
    from model_100m.model import GPT, ModelConfig
    from torch.nn import functional as F
    from training_data.subword import digest, load_frozen

    checkpoint = Path(checkpoint_path)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["step"] == 72805 and saved["tokens"] == 4999934057
    assert digest(model_100m.model.__file__) == saved["source_sha256"]["model"]
    cfg = {k: saved["config"][k] for k in ("n_layer", "n_head", "n_embd")}
    cfg.update(vocab_size=32768, block_size=2048, architecture_version=2)
    torch.manual_seed(42)
    model = GPT(ModelConfig(**cfg)).to(device).eval()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(saved["model"], strict=True)
    assert sum(p.numel() for p in model.parameters()) == 9849536
    assert sum(saved["data"]["loss_token_exposures"].values()) == saved["tokens"]
    assert all(torch.isfinite(p).all() for p in model.parameters())
    manifest_path = Path(
        "/data/scholarly-code/scholarly-small-5b-raw-20261017/snapshots/scholarly-modern-10m-5b-20261017/manifest.json"
    )
    if manifest_override:
        manifest_path = Path(manifest_override)
    assert digest(manifest_path) == saved["data_identity"]["manifest_sha256"]
    manifest = json.loads(manifest_path.read_text())
    dbpath = (
        Path(database_override)
        if database_override
        else manifest_path.parent / manifest["database"]["path"]
    )
    assert digest(dbpath) == manifest["database"]["sha256"]
    db = sqlite3.connect(f"{dbpath.as_uri()}?mode=ro&immutable=1", uri=True)
    tokenizer = load_frozen()
    exposures = {}
    for name, source in manifest["sources"].items():
        maximum = 2 * source["splits"]["train"]["loss_tokens"]
        exposures[name] = {
            "trained_tokens": saved["data"]["loss_token_exposures"][name],
            "unique_train_tokens": source["splits"]["train"]["loss_tokens"],
            "passes": saved["data"]["loss_token_exposures"][name]
            / source["splits"]["train"]["loss_tokens"],
        }
        assert exposures[name]["trained_tokens"] <= maximum
    blocks = {}
    for name in manifest["sources"]:
        parents = set()
        selected = []
        for key, parent, raw in db.execute(
            'SELECT hash,parent,text FROM documents WHERE source=? AND split="validation" AND conflict=0 ORDER BY hash',
            (name,),
        ):
            if parent in parents:
                continue
            parents.add(parent)
            ids = tokenizer.encode(zlib.decompress(raw).decode(), add_special_tokens=False).ids
            tokens = [1, *ids, 2]
            for offset in range(0, min(len(tokens) - 1, 4 * 2048), 2048):
                x = tokens[offset : offset + 2048]
                y = tokens[offset + 1 : offset + 2049]
                count = len(y)
                selected.append(
                    (x[:count] + [0] * (2048 - count), y + [-100] * (2048 - count), count)
                )
            if len(parents) >= 32:
                break
        blocks[name] = {"parents": len(parents), "rows": selected}
        print(
            json.dumps({"source": name, "parents": len(parents), "windows": len(selected)}),
            flush=True,
        )

    def losses(weights):
        model.load_state_dict(weights, strict=True)
        result = {}
        with torch.inference_mode():
            for name, selection in blocks.items():
                total = 0.0
                count = 0
                for offset in range(0, len(selection["rows"]), 8):
                    rows = selection["rows"][offset : offset + 8]
                    x = torch.tensor([r[0] for r in rows], device=device)
                    y = torch.tensor([r[1] for r in rows], device=device)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                        loss = F.cross_entropy(
                            model(x).flatten(0, 1), y.flatten(), reduction="sum", ignore_index=-100
                        )
                    total += float(loss)
                    count += sum(r[2] for r in rows)
                ce = total / count
                result[name] = {
                    "loss": ce,
                    "perplexity": math.exp(ce),
                    "tokens": count,
                    "parents": selection["parents"],
                    "windows": len(selection["rows"]),
                }
        return {
            "per_source": result,
            "mixture_weighted_loss": sum(
                result[n]["loss"] * s["weight_percent"] / 100
                for n, s in manifest["sources"].items()
            ),
        }

    trained = losses(saved["model"])
    untrained = losses(initial)
    model.load_state_dict(saved["model"], strict=True)
    prompts = [
        "The solar system consists of",
        "In mathematics, a prime number is",
        "The purpose of an experiment is to",
        "Once upon a time,",
        "def add(a, b):\n    ",
        "def factorial(n):\n    ",
        "import json\n\ndef read_config(path):\n    ",
        "The capital of France is",
        "2 + 2 =",
        "A language model predicts",
    ]
    samples = []
    with torch.inference_mode():
        for index, prompt in enumerate(prompts):
            for mode in ("greedy", "sampled"):
                torch.manual_seed(1000 + index)
                ids = [1] + tokenizer.encode(prompt, add_special_tokens=False).ids
                generated = []
                for _ in range(128):
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                        logits = model(torch.tensor([ids[-2048:]], device=device))[0, -1].float()
                    if mode == "greedy":
                        token = int(logits.argmax())
                    else:
                        values, indices = torch.topk(logits / 0.8, 50)
                        token = int(indices[torch.multinomial(values.softmax(-1), 1)])
                    ids.append(token)
                    generated.append(token)
                    if token == 2:
                        break
                samples.append(
                    {
                        "prompt": prompt,
                        "mode": mode,
                        "seed": 1000 + index,
                        "completion": tokenizer.decode(generated, skip_special_tokens=True),
                        "generated_tokens": len(generated),
                    }
                )
    db.close()
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "step": saved["step"],
        "trained_tokens": saved["tokens"],
        "parameters": 9849536,
        "strict_load": True,
        "finite_parameters": True,
        "token_accounting_verified": True,
        "per_source_two_pass_limit_verified": True,
        "source_exposures": exposures,
        "evaluation": "First 32 distinct validation parents per source by content hash; at most four 2048-token windows per parent. Fixed, non-random exploratory validation; not a sealed benchmark.",
        "trained": trained,
        "random_initialization_seed42": untrained,
        "samples": samples,
        "data_manifest_sha256": digest(manifest_path),
        "pending_data_audits": manifest["pending"],
        "torch_version": str(torch.__version__),
        "gpu": torch.cuda.get_device_name() if device == "cuda" else "CPU",
        "evaluation_device": device,
    }
    out = Path(output_path)
    out.write_text(json.dumps(report, indent=2) + "\n")
    if not local:
        run_volume.commit()
    return report


@app.local_entrypoint()
def main(cpu: bool = False):
    report = validate.with_options(gpu=None, cpu=4).remote("cpu") if cpu else validate.remote()
    Path("results/scholarly-modern-10m-5b-validation-20261018.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "parameters": report["parameters"],
                "tokens": report["trained_tokens"],
                "trained_loss": report["trained"]["mixture_weighted_loss"],
                "untrained_loss": report["random_initialization_seed42"]["mixture_weighted_loss"],
            }
        )
    )
