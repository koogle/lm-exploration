"""Build tokenizer-independent repetition comparisons from retained raw outputs."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

HERE = Path(__file__).parent


def repetition(seq, n):
    grams = [tuple(seq[i : i + n]) for i in range(max(0, len(seq) - n + 1))]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def sample_metrics(row):
    text = re.sub(r"\s+", " ", row["completion"]).strip().lower()
    words = re.findall(r"\b\w+\b", text)
    return {
        "token_rep4": repetition(row["generation_ids"], 4),
        "word_rep4": repetition(words, 4),
        "char_rep16": repetition(text, 16),
        "chars": len(text),
        "words": len(words),
        "tokens": row["generated_tokens"],
        "char256_rep16": repetition(text[:256], 16) if len(text) >= 256 else None,
        "word32_rep4": repetition(words[:32], 4) if len(words) >= 32 else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bf16", action="store_true", help="Rebuild retained initial BF16 comparison"
    )
    args = parser.parse_args()
    suffix = "-bf16" if args.bf16 else ""

    def output(name):
        p = Path(name)
        return HERE / (p.stem + suffix + p.suffix)

    data = json.loads(output("external-results.json").read_text())
    for model in data["models"]:
        assert len(model["samples"]) == 60
        assert len({(r["prompt"], r["mode"], r["seed"]) for r in model["samples"]}) == 60
        for sample in model["samples"]:
            assert sample["prompt"] in data["prompts"]
            assert len(sample["generation_ids"]) == sample["generated_tokens"]
            assert 1 <= sample["generated_tokens"] <= 128
            assert sample["generated_tokens"] == 128 or sample["ended_with_eos"]
    previous = json.loads(
        (HERE.parent / "repetition/previous-comparison-20261018.json").read_text()
    )["checkpoints"][-1]
    low = json.loads((HERE.parent / "repetition/low-temperature-20261018.json").read_text())[
        "checkpoints"
    ][-1]
    desired = {m["name"] for m in data["modes"]}
    ours = [r for r in previous["samples"] if r["mode"] in desired] + [
        r for r in low["samples"] if r["mode"] in desired and r["mode"] != "greedy"
    ]
    if not args.bf16:
        own_raw = json.loads((HERE / "simple-llm-fp32.json").read_text())
        checkpoint = own_raw["checkpoints"][0]
        assert own_raw["prompts"] == data["prompts"]
        assert checkpoint["revision"] == previous["revision"]
        assert checkpoint["trained_tokens"] == previous["trained_tokens"]
        assert own_raw["cuda_matmul_allow_tf32"] is False
        assert own_raw["cudnn_allow_tf32"] is False
        assert own_raw["dtype"] == "FP32 weights and forward, TF32 disabled"
        ours = []
        for original in checkpoint["samples"]:
            row = dict(original)
            mode = row["mode"]
            pi = data["prompts"].index(row["prompt"])
            offset = row["seed"] - pi
            if mode == "topk50_t0.1":
                row["mode"] = f"topk50_t0.1_seed{offset}"
            elif mode == "topk50_t0.8":
                row["mode"] = "sampled_08_seed1" if offset == 1000 else "sampled_08_seed2"
            elif mode == "baseline_t1_p95":
                row["mode"] = "sampled_10_topp95"
            elif mode != "greedy":
                continue
            assert offset in (1000, 2000)
            ours.append(row)
    assert len(ours) == 60
    models = [
        {
            "name": "simple-llm-124m-5b",
            "repo": "jfrick/scholarly-code-modern-gpt-124m-step2",
            "revision": previous["revision"],
            "trained_tokens": previous["trained_tokens"],
            "precision": "BF16 autocast" if args.bf16 else "FP32 forward; TF32 disabled",
            "samples": ours,
        },
        *data["models"],
    ]

    def mode_group(name):
        if name.startswith("topk50_t0.1"):
            return "topk50_t0.1"
        if name.startswith("sampled_08"):
            return "sampled_08"
        return name

    indexed = [
        {
            (mode_group(r["mode"]), r["prompt"], r["seed"]): re.sub(r"\s+", " ", r["completion"])
            .strip()
            .lower()
            for r in m["samples"]
        }
        for m in models
    ]
    paired = []
    for width in (128, 256):
        common = set.intersection(
            *[{k for k, text in rows.items() if len(text) >= width} for rows in indexed]
        )
        for mode in ("greedy", "topk50_t0.1", "sampled_08", "sampled_10_topp95"):
            keys = sorted(k for k in common if k[0] == mode)
            for model, rows in zip(models, indexed, strict=True):
                paired.append(
                    {
                        "model": model["name"],
                        "mode": mode,
                        "characters": width,
                        "n": len(keys),
                        "normalized_char_rep16": mean(repetition(rows[k][:width], 16) for k in keys)
                        if keys
                        else None,
                        "matched_prompt_seeds": [{"prompt": k[1], "seed": k[2]} for k in keys],
                    }
                )
    output("paired-prefix-summary.json").write_text(json.dumps(paired, indent=2) + "\n")
    summary = []
    completions = [
        "# All retained external base-model completions",
        "",
        "128 generated-token cap; EOS may stop sooner. Text is JSON-escaped to preserve exact output.",
        "",
    ]
    for model in models:
        groups = defaultdict(list)
        for row in model["samples"]:
            key = row["mode"]
            key = (
                "topk50_t0.1"
                if key.startswith("topk50_t0.1")
                else "sampled_08"
                if key.startswith("sampled_08")
                else key
            )
            groups[key].append((row, sample_metrics(row)))
            if model["name"] != "simple-llm-124m-5b":
                completions += [
                    f"## {model['name']} / {row['mode']} / seed {row['seed']}",
                    "",
                    f"Prompt: {json.dumps(row['prompt'])}",
                    "",
                    "```json",
                    json.dumps(row["completion"], ensure_ascii=False),
                    "```",
                    "",
                ]
        for mode, items in groups.items():
            metrics = [m for _, m in items]
            row = {
                "model": model["name"],
                "mode": mode,
                "n": len(items),
                "eos": sum(s["ended_with_eos"] for s, _ in items),
            }
            for key in metrics[0]:
                values = [m[key] for m in metrics if m[key] is not None]
                row[key] = mean(values) if values else None
                if key in ("char256_rep16", "word32_rep4"):
                    row[key + "_n"] = len(values)
            summary.append(row)
    output("summary.json").write_text(
        json.dumps(
            {
                "models": [{k: v for k, v in m.items() if k != "samples"} for m in models],
                "groups": summary,
            },
            indent=2,
        )
        + "\n"
    )
    output("external-completions.md").write_text("\n".join(completions))
    lines = [
        "| Model | Decoder | N | Token4 | Word4 | Char16 | First256char | Coverage | EOS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['model']} | {r['mode']} | {r['n']} | {r['token_rep4']:.1%} | {r['word_rep4']:.1%} | {r['char_rep16']:.1%} | {r['char256_rep16']:.1%} | {r['char256_rep16_n']}/{r['n']} | {r['eos']} |"
        )
    output("metrics-table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
