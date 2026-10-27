"""Analyze paired document losses and checkpoint diagnostic outputs."""

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(folder):
    variants = ["baseline", "none", "rope_triton"]
    records = {
        v: [json.loads((folder / f"{v}-{s}" / "result.json").read_text()) for s in [17, 29]]
        for v in variants
    }
    assert all(x["status"] == "complete" for rows in records.values() for x in rows)
    assert len({x["sample_sha256"] for rows in records.values() for x in rows}) == 1
    assert len({x["next_training_batch"]["sha256"] for rows in records.values() for x in rows}) == 1
    examples = np.load(folder / "baseline-17/examples.npz")
    x, y = examples["x"], examples["y"]
    valid = y != -100
    metrics = {
        v: [np.load(folder / f"{v}-{s}/token_metrics.npz") for s in [17, 29]] for v in variants
    }
    ce = {v: np.stack([a["ce"] for a in rows]) for v, rows in metrics.items()}
    means = {v: a.mean(0) for v, a in ce.items()}
    names = ["climbmix", "code", "math"]
    weights = np.array([0.7, 0.2, 0.1])
    rng = np.random.default_rng(921)
    summary = {
        "sources": {},
        "paired_differences": {},
        "interventions": {},
        "gradient_groups": {},
        "updates": {},
    }
    for i, name in enumerate(names):
        sl = slice(i * 128, (i + 1) * 128)
        summary["sources"][name] = {
            v: float(means[v][sl].sum() / valid[sl].sum()) for v in variants
        }
    for v in variants[1:]:
        diff = means[v] - means["baseline"]
        source_boot = []
        source_result = {}
        for i, name in enumerate(names):
            sl = slice(i * 128, (i + 1) * 128)
            numer = diff[sl].sum(1)
            denom = valid[sl].sum(1)
            indices = rng.integers(0, 128, (5000, 128))
            boot = numer[indices].sum(1) / denom[indices].sum(1)
            source_boot.append(boot)
            source_result[name] = {
                "mean": float(numer.sum() / denom.sum()),
                "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            }
        mixed = weights @ np.stack(source_boot)
        source_result["weighted"] = {
            "mean": float(sum(weights[i] * source_result[n]["mean"] for i, n in enumerate(names))),
            "ci95": np.quantile(mixed, [0.025, 0.975]).tolist(),
            "seed_deltas": [
                float(
                    sum(
                        weights[i]
                        * (
                            ce[v][s, i * 128 : (i + 1) * 128]
                            - ce["baseline"][s, i * 128 : (i + 1) * 128]
                        ).sum()
                        / valid[i * 128 : (i + 1) * 128].sum()
                        for i in range(3)
                    )
                )
                for s in range(2)
            ],
        }
        summary["paired_differences"][v] = source_result
    for v, rows in records.items():
        keys = rows[0]["interventions"].keys()
        summary["interventions"][v] = {
            k: [r["interventions"][k] - r["interventions"]["unaltered"] for r in rows] for k in keys
        }
        summary["updates"][v] = [
            {
                "gradient_norm": r["fresh_update"]["raw_gradient_norm"],
                "train_ce_change": r["fresh_update"]["train_ce_after"]
                - r["fresh_update"]["train_ce_before"],
                "heldout_ce_change": r["fresh_update"]["heldout_ce_after"]
                - r["fresh_update"]["heldout_ce_before"],
                "predicted_heldout_ce_change": sum(
                    a["heldout_gradient_dot_update"] for a in r["fresh_update"]["groups"].values()
                ),
            }
            for r in rows
        ]
        grouped = {}
        for r in rows:
            for name, g in r["heldout_gradient_groups"].items():
                grouped.setdefault(name, []).append(
                    {
                        "grad_rms": (g["gradient_sq"] / g["parameters"]) ** 0.5,
                        "relative_gradient": (g["gradient_sq"] / max(g["weight_sq"], 1e-30)) ** 0.5,
                        "gradient_moment_cosine": g["gradient_dot_moment_direction"]
                        / max((g["gradient_sq"] * g["moment_direction_sq"]) ** 0.5, 1e-30),
                    }
                )
        summary["gradient_groups"][v] = grouped
    summary["position_bins"] = {}
    for start, end in [(0, 4), (4, 16), (16, 64), (64, 256), (256, 1024)]:
        summary["position_bins"][f"{start}:{end}"] = {
            v: float(means[v][:, start:end].sum() / valid[:, start:end].sum()) for v in variants
        }
    repeated = np.full(y.shape, -1, dtype=np.int32)
    for row in range(len(x)):
        seen = {}
        for col in range(1024):
            seen[int(x[row, col])] = col
            if int(y[row, col]) in seen:
                repeated[row, col] = col + 1 - seen[int(y[row, col])]
    summary["repeat_bins"] = {}
    for name, mask in [
        ("unseen", repeated == -1),
        ("1-4", (repeated > 0) & (repeated <= 4)),
        ("5-32", (repeated > 4) & (repeated <= 32)),
        ("33-128", (repeated > 32) & (repeated <= 128)),
        ("129+", (repeated > 128)),
    ]:
        mask &= valid
        summary["repeat_bins"][name] = {
            "count": int(mask.sum()),
            **{v: float(means[v][mask].mean()) for v in variants},
        }
    (folder / "analysis.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.folder), indent=2))
