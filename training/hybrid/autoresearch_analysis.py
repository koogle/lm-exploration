"""Evidence labels for paired runs; never turn partial curves into final winners."""

import json
from pathlib import Path

from .autoresearch_specs import VARIANTS


def paired(candidate, baseline):
    by_step = {r["step"]: r for r in baseline["curve"]}
    common = [r for r in candidate["curve"] if r["step"] in by_step]
    if not common:
        return None
    row = common[-1]
    control = by_step[row["step"]]
    return {
        "step": row["step"],
        "tokens": row["tokens"],
        "candidate_ce": row["validation"]["ce"],
        "baseline_ce": control["validation"]["ce"],
        "delta": row["validation"]["ce"] - control["validation"]["ce"],
        "candidate_tokens_per_second": row["tokens_per_second"],
        "baseline_tokens_per_second": control["tokens_per_second"],
    }


def verdict(pairs, complete):
    if len(pairs) != 2:
        return "insufficient_paired_seeds"
    if not complete:
        return "interim_only"
    deltas = [p["delta"] for p in pairs]
    if max(deltas) < 0 and sum(deltas) / 2 < -0.003:
        return "improved_proxy_ce"
    if min(deltas) > 0 and sum(deltas) / 2 > 0.003:
        return "worse_proxy_ce"
    return "inconclusive_proxy_ce"


def analyze(folder):
    folder = Path(folder)

    def read(path):
        p = folder / path
        return json.loads(p.read_text()) if p.exists() else None

    result = {
        "arms": [],
        "scope": "Proxy CE evidence only; no target-scale or long-context winner is inferred.",
    }
    for variant in VARIANTS:
        if variant == "baseline":
            continue
        qualification = read(f"qualification/{variant}.json")
        arm = {
            "variant": variant,
            "qualification": qualification["status"] if qualification else "pending",
        }
        for stage in ("screen", "confirm"):
            pairs = []
            complete = True
            for seed in (17, 29):
                candidate = read(f"{stage}/{variant}-{seed}/metrics.json")
                baseline = read(f"{stage}/baseline-{seed}/metrics.json")
                if candidate is None or baseline is None:
                    complete = False
                    continue
                comparison = paired(candidate, baseline)
                if comparison:
                    pairs.append({"seed": seed, **comparison})
                complete &= candidate["status"] == "complete" and baseline["status"] == "complete"
                if comparison:
                    complete &= (
                        comparison["step"]
                        == candidate["planned_steps"]
                        == baseline["planned_steps"]
                    )
            arm[stage] = {"pairs": pairs, "verdict": verdict(pairs, complete)}
        if qualification and "benchmark" in qualification:
            arm["layer_slowdown"] = qualification["benchmark"]["slowdown"]
        result["arms"].append(arm)
    (folder / "analysis.json").write_text(json.dumps(result, indent=2))
    return result
