"""Build the open-model frontier dataset and page from the collected source files.

Run from training/ with the source files downloaded from the HF dataset
`jfrick/lm-exploration-results` (folder `training/references/open-model-frontier/`):

    python references/open-model-frontier/build.py --sources <folder> --out <folder>

Sources, each a JSON list collected by hand from model cards, papers and leaderboards
(every number with its source URL):

- `small.json`, `mid.json`, `large.json`: one entry per model variant, with
  `scores` keyed by benchmark (total parameters <5B, 5-150B, >150B).
- `agentic_cards.json`: agentic scores from the cards of those same entries.
- `agentic_swe.json`, `agentic_tb.json`: SWE-bench and Terminal-Bench results, one
  row per (model, benchmark, setting), including research fine-tunes and independent
  Artificial Analysis runs.

Writes `models.json` (one entry per model with every benchmark), `pairs.json`
(same-paper scores of a base model before and after agentic training) and
`open-model-frontier.html` (`page.html` with both embedded).
"""

import argparse
import collections
import json
import re
from pathlib import Path

AGENTIC = {
    "swe_bench_verified",
    "swe_bench_pro",
    "swe_bench_multilingual",
    "terminal_bench_1",
    "terminal_bench_2",
    "terminal_bench_2_1",
    "terminal_bench_hard",
}
PAIRED = {"swe_bench_verified", "terminal_bench_1", "terminal_bench_2", "terminal_bench_2_1"}
ALIASES = {"terminal_bench_2.1": "terminal_bench_2_1"}
FIELDS = (
    "name", "org", "release", "total_b", "active_b", "variant", "thinking",
    "scores", "source", "notes", "research", "ssrc", "snote",
)  # fmt: skip
# Settings that are not one attempt per task on the whole benchmark.
NOT_PASS_AT_1 = re.compile(
    r"best@|best-of|best of|verifier|tts|^pass@([2-9]|\d\d)|expert|best over evaluated",
    re.IGNORECASE,
)
SUBSET = re.compile(r"verified mini|231-instance|subset", re.IGNORECASE)
AGENT_TRAINING = ("agentic SFT", "agentic RL", "SFT+RL")


def norm(name):
    """A model name for matching across sources (AA says reasoning, cards say thinking)."""
    key = re.sub(r"[^a-z0-9]", "", (name or "").lower().replace("+", "plus"))
    return key.replace("nonreasoning", "nonthinking").replace("reasoning", "thinking")


def benchmark(row):
    return ALIASES.get(row["benchmark"], row["benchmark"])


def pass_at_1(row):
    """One attempt per task on the whole benchmark, by the model alone."""
    if not isinstance(row.get("score"), (int, float)) or not row.get("total_b"):
        return False
    if SUBSET.search(" ".join(str(row.get(k) or "") for k in ("name", "attempts", "notes"))):
        return False
    return not (
        NOT_PASS_AT_1.search(row["name"]) or NOT_PASS_AT_1.search(row.get("attempts") or "")
    )


def setting(row):
    return "; ".join(str(row[k]) for k in ("scaffold", "attempts", "notes") if row.get(k))


def models(base, cards, rows):
    """Base entries with card agentic scores, then research and leaderboard rows.

    A score already on an entry's own card is kept. Otherwise, per (model, benchmark),
    an independent Artificial Analysis run wins over other settings, then the highest.
    """
    for entry in base:
        entry.update(research=False, ssrc={}, snote={})
    by_key = {norm(e["name"]): e for e in base}
    by_name = {e["name"]: e for e in base}

    def put(entry, name, score, source, note):
        if name in entry["scores"] and name not in entry["snote"]:
            return
        entry["scores"][name] = score
        entry["ssrc"][name] = source
        entry["snote"][name] = note

    for card in cards:
        name = benchmark(card)
        if name in AGENTIC and isinstance(card["score"], (int, float)):
            note = " · ".join(str(card[k]) for k in ("scaffold", "notes") if card.get(k))
            put(by_name[card["name"]], name, card["score"], card["source"], note)

    groups = collections.defaultdict(list)
    for row in rows:
        if benchmark(row) in AGENTIC and pass_at_1(row):
            groups[(norm(row["name"]), benchmark(row))].append(row)
    for (key, name), options in groups.items():
        independent = [
            r for r in options
            if "AA" in (r.get("scaffold") or "") or "artificialanalysis" in r["source"]
        ]  # fmt: skip
        row = max(independent or options, key=lambda r: r["score"])
        note = setting(row)
        if len(options) > 1:
            how = "independent AA run preferred" if independent else "highest"
            note += f" (chosen from {len(options)} reported settings, {how})"
        entry = by_key.get(key)
        if entry is None:
            title = row["name"]
            thinking = bool(re.search(r"think|reason", title, re.IGNORECASE)) and not re.search(
                r"non[- ]?(think|reason)|no think", title, re.IGNORECASE
            )
            entry = {
                "name": title,
                "org": row.get("org") or "",
                "release": row.get("release") or "",
                "total_b": row["total_b"],
                "active_b": row.get("active_b") or row["total_b"],
                "variant": "reasoning" if thinking else "instruct",
                "thinking": thinking,
                "scores": {},
                "source": row["source"],
                "notes": f"Base model: {row.get('base_model') or 'n/a'}. "
                f"Training: {row.get('training')}.",
                "research": row.get("training") in AGENT_TRAINING,
                "ssrc": {},
                "snote": {},
            }
            by_key[key] = entry
            base.append(entry)
        put(entry, name, row["score"], row["source"], note)
    for entry in base:
        entry["release"] = str(entry.get("release") or "")
    return [{k: e[k] for k in FIELDS} for e in base]


def pairs(rows):
    """Same-source scores of a base model and its agentic fine-tunes of similar size."""
    sources = collections.defaultdict(list)
    for row in rows:
        if benchmark(row) in PAIRED and pass_at_1(row):
            sources[(row["source"], benchmark(row))].append(row)
    out, seen = [], set()
    for (source, name), group in sources.items():
        bases = [r for r in group if r["training"] in ("none", "vendor")]
        for tuned in group:
            if tuned["training"] in ("none", "vendor") or (tuned["name"], name) in seen:
                continue
            key = norm(tuned.get("base_model"))
            matches = [
                r for r in bases
                if key
                and (norm(r["name"]).startswith(key[:10]) or key.startswith(norm(r["name"])[:10]))
                and 0.75 < r["total_b"] / tuned["total_b"] < 1.34
            ]  # fmt: skip
            if not matches:
                continue
            seen.add((tuned["name"], name))
            out.append(
                {
                    "bench": name,
                    "base": matches[0]["name"],
                    "before": matches[0]["score"],
                    "after": tuned["name"],
                    "score": tuned["score"],
                    "size": tuned["total_b"],
                    "training": tuned["training"],
                    "source": source,
                    "setting": "; ".join(
                        str(tuned[k]) for k in ("scaffold", "attempts") if tuned.get(k)
                    ),
                }
            )
    return out


def embed(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    def load(name):
        return json.loads((args.sources / name).read_text())

    base = load("small.json") + load("mid.json") + load("large.json")
    rows = load("agentic_swe.json") + load("agentic_tb.json")
    built, paired = models(base, load("agentic_cards.json"), rows), pairs(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "models.json").write_text(json.dumps(built, ensure_ascii=False) + "\n")
    (args.out / "pairs.json").write_text(json.dumps(paired, ensure_ascii=False) + "\n")
    page = (Path(__file__).parent / "page.html").read_text()
    page = page.replace("__DATA__", embed(built)).replace("__PAIRS__", embed(paired))
    (args.out / "open-model-frontier.html").write_text(page)
    counts = collections.Counter(k for e in built for k in e["scores"] if k in AGENTIC)
    print(json.dumps({"models": len(built), "pairs": len(paired), "agentic": counts}, indent=1))


if __name__ == "__main__":
    main()
