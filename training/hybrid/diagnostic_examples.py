"""Token-category summaries and explicitly selected illustrative error contrasts."""

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("folder", type=Path)
parser.add_argument("tokenizer", type=Path)
args = parser.parse_args()
p = args.folder
tokenizer = Tokenizer.from_file(str(args.tokenizer))
z = np.load(p / "baseline-17/examples.npz")
x, y = z["x"], z["y"]
valid = y != -100
loss = {
    v: np.stack([np.load(p / f"{v}-{s}/token_metrics.npz")["ce"] for s in [17, 29]])
    for v in ["baseline", "none", "rope_triton"]
}
means = {v: l.mean(0) for v, l in loss.items()}
meta = json.loads((p / "baseline-17/documents.json").read_text())
classes = np.full(y.shape, "invalid", dtype="<U24")
for token in np.unique(y[valid]):
    text = tokenizer.decode([int(token)], skip_special_tokens=False)
    stripped = text.strip()
    if "\n" in text:
        label = "newline"
    elif text.isspace():
        label = "whitespace"
    elif stripped.isnumeric():
        label = "number"
    elif stripped.replace("_", "").isalnum():
        label = "word_or_identifier"
    elif stripped and all(not ch.isalnum() and not ch.isspace() for ch in stripped):
        label = "punctuation_or_operator"
    else:
        label = "other"
    classes[y == token] = label
summary = {}
for name in np.unique(classes[valid]):
    mask = classes == name
    summary[str(name)] = {
        "count": int(mask.sum()),
        **{v: float(l[mask].mean()) for v, l in means.items()},
    }
examples = {}
for winner, loser in [("baseline", "rope_triton"), ("rope_triton", "baseline")]:
    delta = (means[loser] - means[winner]).copy()
    eligible = valid & ((loss[loser] - loss[winner]).min(0) > 0.5) & (means[winner] < 1.5)
    delta[~eligible] = -np.inf
    chosen = []
    seen = set()
    counts = {}
    for flat in np.argsort(-delta, axis=None):
        row, col = np.unravel_index(flat, y.shape)
        if not np.isfinite(delta[row, col]):
            break
        source = meta[row]["source"]
        if row in seen or counts.get(source, 0) >= 2:
            continue
        chosen.append(
            {
                "source": source,
                "document": meta[row]["document"],
                "row": int(row),
                "position": int(col),
                "context": tokenizer.decode(x[row, max(0, col - 35) : col + 1].tolist()),
                "target": tokenizer.decode([int(y[row, col])], skip_special_tokens=False),
                "target_probability": {v: float(np.exp(-l[row, col])) for v, l in means.items()},
                "loss_difference": float(delta[row, col]),
            }
        )
        seen.add(row)
        counts[source] = counts.get(source, 0) + 1
        if len(chosen) >= 6:
            break
    examples[winner] = chosen
out = {
    "categories": summary,
    "illustrative_examples": examples,
    "selection": "Largest paired loss contrasts with winner CE <1.5 and same-sign margin >0.5 nats in both seeds; two documents per source. Illustrative, not a random sample.",
}
(p / "examples-analysis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps(out, indent=2, ensure_ascii=False))
