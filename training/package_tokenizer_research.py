"""Package measured tokenizer artifacts for a Hugging Face model repository."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

SIZES = (64, 128, 256, 512, 1024, 2048)
VOCABS = (16384, 32768, 50000)
SPECIAL = {f"{name}_token": f"<|{name}|>" for name in ("pad", "bos", "eos", "unk")}


def package(results, artifacts, output, repo_id):
    results, artifacts, output = Path(results), Path(artifacts), Path(output)
    measurements = json.loads((results / "measurements.json").read_text())
    indexed = {(r["size_mib"], r["requested_vocab_size"]): r for r in measurements}
    if set(indexed) != {(s, v) for s in SIZES for v in VOCABS} or len(measurements) != 18:
        raise ValueError("Publication requires all 18 measured candidates")
    evaluation_hash = hashlib.sha256(
        (results / "evaluation-corpus-manifest.json").read_bytes()
    ).hexdigest()
    for (size, vocab), result in indexed.items():
        if (
            not result["correctness_passed"]
            or result["evaluation_manifest_sha256"] != evaluation_hash
        ):
            raise ValueError(
                "Publication requires correct candidates and one audited evaluation corpus"
            )
        path = artifacts / "variants" / f"{size}mib-{vocab}" / "tokenizer.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != result["tokenizer_sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {path}")
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "clean_up_tokenization_spaces": False,
        "model_input_names": ["input_ids", "attention_mask"],
        "add_bos_token": False,
        "add_eos_token": False,
        **SPECIAL,
    }
    for size, vocab in indexed:
        name = f"{size}mib-{vocab}"
        destination = output / "variants" / name
        destination.mkdir(parents=True)
        shutil.copyfile(
            artifacts / "variants" / name / "tokenizer.json", destination / "tokenizer.json"
        )
        (destination / "tokenizer_config.json").write_text(json.dumps(config, indent=2) + "\n")
        (destination / "special_tokens_map.json").write_text(json.dumps(SPECIAL, indent=2) + "\n")
    default = "2048mib-32768"
    for path in (output / "variants" / default).iterdir():
        shutil.copyfile(path, output / path.name)
    research = output / "research"
    research.mkdir()
    for pattern in ("*.png", "*.svg", "*.pdf"):
        for path in results.glob(pattern):
            shutil.copyfile(path, research / path.name)
    for name in (
        "REPORT.md",
        "measurements.json",
        "corpus-manifest.json",
        "evaluation-corpus-manifest.json",
        "execution-history.json",
        "modal-smoke.json",
    ):
        shutil.copyfile(results / name, research / name)
    recipe = json.loads((results / "corpus-manifest.json").read_text())["identity"]["recipe"]
    (output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    manifest = {
        "default_variant": default,
        "selection": "32K is a practical starting point balancing compression and vocabulary parameter overhead; no downstream optimum has been established.",
        "variants": {f"{s}mib-{v}": indexed[(s, v)]["tokenizer_sha256"] for s, v in indexed},
        "evaluation_manifest_sha256": evaluation_hash,
    }
    (output / "publication-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    table = ["| Fitting MiB | Vocabulary | Bytes/token | Variant |", "|---:|---:|---:|---|"]
    for size in SIZES:
        for vocab in VOCABS:
            r = indexed[(size, vocab)]
            name = f"{size}mib-{vocab}"
            table.append(
                f"| {size} | {vocab:,} | {r['metrics']['bytes_per_token']:.4f} | [`{name}`](variants/{name}) |"
            )
    card = f'''---
library_name: transformers
language:
- en
tags:
- tokenizer
- byte-level-bpe
- code
- scientific-text
- from-scratch
---

# Scholarly/code tokenizer learning curve

18 byte-level BPE tokenizers trained from scratch on Modal: 64, 128, 256, 512,
1,024 and 2,048 MiB of text, each at 16,384, 32,768 and 50,000 vocabulary entries.
This is a tokenizer resource, with no language-model weights.

The default is **32,768 entries trained on 2 GiB**, a practical starting point
balancing compression and embedding/output size. This is not a demonstrated
model-quality optimum. All 18 variants are included for reproducible comparisons.
The existing scholarly-code-v1 tokenizer in GitHub has not been replaced.

## Usage

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
text = "def square(x):\\n    return x ** 2\\n"
ids = tokenizer.encode(text, add_special_tokens=False)
assert tokenizer.decode(ids, clean_up_tokenization_spaces=False) == text

# Choose any measured vocabulary and fitting size:
tokenizer_50k = AutoTokenizer.from_pretrained(
    "{repo_id}", subfolder="variants/2048mib-50000"
)
```

Tested with `transformers==4.57.6` and `tokenizers==0.22.2`. No custom remote code
is required. The Rust `tokenizers.Tokenizer.from_file("tokenizer.json")` API also
works. BOS/EOS are not inserted automatically; training pipelines must add document
boundaries explicitly. Special IDs: PAD=0, BOS=1, EOS=2, UNK=3. Decode with
`skip_special_tokens=False` when exact round trips including special literals matter.
Byte coverage avoids unknown tokens for ordinary valid UTF-8 text. No maximum
model context is implied by this tokenizer.

## Results

![Learning curve](research/learning-curve.png)

All variants were evaluated on the same 32 MiB source-weighted held-out corpus,
with parent separation and a 16 KiB evaluation parent cap. Higher bytes/token means
fewer tokens for identical text. The [research report](research/REPORT.md) and
[PDF](research/research-report.pdf) include per-source curves, paired bootstrap
intervals, vocabulary utilization, fragmentation probes, fit time and peak memory.

{chr(10).join(table)}

## Data and method

All 12 pinned source inventories in [recipe.json](recipe.json) were eligible:
FineWeb-Edu 10%, Python code 30%, FineMath 20%, peS2o 10%, PubMed 3%, arXiv 2%,
Pressbooks 3%, LibreTexts 3%, DOAB 4%, Gutenberg 5%, pre-1929 books 7%, and
Library of Congress 3%. Percentages apply to UTF-8 bytes. This uses samples from
all sources, not every record in those datasets. Source revisions and full shard
inventories are pinned in the manifests. Raw source text is not distributed here;
source datasets retain their respective licenses and provenance.

Fitting uses nested samples, a seeded shard permutation and sequential records,
64 KiB per document and 256 KiB per parent. Evaluation initially had sparse book
coverage and was rebuilt with a 16 KiB parent cap before interpreting curves;
all unchanged tokenizer weights were rescored. Normalized exact-chunk deduplication
is applied, including fitting/evaluation overlap checks. Byte-level BPE uses the
same fixed pretokenization rules, a full byte alphabet, min frequency 2 and maximum
learned token length 32. No normalization or automatic prefix space is applied.

## Limitations and reproducibility

One sampling seed; no global reservoir sampling, fuzzy deduplication guarantee,
or cross-source parent reconciliation. Bootstrap intervals measure evaluation
uncertainty, not training-seed variance. Compression is not language-model quality.
Compare downstream bits per byte under matched parameter and compute budgets
before selecting a final vocabulary. At width 768, 32K adds 12.58M tied embedding
parameters over 16K; 50K adds 25.82M. Untied input/output matrices double these deltas.

[Implementation and PR](https://github.com/koogle/lm-exploration/pull/8).
[Publication manifest](publication-manifest.json) pins every tokenizer hash.
The `research/` manifests and measurements record corpus hashes, source coverage,
correctness, resource measurements and original result hashes. Candidate tokenizers
and full parent-level results are also preserved on the original Modal volumes.
'''
    (output / "README.md").write_text(card)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results")
    parser.add_argument("artifacts")
    parser.add_argument("output")
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    package(args.results, args.artifacts, args.output, args.repo_id)
