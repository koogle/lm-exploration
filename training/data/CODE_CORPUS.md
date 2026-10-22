# Multilingual code collection and readiness audit

CPU-only tools for adding permissive-license source code in many languages to a raw
corpus, then auditing the result before training. The code comes from pinned
`codeparrot/github-code-clean` (`multilingual_code.CODE_REVISION`).

| Module | Role |
| --- | --- |
| [`multilingual_code.py`](multilingual_code.py) | Row filter: 29 languages; MIT/Apache-2.0/BSD/ISC; 64 B–4 MB; no vendored or generated files. Also the task family (tests, examples/algorithms, tooling, implementation) and repository split |
| [`parallel_code.py`](parallel_code.py) | Exclusion ledger from prior corpora, disjoint shard partitions per worker, idempotent global merge |
| [`lexical_screen.py`](lexical_screen.py) | Whitespace-normalized hashes and 13-word shingle matching |
| [`readiness_audit.py`](readiness_audit.py) | Read-only audit across databases, with capacity and a stratified frozen held-out set |

## Collect

From `training/`, inside CPU jobs that mount the corpus volume:

```python
from training_data.parallel_code import build_exclusions, merge_partitions, prepare_partition

identity = build_exclusions([PRIOR / "raw.sqlite"], BASE)  # once
prepare_partition(
    WORKERS / f"w{i}", BASE / "exclusions.sqlite", identity, i, N, TARGET
)  # per worker
merge_partitions(CENTRAL / "raw.sqlite", [WORKERS / f"w{i}/raw.sqlite" for i in range(N)])
```

- **Excluded from collection:** every document hash in the prior corpora, every whitespace-normalized copy of their code, and every repository that has held-out documents.
- **Splits:** each new repository gets a split from the same bucket as the scholarly corpus's `code_python` source (parents are `code_python:owner/repo`), so a repository is held out consistently. Test-bucket repositories are skipped.
- **Workers:** each worker resumes from its cursor and refuses a changed identity.
- **Merge:** the merge writes one central database. A normalized duplicate that appears in another split marks the earlier copy as a conflict, and a replayed merge adds nothing.

## Audit

`readiness_audit.audit([(label, path, fresh), ...], OUT, *benchmark_index(OUT))` scans all
databases into `OUT/audit.sqlite` without changing them and writes `report.json` with:

- eligible training capacity per source, after normalized dedup across databases;
- held-out leakage removal by text and hash;
- blocked parents: HumanEval/MBPP 13-gram hits, conflicts, missing parents and multi-split
  parents. Code repositories are matched across code sources and letter case.

It also writes `heldout-windows.jsonl.gz`: at most 256 parents per stratum (for example
`code/rust/tests`), with the first and last `context + 1` tokens of one document each.
This is the frozen held-out window format the trainer evaluates. `resume=True` reuses the
scan and recomputes the derived tables. The screen is lexical only: it does not detect
semantic contamination or other benchmarks.

No Modal launcher is kept. A new run needs one that follows the Modal runtime rules in
the repository `AGENTS.md`.
