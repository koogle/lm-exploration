# Data first: no substantial training launch

The next experiment is blocked on corpus readiness. `modal_data_readiness.py`
contains CPU functions only. The earlier unlaunched cached-bootstrap training
workflow is not included. No checkpoint, optimizer, tokenizer or GPU settings change.

## Capacity requirements

| Scenario | Target tokens | Code share | Code positions | Maximum passes | Minimum unique eligible code |
|---|---:|---:|---:|---:|---:|
| Historical 20B proposal | 20B | 30% | 6B | 2 | 3B |
| Latest continuation request | 10B additional | 60% | 6B | 1 | 6B |

Collect toward 6.5B *new* code targets, then audit. This is a collection target,
not a measured usable capacity or a commitment to launch either scenario. No
silent change from the latest 60% code mix to the historical 30% proposal.

## Implemented audit

- Read-only scan of previous original/code continuation staging and private EU data.
- Collect pinned `codeparrot/github-code-clean`, including Python and other
  languages. StarCoder/The Stack sources tested inaccessible under current access;
  this workflow does not bypass their gates. The Pile is not currently ingested.
- Restrict licenses to MIT, Apache-2.0, BSD-2/3-Clause and ISC; exclude common
  generated/vendor/build paths. Preserve per-record repository/path/license/revision.
- Exclude exact prior documents and whitespace-normalized prior code clones during
  collection. Audit globally deduplicates whitespace-normalized content again.
- Canonicalize repository names across code source aliases/case. Exclude whole
  parents that cross existing split boundaries, have conflict flags, lack parent
  identity, or contain a benchmark match. Do not repair leakage by moving an old
  validation/test family into training. Preserve existing held-out assignments.
- Pin HumanEval and MBPP revisions at audit start; fingerprint every 13 consecutive
  lowercase word tokens in prompt/solution/test fields. Screen every document;
  exclude all documents from matching parents. Persist benchmark revisions/files.
- Count remaining training tokens by source and freshness; tokens from earlier runs
  are **not new**. Counts describe unique document targets before packing/truncation;
  reserve margin is required for actual sequence/batch boundaries.
- Build up to 256 held-out parents **per source/language/path-task stratum**, with
  up to two 2K windows per selected document. Persist token windows privately.
- Sample frozen-tokenizer efficiency per stratum (up to 128 documents, first 32K
  characters): tokens, UTF-8 bytes, characters, UNK counts. This is a bounded
  pre-exclusion diagnostic, not a balanced capability benchmark.

Derived audit SQLite, held-out windows and corpus text stay on private Modal volumes.
Public PRs contain code and aggregate records only.

## Remaining gates (no automatic training)

1. Completed capacity table after filtering/deduplication; verify at least 6B new
   eligible code targets for the one-pass plan, plus packing margin.
2. Review per-language/token/task balance. Path buckets are proxies for tasks, not
   evidence of instruction following, repair, executable tests, or task quality.
3. Audit cross-source family aliases for non-code sources (URL/edition/document
   variants are not universally canonicalized); perform near-duplicate checks.
4. Extend contamination checks to other chosen capability benchmarks before using
   them for claims. HumanEval/MBPP lexical screening does not establish absence of
   semantic contamination; historical exposure cannot be undone.
5. Review representative held-out coverage and tokenizer efficiency, including
   rare languages and task strata; inspect limited private samples. The existing
   tokenizer stays frozen for checkpoint compatibility unless a separate migration
   is explicitly designed and validated.
6. Integrate the frozen representative held-out windows into periodic training
   evaluation. The existing one-window diagnostic is not replaced in the trainer
   by this data-only PR, and is not an acceptable readiness gate.
7. Produce a frozen eligible-only manifest and one-pass reservation policy; wire a
   fail-closed readiness check before enabling the next training workflow.

## Operations

From `training/`, deploy `modal deploy modal_data_readiness.py`. Spawn
`simple-llm-data-readiness.audit_existing` and `collect_code` separately. The
collector schedules `audit_fresh` only after collection returns; it does not train.
Each audit requires a new output directory on retry (partial outputs are not silently
accepted as completed). Collection has a durable shard/row cursor and identity check.
Collection timeout or an exhausted corpus is a data shortage, never permission to
repeat code silently or start GPUs. Progress is under
`/data/scholarly-code/readiness-20261020-v1/`.

Launched CPU jobs:

- Existing-corpus audit: `fc-01M5B0XDDEJ64AF7N89648Y8JJ`.
- New multilingual code collection: `fc-01M5B0XDN5RP04P0H54ND57TH9`.
- [Modal deployment](https://modal.com/apps/radiantai/main/deployed/simple-llm-data-readiness).

These job IDs establish launch, not successful collection or a passed readiness gate.
