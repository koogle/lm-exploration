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

## Authorized overnight handoff (supersedes the manual-review pause above)

The user's final overnight instruction authorizes collection, checks, then training
without another confirmation. The continuation remains **additional 10B, 60% code**.
The CPU-only readiness app is separate from `modal_overnight.py`, whose controller
waits for preparation and schedules GPUs only after the following operational gates:

- At least 6.1B *new* eligible code targets after the combined audit, with a 2% margin
  required in the frozen selected corpus for every mixture source.
- At least eight languages have 10M targets each and at least four held-out parents;
  at least three path-task buckets have 10M each, including 50M test-code targets.
- Selected code has at most 2.4B targets per language (40% of the intended code
  budget); code loader enforces one pass across all eight-H100 segments/restores.
- Frozen tokenizer diagnostic has no UNK and aggregate sampled code fragmentation
  is at most one token per character. This screens gross mismatch, not optimality.
- Completed HumanEval/MBPP lexical checks, normalized deduplication, parent split
  exclusions, and frozen held-out coverage for every training source.
- Fresh-process two-step qualification and restore validation must match before the
  full segments proceed. Adam moments, per-rank RNG and lifetime counters are preserved.

Fresh original non-code collection runs alongside code preparation, from the prior
upstream cursors with an exclusion ledger. Remaining non-code capacity may reuse
older data, with at most two passes **in this phase**. Code never falls back to old
training documents. Code is 60%, EU replay 5%, remaining prior sources scaled to35%.
There is no silent fallback to the historical 30% code plan.

The conservative split/lexical checks and whitespace deduplication do not establish
semantic decontamination or universal family/edition reconciliation. These remain
explicit experimental limitations, rather than claims that all audits passed.
The frozen representative evaluation replaces the one-window diagnostic in this
new workflow; older plots keep their original diagnostic labels.

Controller waits are chained in bounded CPU jobs; preparation does not reserve
GPUs. Prepared data is frozen before training. GPU workers use eight H100s, 8K,
compiled microbatch 2, cropping, cached RoPE, global batch32 and prefetch4.
Periodic full optimizer checkpoints remain on the Modal run volume. Loss curves,
CSV and provenance are saved there automatically, with final matched expanded
validation and private HF publication attempts. HF currently rejects uploads due to
the owner's private storage quota; no billing upgrade or public disclosure is used.
The final model export, if upload succeeds, is named by actual lifetime tokens
(e.g. `124M - 20.879B`); older default model weights are not overwritten.

The first deployment missed an imported Modal module. The next audit encountered
intentional hash-only test tombstones. Both were corrected before any GPU launch;
stalled/failed jobs were cancelled. The active version is **readiness-20261020-v2**.
The old v1 directories are retained as failed preparation evidence.

Active overnight launch (commit `1b8dfca`):

- Run: `simple-llm-code60-unique-10b-20261020-r01`.
- Code collection: `fc-01M5B1BT8CHMRZA77RXYHPM9XR`.
- Original-source refresh: `fc-01M5B1N6PMEA0J3XQANA2436QS`.
- Durable data-gate controller: `fc-01M5B1N6WXAHC2XXN7KWQT3BES`.
- Initial existing-corpus scan: `fc-01M5B1BT1CA3J0FFT8JZ34WKFD`.

`/runs/<run>/workflow.json` records the current stage and chained call IDs.
Collection is followed by a combined audit including fresh original records;
only new records from the original fork are counted as fresh. The collector's
intermediate code-only audit is superseded/cancelled by the controller before
scheduling the combined refresh audit. No GPU allocation is needed for these steps.

If the post-filter code capacity/variety gate is short before snapshot construction,
the controller may resume the same pinned collection cursor for up to two additional
rounds (8.5B then 10.5B pre-audit targets), with a new combined audit per round.
It never fills code shortages by repeating old code or relaxing the pass limit.
Other failures, or exhausted retry bounds, remain explicit stopped states.

Final active controller: `fc-01M5B1SQR0S9GK95749CSNB0HA` (commit `c717928`).
This replaces the initial CPU waiter with the bounded capacity-recollection workflow.
The data workers were not restarted. Existing-corpus audit was observed past150,000
records; code collection was observed at7,135,227 new targets across19 languages.
Those are ingestion counts, not the post-audit 6B capacity gate.

Historical loss/optimizer curves **were published successfully as SVG** after the
binary upload hit quota: `jfrick/124M-10.879B`, revision
`8072e015339494dbb25dd17f8788488c3d230aab`. Download hashes were verified. Future
finalization publishes small SVG/provenance artifacts independently of model blobs;
full CSV/PDF/PNG is retained on Modal, and binary checkpoint quota remains a blocker.

## Storage increase and live publication

The user increased HF upload storage limits and requested continued uploads.
Binary PNG/PDF curves and full metrics CSV were uploaded and downloaded/hash-verified
at `jfrick/124M-10.879B` revision `cd890ff592a9c7e760235d560bd810210a8f512a`.
The previously missing step66,000 archive upload was resumed independently.

`modal_live_publication.py` is a separate CPU-only app. Its chained watcher publishes
every committed overnight segment checkpoint under
`continuations/<run>/checkpoints/step-NNNNNNNNN/` in the same private parent repo,
with a live `checkpoint-index.json`. Parent root weights remain available. Each
self-contained FP32 inference bundle is checked for exact weight/logit round-trip
and remote LFS hash. Full optimizer/RNG state remains on Modal. Publication does not
block or modify training; bounded retries and failures have a separate durable state
at `/runs/<run>/hf-live-publication.json`. Finalization still publishes the completed
model under a name based on actual parameter size and lifetime training tokens.
