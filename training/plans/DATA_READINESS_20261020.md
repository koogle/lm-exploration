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

Storage retry completed: all **70/70** saved checkpoints from the completed 5B
code-heavy continuation are present in the private HF archive, with remote hashes
verified. The overnight live publication watcher was launched as
`fc-01M5B66NVTQ78BPV4SFQ78RVA2`. It includes the new phase seed once the first
segment's export source exists, then every saved segment checkpoint. Full overnight
PNG/PDF/SVG/CSV publication is enabled again with the increased storage allowance.

## Public latest completed inference release

The user explicitly authorized making the latest model public. Published a separate
allowlisted inference-only release at
[Simple LLM124M -10.879B](https://huggingface.co/jfrick/simple-llm-124M-10.879B),
revision `62b9c8735949798246efd5efbde27c156ae49c43`. It contains the completed root
weights, tokenizer, generation/model code, sanitized release metadata and aggregate
loss/optimizer plots. No raw training documents, private data manifest, historical
checkpoint archive, optimizer/RNG state or raw generated evaluation text was copied.

The weights match SHA256
`f50314c30b42b2c871cdec6aab4796109ca5ac11e1f2e055e795ce97e1e469cc`.
Standalone exported inference and anonymous public API access were verified.
`jfrick/124M-10.879B` remains private, so continuous overnight checkpoint uploads
and its archive-privacy checks continue to work. The public release represents
10,878,562,619 lifetime targets, not the still-preparing multilingual experiment.
The explicit publication script is `publication/publish_latest_public.py`; it stages
privately, verifies weights, then changes visibility only for the inference bundle.

## Ensuring the run starts: remove serial CPU preparation bottlenecks

The GPU handoff was still waiting at808M fresh code targets after several hours;
the non-code refresh completed with1.103B new targets and allows cached non-code
supplementation as already planned. No GPU work had started. The code collector
was progressing, but serial downloads/tokenization and Python lexical scanning
would postpone training unnecessarily.

`modal_parallel_code.py` prepares disjoint pinned shard ranges on eight8-CPU workers.
It preserves all committed serial documents/cursors and snapshots an immutable
exact/whitespace exclusion ledger before starting those workers. Each worker caches
the hash ledger on local SSD, persists its own cursor, and batches statistics updates.
A single merger deduplicates across workers, excludes cross-split duplicate conflicts,
and records idempotent merge identities. Initial workers target1.1B each, providing
extra pre-merge capacity so global deduplication has room to remove clones. The same
6.1B post-audit minimum, code-variety checks and one-pass training cap remain required.
If a later audit needs more code, the same workers resume their own cursors.

Benchmark screening now optionally uses a C Aho–Corasick automaton over exactly the
same lowercase13-word fingerprints. Word-boundary/reference-equivalence tests pass;
this is a speed change, not a relaxed contamination rule. The final combined audit
still runs before snapshot selection or GPU allocation.

Tests now exercise the actual controller locally with fake calls: incomplete data
chains CPU waiters without scheduling GPUs, completed audits schedule freezing,
and insufficient capacity requests more code. Additional fixture tests cover disjoint
shard coverage, initial partial-row resume, tokenizer/cursor restart, held-out
repository exclusion, global merge deduplication/conflicts, and merge replay.

The serial collector and its waiting CPU controller were stopped after preserving
committed data. Replacement parallel coordinator:
`fc-01M5BBTD60E5XHKDV8SGXWVV0X`; replacement training/data-gate controller:
`fc-01M5BBTDF5B7XDT85QAX44088H`. Source commit: `44e033d`.
The prepared corpus path and training run ID are unchanged.

A further handoff check found a reserve mismatch: a6.1B preflight could pass even
though snapshot construction needs6.12B plus8,192 targets. The early guard now
uses the exact same reserve, and snapshot construction stages `raw.building.sqlite`
before atomically publishing `raw.sqlite`. A failed code-capacity/variety selection
can therefore resume collection instead of being blocked by a partial database.
29 relevant tests passed, including real frozen-tokenizer partition resume and the
controller handoff fixtures. No GPU allocation is claimed before the data gate.

Remote startup validation caught a real coordinator failure after eight partition
workers had launched: the parent container had not reloaded the volume directory
created by its exclusion-snapshot child. The eight workers continued preparing data;
no worker data was lost. The coordinator now explicitly reloads that volume and
reconnects recorded worker call IDs instead of launching duplicate writers. Worker,
merge, audit and freeze entrypoints also reload before reading another job's commits.
The already-running eight calls were recovered into `calls-r0.json` and are reused.
Two new regression tests cover the missing parent directory and recovery without
spawning duplicate workers. The redundant slow interim audit was stopped to free
CPU/I/O; the mandatory faster combined audit still gates training.

Recovered live coordinator: `fc-01M5BCPFSA17CH19W8XKK5PMMR`; live data/training
controller: `fc-01M5BCPFZK43HJHJNT2QCH5S0Q`. The original eight workers were
reconnected, not duplicated. All eight were observed producing token/cursor progress;
first measured aggregate was817M partition targets plus839M serial targets, before
global deduplication/audit. Publication watcher was restarted after the temporary
controller failure as `fc-01M5BCQ76HG92RBW2MZ78J7NY5`.

The global merger now uses a separate implementation with RAM-resident hash indexes
and4GiB SQLite page caches (within its32GiB allocation), avoiding millions of random
remote-volume lookups. The running partition implementation stays byte-for-byte
unchanged, so saved ingestion identities and retry cursors remain valid. Both merge
engines pass the same independent expected-count, cross-split conflict and replay
fixtures. This change does not relax data eligibility or the fresh-code gate.

Audit metadata uses a4GiB SQLite cache plus512MiB/source; snapshot assembly uses
1GiB for its output and1GiB/source. These are lazy page-cache upper bounds within
32GiB CPU containers, intended to avoid random-volume I/O becoming the next serial
bottleneck. They do not change filtering, counts or split assignments.

Final active binding after deployment checks: collector/coordinator
`fc-01M5BCZKTQAHWKNKYRNXEMG5MD`; training handoff controller
`fc-01M5BDG48HZF39BX0MWN47NFHY` (source `43927af`). The eight original partition
calls remain active and retain their ingestion implementation/cursors. All eight
were observed advancing; raw collection passed3.2B targets (serial+partitions,
before cross-worker deduplication/audit).31 relevant tests pass. No GPU work is
claimed until the unchanged readiness gate succeeds.

Capacity-only recollection can now continue for up to eight rounds if filtering
removes more code than the initial reserve allows, while retaining the exact gate.
An exhausted shard partition returns its valid staged capacity; it does not discard
that data or fail a globally sufficient pool. Worker implementation identities and
pass limits are unchanged. All source/quality failures outside these capacity cases
remain explicit stops instead of silently training on an unsuitable corpus.

After the final deployment, collector, handoff controller, and HF watcher
`fc-01M5BCQ76HG92RBW2MZ78J7NY5` were each confirmed running. The workflow remains
`collecting_fresh_code`; the repaired controller will audit, freeze, qualify restore
on eight H100s, and continue the 10B phase after the gates pass. No further user
approval is required. The final focused handoff/collector/gate suite passed 13 tests.

## Patch supplement authorized by the user

The user explicitly permits overlapping versions in their patch datasets. Include
Git diff text from pinned `changes/**/*.jsonl.gz` in the two private collections
`jfrick/mit-claude-codex-code-20261019` at
`b27f9cef49fa44643247d8a97173201fd47896c8` and
`jfrick/curated-oss-code-20261019` at
`1bda7edae2fac3b6d646422377817ee53893bc3c`.
Use the patch itself, not concatenated before/after buffers or author metadata.
Related changes may share context and code. Exact/whitespace-equivalent whole
records still deduplicate; repository splits and benchmark checks still apply.
Keep existing permissive-license eligibility and retain root license evidence.
These are eligible patch-document tokens, not unique underlying code spans.

A resumable CPU importer runs separately from the eight unchanged source-code
workers. The combined audit waits for patch import completion and includes that
SQLite database; snapshot selection includes eligible patches within the same
60% code budget. `code/diff/patches` records are reported separately and never
count as an additional programming language for the eight-language gate. The
snapshot records selected patch targets and the overlap policy. The 6.12B code
capacity reserve, code pass cap, noncode weights, and GPU settings are unchanged.
The focused importer/collector/audit/gate/handoff suite passes 16 tests.
