# Preparation acceleration

The active 9.85M / 5B workflow now uses batched tokenization, atomic bulk writes
and three-source concurrent fetch/tokenization. It retains a single deterministic
SQLite writer for exact deduplication and split-conflict handling. The original
staging data identity is preserved. The prior worker stopped on an interrupted
HTTP gzip response; cached compressed downloads and four-attempt batch retries
were added before resuming its committed cursor.

Deployment code: `4d24240`.
Workflow call: `fc-01M55MH646C21S71HM79JNKNY7`.
Prior stopped call: `fc-01M55HE3MATB1X0JQPGSBZPNE9`.
App: https://modal.com/apps/radiantai/main/deployed/lm-exploration-modern-10m-5b .

Settings: 16 requested CPUs, 16 GiB RAM, eight tokenizer threads, three source
workers, atomic batches of 64 rows bounded near 4 MiB of text. Trigger-maintained
counts eliminate repeated full-database progress scans. JSON gzip shards are
cached locally and reused on retry, with the last completed shard evicted per
source. The existing source weights, parent split policy, frozen tokenizer,
capacity checks and two-pass limit remain in place. New accepted rows are written
in deterministic round-robin source batches; this can change ownership of newly
encountered cross-source duplicates relative to the older serial source order.
Existing records are preserved.

## Measured controlled CPU performance

Two Modal trials, 16 requested CPUs, two sources with 1,024 distinct generated
text records each, the actual frozen tokenizer and durable SQLite writes:

| Trial | Serial | Batched/parallel | Speedup |
|---|---:|---:|---:|
| 1 | 3.894 s | 0.839 s | 4.64x |
| 2 | 3.913 s | 0.916 s | 4.27x |

Each mode counted exactly 1,519,097 tokens per trial and produced identical
fingerprints over documents, cursors and decisions. This isolates tokenization,
compression, SQL and progress counting; it excludes network fetches and Modal
volume commits. On 256 pinned Python records and 256 pinned FineMath records, the serial path
took 1.847 seconds and the optimized path 0.517 seconds: **3.57x faster**. Both
counted 768,670 tokens and produced exactly matching fingerprints. Fetching these
samples took 70.86 seconds separately; it is not included in either CPU timing.
Benchmark app: https://modal.com/apps/radiantai/main/ap-uiCPyL9Q05vvpcctWBF4Bt .
The JSON report is the source of exact measurements.

## Live verification

The resumed workflow committed progress for Python, FineMath and peS2o together.
A post-startup processing window scanned 8,832 records in 20.10 seconds. At that
check, unique training token counts were Python 669,309,902; FineMath 8,157,350;
peS2o 40,241,044. The live status file records per-source counts and cursors so
subsequent checks can measure end-to-end throughput including volume commits.
No whole-run completion-time estimate is inferred from the short CPU benchmark.

## Verification

Tests compare serial preparation and optimized preparation on matching source
batch order, including duplicates, cross-split conflicts, multiple shards and
restart. An injected write failure rolls back both documents and cursor. An
injected download failure retries without advancing the durable cursor twice.
Migration from the existing raw database preserves data identity and token counts.
A cached-shard test checks exact resume row indices, upstream revision pinning and
old-shard eviction. These tests passed before the resumed live run.
