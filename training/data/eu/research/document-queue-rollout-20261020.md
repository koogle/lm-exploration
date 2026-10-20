# Durable download queue and separate parsing

The owner requested that links/documents be queued in their own thread and parsed
separately. Previously each source worker fetched, parsed and checkpointed serially.

Both ECB/EMA mappings now enable `download_queue`. Existing immutable inventories
remain the durable link backlog. One network thread uses the original Collector,
including its robots checks, redirects, host pacing and size limits. It spools no
more than one checkpoint batch ahead (20 documents in production). The calling
thread alone writes HF state, stages compressed originals plus provenance in
`download_queue`, and parses the preceding batch while the next batch downloads.
Parsed Parquet, exact deduplication state and queue removal commit atomically.
Previously staged bodies replay from a pinned HF revision after interruption,
without another website request. Parse failures retain the raw under existing
attempt limits and cooldowns. Only parsed records count toward corpus/token totals;
status reports raw queue depth separately.

There is one speculative batch at most, covered by the existing interrupted-run
batch reservation. Attempt allocation never exceeds the logical run authorization.
The producer stops immediately on a 429 and cannot mutate HF progress. On writer
failure the producer stops and joins before the HTTP client closes. No additional
source writer, independent parser service, new schedule, token-stop condition or
budget increase was introduced. Existing workers retain their original image until
their next round; deployment alone is not proof of adoption. Larger independent
download/parse services remain future work.

Validation: **407 tests passed**, plus Ruff lint/format and whitespace checks.
Tests demonstrate concurrent downloading while parsing is blocked, that durable
raw staging precedes parsing, that every commit stays on the calling thread,
restart after failed Parquet commit without another fetch, stage-upload failure,
429 stops, parse cooldowns and exhausted retry budgets, authorized attempt bounds,
alternate validation on the parser thread, exact deduplication and deferred parsing
recovery. A storage regression checks that staged raw is later linked to its
Parquet shard without reupload or double-counted bytes. Existing serial workers
remain covered by the preceding tests.

The goal app and manual collector app were deployed; the recovery schedule was not
enabled. Existing tick `fc-01M5C1SPK1C7ZKQHBEC0DT0H8T` launched ECB campaign
`goal-ecb-1792479654-943c165083`, supervisor `fc-01M5C1T0188WRE15N2AQB45HEG`.
The new worker started at 07:01:20 UTC with pipeline digest
`6e0c634991a964b67794f1a4fdef7fe1582deca2ea3a8679ccb25efd7f5d9398`.
EMA's existing campaign was left running. New ECB work comes from the three
verified series in `ecb-queue-expansion-20261020.{md,json}`: 69 candidates, nine
selected successful samples, with document counts measured separately.

The first new ECB worker failed before reaching the document queue: the latest
official catalogue snapshot `1792478909/i3v6cwlB` returned HTTP 404 for
`data/0/chunk_50.json`. Its own supervisor/descendant tree was confirmed terminal
by the controller. The failure remains in history. The existing controller then
launched `goal-ecb-1792479829-d2116daa48`, supervisor
`fc-01M5C1Z9NK01WX7YYXD8QKPA33`; that worker started at 07:04:05 UTC and
successfully persisted the first two new inventories (24 and 21 candidates) by
07:08 UTC. A cached-inventory fallback was prepared locally but not applied.
The live retry was allowed to continue; no second source writer was introduced.
