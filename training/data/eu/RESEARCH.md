# EU corpus research findings

Condensed findings for the ECB/EMA-focused EU corpus. Run instructions live in
[README.md](README.md). Numbers are measured snapshots with their date; refresh
private HF state (`uv run status.py both --cached-tokens`) for live totals.
Detailed per-expansion evidence is in Git history (see [Provenance](#provenance)).

## Current measured state

| Date (UTC) | Measurement |
| --- | --- |
| 2026-10-16 | First local pilot: 14 unique documents, 491,330 characters, 76,499 whitespace words from four of five investigated sources. |
| 2026-10-16 | First Modal pilot readback: 20 ECB and 139 EMA rows passed shard, text-hash, unique-ID and progress checks. |
| 2026-10-16 | Continuation baseline: 903 rows, 16,940,394 characters; later 2,866 rows / 182,334,783 characters. |
| 2026-10-17 | Four sources resumed: 7,592 stored rows across ECB, EMA, Eur-Lex and Commission press; Eur-Lex campaign finished with 2,997 documents. |
| 2026-10-18 05:22 | 18,206 ECB/EMA documents, at least 412,205,746 verified `o200k_base` tokens (ECB 10,223 docs / 172,913,672 tokens; EMA 7,983 / 239,292,074). |
| 2026-10-19 | After the HF private-storage upgrade: **38,814 saved documents and 1,001,931,446 verified tokens**. |
| 2026-10-20 | ECB reached 19,374 parsed rows after the queued 69-document batch (69 downloads, 69 extractions, 0 failures); a later read showed 40 more parsed from the 139-link backlog with 0 failures. |

Token counts come only from completed `o200k_base` audits matched to current shard
hash, path and row count; they are never estimated from characters. One billion
tokens is a reporting milestone, not a stop condition. Neither website has been
shown to be exhausted, and depleting a configured inventory is not exhaustion.

## Sources and storage

| Source | HF dataset (private) | Interface |
| --- | --- | --- |
| ECB | `jfrick/eu-ecb` | Versioned FOEDB publication database behind the ECB website; exact type IDs per collection; HTML preferred over PDF per language. |
| EMA | `jfrick/eu-ema` | Official JSON website-data inventories, partitioned by document category, with listed translations. |
| Eur-Lex | `jfrick/eu-eurlex` | CELLAR SPARQL keyset pagination, sector-3 native multilingual XHTML; bounded year inventories. |
| Commission press | `jfrick/eu-commission-press` | Press Corner search + JSON document API; only validated `htmlContent`/quotes become text; no eTranslation or language fallback. |
| Control | `jfrick/eu-corpus-control` | Goal controller claims, Modal call IDs, reservations, token caches, code backups. |

The collection goal counts ECB and EMA only ("EMR" was interpreted as EMA).
Eur-Lex and Commission press batches may finish but receive no successors.
All repositories were made public on 17 October and private again later that day
at the owner's request; mappings create private repositories.

## How documents are loaded

1. **Enumerate.** Each adapter saves an immutable inventory per collection
   (`inventories/<collection>/<hash>.json`) with coverage evidence. Inventories
   only grow; expanding a limit keeps prior candidates.
2. **Schedule.** Completed and unsupported candidates are filtered, then
   collections are interleaved by weight (ECB working papers 5, occasional papers
   and most report series 3; EMA product information 20, assessment reports 6).
3. **Download queue** (`document_queue.py`, adopted on 20 October). One paced
   downloader thread fetches at most one checkpoint batch ahead and never writes
   HF. The single writer stages gzip originals plus `download_queue` metadata in
   one commit, then parses the batch while the next downloads. Parsed Parquet,
   the exact-text dedup index and queue removal commit atomically. Restarts parse
   staged bodies before new downloads. Parse failures keep the body under the
   normal 3-attempt / 300 s × attempt cooldown; exhausted or unreadable entries
   leave the queue. Live adoption: ECB 69/69 queued documents, EMA round five
   staged and parsed its first 20 with zero failures.
4. **Measure.** Separate hourly token-audit workers count `o200k_base` per shard
   into `tokens/<source>.json` in the control repo.

Operational bounds: ECB 5 s and EMA 10 s minimum request spacing; robots checks;
the first HTTP 429 stops the source and honours Retry-After (15 min fallback);
1,000 attempts and 3,000 work seconds per worker; 3 rounds per campaign (EMA 6);
150-campaign ceiling; 96 audit pairs (owner extension on 19 October from 48).

## ECB coverage

All collections use exact FOEDB type IDs from the English database. Sample checks
fetch newest/middle/oldest documents per series before enabling it.

| Expansion (date) | Collections | Candidates | Sample outcome |
| --- | --- | --- | --- |
| Initial (10-16) | speeches, press releases, interviews, meeting accounts, decisions | 6,752 doc-language URLs | Pilot readback passed |
| Papers/bulletins (10-16) | working/occasional papers, Economic Bulletin articles and boxes | 6,032 records | Historic unsuffixed PDFs kept as `und` |
| Reports (10-17) | annual reports (36), convergence reports (20), annual-report translations | 23 non-English 2025 links | DE/FR samples > 360,000 chars each |
| Next series (10-17) | Financial Stability Review (137), Statistics Papers (5), Research Bulletin (61) | 224 records | All three sampled |
| Distinct series (10-18) | legal working papers, discussion papers, central-banking and legal conference proceedings, letters to MEPs, blog | — | Proceedings may overlap speeches |
| Report series (10-18) | Macroprudential Bulletin, financial integration, international role of the euro, TARGET annual reports | 150 records | All passed native samples |
| Other publications (10-18) | type 23 | 772 records | 3 PDFs passed |
| Surveys/T2S (10-18) | bank lending (99), SPF (158), SAFE (157), T2S annual reports (159) | 198 URLs | 11/12 passed; one bank-lending PDF HTTP 503 |
| Statistics (10-19) | 12 statistical series (weekly statements … money markets) | 2,239 URLs | 36/36 passed |
| Policy/digital euro (10-19) | 22 types incl. digital-euro documents, AMI-SeCo, quality and climate reports | 1,687 URLs | 65/66 passed; one 503 |
| Remaining types (10-20) | 28 types: EP feedback, €STR reviews/errors, consultations, legal acts, TARGET Services, oversight, IReF, FAQs, … | 139 URLs | 62/62 passed; none exactly match saved text |

URL overlap between new and previously configured types was zero in every
expansion; passage-level overlap has not been measured.

Link recovery: three convergence reports (2000, May 2006, May 2007) map to exact
replacement PDFs verified by SHA-256 after a primary 404. Historical Monthly and
Economic Bulletin boxes (1999–2018) and articles (1999, 2015–2018) may try one
guarded same-basename replacement; it must be a PDF of at most 20 pages whose
first two pages contain the full catalogue title. Filename inference alone is
never accepted.

## EMA coverage

| Collections | Candidates | Notes |
| --- | --- | --- |
| news, scientific guidelines, committee documents | 11,864 initial doc-language URLs | First run hit persistent 429s (139 ok / 361 failed of 500) before pacing fixes |
| EPAR overviews / product information / assessment reports | 50,656 / 55,214 / 1,891 PDF language URLs | 26 languages; assessments skew English |
| non-EPAR documents | 104,799 supported candidates (100,000 selected) | Weight 2 |
| EPAR variation reports / scientific discussions / discussion variations | 3,401 / 529 / 191 | Zero URL overlap with prior collections |

The `epar_documents` umbrella collection stays excluded (`exclude_collections`)
because it overlaps the per-type partitions. Recent EMA rounds hit the time limit
after about 300 attempts, so throughput is bound by the 10 s pacing, not the
attempt ceiling; campaigns therefore run six rounds.

## Negative results and known gaps

- **Deferred ECB types:** podcast type 290 (23 candidates): episode URLs redirect
  to one generic landing page with identical hashes, despite passing a
  length-only parse test. T2S harmonisation and Euro Money Market reports: two of
  three historical samples returned 404. Annual-account sections duplicate
  already collected Annual Reports.
- **Not enumerated:** FSR language-picker routes (German route served English plus
  an unavailable notice), other ECB language databases, linked attachments.
- **Retrieval failures** are kept with their HTTP outcome; no guessed replacements.
- **Change detection:** ECB inventory records lack modification timestamps, so
  content changes at an existing URL go undetected; conditional requests,
  deletions and version relationships are not implemented.
- **Eur-Lex** is bounded to selected years (5,000–10,000 items per year), not
  exhaustive law coverage. **Commission press** search totals (131,142) count
  families/fallbacks, and scans are bounded to 5,000 rows per language.
- **Extraction:** some footers and related links remain in text; no DOC/DOCX,
  OCR or near-duplicate detection.

## Incidents and fixes

| Date | Incident | Resolution |
| --- | --- | --- |
| 10-16 | EMA continued after HTTP 429 | Stop on first 429, durable source cooldown, EMA pacing raised to 5 s then 10 s |
| 10-16 | Restart could grant a fresh budget | Logical run IDs; interrupted runs reserve one uncommitted batch |
| 10-17 | Modal API upstream failure and HF 503s blocked deployment | Fresh recovery app (`EU_MODAL_APP_NAME`, schedule off) confirmed 6,889 docs at 09:37 UTC |
| 10-17 | `o200k_base` vocabulary download denied locally (proxy 403) | Token audits run on Modal only |
| 10-17 | HF 429 on token audits (~1 h Retry-After) | 2 s pacing per uncached shard, checkpoint every 50 shards, deferred resume |
| 10-18 | ECB hit HF's 10,000-files-per-directory limit in `raw/` | New blobs under `raw-v2/<prefix>/<sha256>.gz`; old files kept |
| 10-18 | Minute-level empty retry campaigns used commit quota | 10-minute backoff after a zero-attempt round with remaining items |
| 10-19 | Account-level private storage limit | Owner upgraded storage; read-back verified; no data or failure history reset |
| 10-20 | Sibling campaigns kept a finished source unsettled | Source inspection considers only the requested Modal call subtree |

## Provenance

The dated research notes, per-expansion sample/inventory JSONs, live evidence and
agent handoff reports were removed from the working tree after this summary. They
remain in Git history at commit
[`3d25fa6`](https://github.com/koogle/lm-exploration/tree/3d25fa6905fff7521106809565e3df8fef090b52/training/data/eu)
under `research/` and `reports/`. The 16 files PR #22 added under `research/` and
its version of `reports/CONTINUATION_20261018.md` are also archived, byte-verified,
in the private dataset
[`jfrick/lm-exploration-results`, folder `pr-22/training/data/eu/`](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/pr-22/training/data/eu)
(HF commit `4740d892b85a33ba6cdebda5ab1113c9eb065ba2`). HF checkpoint state (`state/progress.json`,
`metrics_history`, campaign records) and the control repository hold the durable
operational record.
