# Next ECB expansion — 17 October 2026

The best immediately verified addition is **Financial Stability Review**, FOEDB
type **137**: the current official inventory reports **44 publications**. The
latest native HTML report is accessible again after the earlier 503 failures.
Two complementary, distinct series also passed full-text probes: **Statistics
Paper Series** (type **5**, 54 publications) and **Research Bulletin** (type **61**,
126 publications). Together these are 224 metadata records before representation
filtering, failures and text deduplication, not 224 newly collected documents.

| Verified document | Representation | Extracted characters |
| --- | --- | ---: |
| Financial Stability Review, May 2026 | English HTML | 201,824 |
| Statistics Paper 54, Domestic and cyclical inflation in the euro area | Text-bearing PDF | 94,186 |
| Research Bulletin, Private equity beyond the target: how buyouts reshape supply chains | English HTML | 10,930 |

Exact URLs, UTC retrieval provenance, HTTP outcomes, raw hashes, extracted text
prefixes and enumeration requests are retained in the companion JSON. These are
actual document bodies, not inventory abstracts. No full historical backfill was
performed in this investigation and these samples are not a token-volume forecast.

## Enumeration and integration

All three use the existing official English FOEDB publication database and its
shared adapter. The live snapshot was `1792213219/2sP68GoB`, with 20,162 publication
records overall. Load `versions.json`, its version/hash `metadata.json`, and
`indexes/type/index.json`; select the named type, read its index chunks, then
load referenced publication data chunks and explicit `documentTypes` URLs.
The current probe selected only the newest document in each series (`limit=1`),
with incomplete output explicitly recorded. No guessed document paths were used.

The official type names were independently established in the retained
`publications_types` database (see `ecb-expansion.md`); this investigation also
re-fetched its version manifest successfully. A parent integration can add these
three named type mappings and use the same bounded persistent worker without a
new crawling architecture. Prefer HTML over same-language PDF as already done.

FSR whole reports overlap their own article (type 138) and box (type 217)
subcollections; do not add both representations and count their text as novel.
Those FSR subcollections are not currently configured. Full Economic Bulletin
issues would substantially overlap the already configured Economic Bulletin
articles/boxes, so they are a lower priority. Research Bulletin is a separate
short-form publication series; thematic overlap with research papers remains
possible even though the documents differ.

## Critical multilingual negative finding

The FSR page explicitly links the same report under 24 language suffixes. The
German route returns HTTP 200 and `<html lang="de">`, but its actual report is
English preceded by **“Nicht auf Deutsch verfügbar.”** English extraction is
201,824 characters; the German-route extraction is 201,853. This is a translated
UI around the same English document, not a German translation. Generic language
anchor crawling would create near-duplicates that exact text hashing alone would
not remove. Do not extend the annual-report translation strategy blindly to FSR.

## Existing failures and remaining limits

A fresh `jfrick/eu-ecb` progress snapshot showed 222 Economic Bulletin box HTTP
404s, 63 article HTTP 404s, and 10 article fallback title mismatches. Sample URLs
are preserved in the JSON. The dominant gap is broken historical individual
bulletin paths, not current FOEDB availability. Preserve title validation rather
than accepting unrelated replacement PDFs. Five working papers exceed the current
body-size bound; two more have PDF font-descriptor extraction errors. Those small
residuals are separate from adding new publication series.

Used the shared Collector with five-second per-origin spacing, robots checks,
normal proxy/TLS settings, and the existing 20 MB body limit. Raw bodies and probe
scripts remain ignored under `artifacts/`. No adapters, mappings, schedules or HF
corpus data were changed by this research task.
