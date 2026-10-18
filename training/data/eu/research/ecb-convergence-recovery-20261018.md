# Remaining ECB convergence report recovery — 18 October 2026

Ten primary 404 failures in pinned ECB state were investigated through bounded
same-basename `/pub/pdf/conrep/` hypotheses. All ten returned HTTP 200 PDFs.
Opening pages were reviewed individually for title, issuer and report year,
including the 1996 and 1998 European Monetary Institute reports. Months agree
where printed. The 2002 report identifies its year and statistics cutoffs; the
catalog publication date is not used as an exact content date.

| Original basename | Pages | Extracted characters |
| --- | ---: | ---: |
| `cr201406en.pdf` | 274 | 860,220 |
| `cr201306en.pdf` | 88 | 275,220 |
| `cr201205en.pdf` | 272 | 848,390 |
| `cr201005en.pdf` | 273 | 840,915 |
| `cr200805en.pdf` | 265 | 808,804 |
| `cr200612en.pdf` | 244 | 693,426 |
| `cr2004en.pdf` | 248 | 636,861 |
| `cr2002en.pdf` | 66 | 145,461 |
| `cr1998en.pdf` | 368 | 789,864 |
| `cr1996en.pdf` | 162 | 393,614 |

The probes yielded 6,292,775 extracted characters, not newly stored corpus tokens.
Exact URLs, raw SHA-256 values, publisher provenance, source revision and opening
identity excerpts are in the companion JSON. The current index response is
also retained; replacements were hypotheses, not inferred links from that page.

Integration extends the existing exact map to thirteen verified assets. Every
runtime replacement requires its recorded SHA-256; a changed file fails closed.
No broad prefix rewrite, completion reset, retry-budget reset or duplicate writer
is introduced. Existing `pending()` logic permits the previously untried alternate
after a primary 404 and preserves old attempts and cooldowns.

Shared Collector robots checks, five-second pacing, byte limits and TLS trust
were retained. These samples do not establish website exhaustion or verify all
other historical formats. Subsequent collection must establish persisted yield.

Validation: the full 269-test suite, Ruff lint/format and whitespace checks passed.
The collection, manual and recovery apps were deployed successfully; the recovery
schedule remained disabled. Existing active workers were preserved. New mappings
will be used by subsequent worker images; saved yield is still to be verified.
