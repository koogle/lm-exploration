# ECB Monthly Bulletin box recovery — 18 October 2026

The pinned ECB checkpoint `24db2891873872b549ce5bbab945bf8d65a7562b`
contains primary HTTP 404 failures for Monthly Bulletin boxes before the fallback's
2014 lower bound. Three bounded same-basename `/pub/pdf/other/` probes used the
shared Collector with robots checks and five-second source pacing.

| Sample | HTTP | Pages | Complete opening-title match |
| --- | --- | --- | --- |
| `mb201312_focus07.en.pdf` | 200 | 6 | Yes |
| `mb201212_focus08.en.pdf` | 200 | 2 | Yes |
| `mb201112_focus13.en.pdf` | 200 | 2 | No |

The two accepted samples contain 19,774 extracted characters. These are probe
results, not additional persisted corpus documents or verified tokenizer counts.
The 2011 response is retained as negative evidence and is outside the expanded
rule. Full retrieval provenance and hashes are in the companion JSON.

The adapter now proposes one alternate for Monthly Bulletin boxes from 2012–2013,
in addition to its existing 2014–2018 scope. It still requires matching issue/year,
a recognized basename, PDF content, at most 20 pages, and the full normalized
catalog title in the opening two pages. Earlier Economic Bulletin filename families
remain excluded. No complete issue can substitute for a box.

The pinned inventory has 175 qualifying 2012–2013 candidates; all are eligible
under existing completion/retry rules at this snapshot. This is an opportunity
count, not a predicted yield. Previously exhausted primary 404s get one attempt
at the newly available alternate through existing `pending()` logic; completed
candidates, retry timestamps and attempted-alternate history remain intact.

Remote workers keep their current images until subsequent rounds/campaigns use
the deployment. No source checkpoint is edited and no replacement writer is
launched for this repair. Website exhaustion remains unproven.
