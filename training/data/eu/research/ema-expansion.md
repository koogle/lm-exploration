# EMA expansion: targeted EPAR collections

This investigation uses the complete cached EPAR inventory captured during the
initial investigation (source timestamp `2026-10-16T05:49:58Z`). It makes **zero
new EMA network requests**, preserving the source cooldown after rate limiting.
The inventory has 20,266 source records and is 29,222,731 bytes. This is a cached
coverage measurement, not a claim that the inventory is current today.

Three exact inventory `type` filters now expose coherent series:

| Adapter collection | Official type | Source records | Accepted language URLs | Languages |
| --- | --- | ---: | ---: | ---: |
| `epar_product_information` | `product-information` | 2,219 | 55,214 | 26 |
| `epar_overviews` | `overview` | 2,224 | 50,656 | 26 |
| `epar_assessment_reports` | `assessment-report` | 1,796 | 1,891 | 26 |

Product information contains 72 additional `/media/` landing links unsupported by
the direct-document adapter; this is recorded as skipped coverage and `complete`
remains false. Overviews and assessment reports completely enumerate their cached
inventory subsets at limit 100,000. All accepted links in these three subsets are
PDF URLs. None of these new bodies was downloaded or extraction-validated here.
Assessment reports are overwhelmingly English (1,796 English, 95 translations),
while product information and overviews have broad multilingual coverage.

Prioritize overviews for relatively concise multilingual prose, product
information for detailed regulatory/clinical text, and assessment reports for
longer technical assessments. These are document-type interpretations, not
measured full-text lengths or quality ratings.

## Shared adapter behavior

- Use the same EPAR inventory fetch and cache it across collections.
- Preserve source document IDs, original-language links, translations and dates.
- Retain `medicine_name` and `ema_product_number` from observed inventory fields
  to support grouping related documents by medicine.
- The new collections are mutually exclusive by exact official type. They are
  **subsets of existing `epar_documents`**, whose behavior remains compatible.
  Do not schedule the umbrella alongside these without document-ID deduplication.
  Collection metadata and enumeration output declare this overlap.
- All limits now expose `selection_limit`, `unselected_count` and
  `selection_truncated`, in addition to the existing completeness flag. Thus an
  umbrella collection exceeding 100,000 is explicitly incomplete in machine-readable
  output, rather than relying on a human reading its limitation prose.
- Stable document IDs do not change when moving from umbrella to targeted series.

Recommend mapping these three collections to type-specific partitions inside the
existing EMA organization dataset. Enumerate with limit 100,000 to retain all
accepted candidates, then let the durable parent harness select pending documents
within its authorized run budget. Resume after durable source cooldown; enumeration
is not permission to restart requests early. No source mapping, cloud launch or
publication change was made in this task.

Eight adapter tests pass, including exact series filters, stable umbrella/subset
IDs, multilingual expansion, preserved medicine metadata and explicit truncation.
Machine-readable evidence is in [ema-expansion.json](ema-expansion.json).
