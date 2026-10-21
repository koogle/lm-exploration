# EMA additional document collections — 2026-10-17

The existing `non_epar_documents` adapter can add **104,799 supported document-language URL candidates** outside the five currently selected document collections (news is separate). This is a count from the official cached inventory timestamped 2026-10-16T05:53:32Z, not an estimate of unique extracted texts. Exact URL overlap with the three selected EPAR series, guidelines and committee collections was **zero**. Normalized-text deduplication remains necessary.

The inventory contains 42,854 matching base records. Expanding explicit translation URLs produces 104,799 supported candidates across 26 language codes: 41,928 English and 62,871 other-language candidates. These are overwhelmingly PDFs; the JSON inventory is metadata, not native document full text. No new pagination or guessed endpoints are needed. The existing collection filter excludes scientific guidelines, agendas, minutes and committee reports, so adding the collection does not intentionally repeat those partitions.

Large categories include referrals (37,559), periodic safety update assessments / `psusa` (17,885), medicine Q&A (6,710), paediatric investigation plan decisions (6,627), presentations (6,333), reports (1,842), regulatory/procedural guidelines (780), herbal assessment reports (533), and annual reports (294). Category counts describe language-level candidates. Some are short administrative records; sheer document count is not proportional to token yield.

## Verified full-text samples

Three document GETs and the shared Collector's robots check were made at 09:00 UTC, respecting its ten-second source request interval, preserved proxy/TLS settings, and 20 MiB response cap. All three returned HTTP 200 PDF and passed the same `pipeline.extract_record` used in production:

| Sample | Characters | Whitespace words |
| --- | ---: | ---: |
| Rifadin Article-31 referral timetable | 1,630 | 228 |
| HMA–EMA AI industry stakeholder meeting notes, October 2026 | 7,953 | 1,068 |
| Fragaria herbal assessment report, revision 1 | 60,351 | 8,434 |

The report and herbal assessment contain substantive original prose. The referral is a brief procedural timetable; its relatively large letterhead share illustrates why this three-item convenience sample must not be used to forecast the whole collection. PDF text retains agency notices, page labels, references and table-reading-order limitations; no LLM rewriting, OCR or boilerplate stripping was applied. Full bodies and extracted texts are under ignored `artifacts/source-probes/ema-extra-20261017/`; compact URLs, response headers, hashes and extraction counts are in the sibling JSON report. These probes were not uploaded or counted as corpus growth.

## Integration recommendation and gaps

Add `non_epar_documents` to future EMA goal-worker mappings with a modest scheduling weight (suggested 2), retaining weight 6 for assessment and product-information series. Existing adapter support and organization-wide deduplication suffice; no adapter change is required. The current 100,000-per-collection ceiling omits the oldest 4,799 supported candidates, and another 1,350 unsupported URL variants are explicitly excluded (mainly DOC/DOCX, spreadsheets, ZIP and presentations). These are persistent coverage gaps, not exhausted documents. The recorded full census used a larger offline enumeration limit only; it did not change worker limits.

The complete cached EPAR inventory also contains **72,232 supported candidates** beyond the selected three EPAR types. Of these, 54,652 are authorised-presentation/package listings, so adding the whole overlapping `epar_documents` collection would waste candidate scheduling and favor low-value package tables. More promising later targeted types are `variation-report` (3,401), `scientific-discussion` (529), and `scientific-discussion-variation` (191). Those types are inventory-verified here but their full-text extraction was not newly sampled, and they need explicit adapter partitions before integration.

These inventories do not prove coverage of all EMA websites, removed history, unsupported document formats, clinical data systems, event media or updated documents since the cached snapshot. Resume collection from durable HF checkpoints, preserving original URLs, inventory IDs, languages and dates.
