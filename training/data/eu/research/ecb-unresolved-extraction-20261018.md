# Unresolved ECB extraction gaps, 18 October 2026

Five previously failed documents were fetched again in a bounded investigation.
All returned HTTP 200, but none is counted as a recovered corpus document.
[Evidence](ecb-unresolved-extraction-20261018.json) preserves source receipts,
raw hashes, production errors and independent-extractor probes.

Two Sintra takeaways pages (2018 and 2021) return a publication-viewer shell with
177 extracted characters: duplicated titles, without substantive body text.
There are no PDF anchors in those responses. The official viewer script provides
navigation behavior; inspection did not establish an alternative content URL.
Two convention-based `_content.en.html` probes both returned 404. These pages
remain source/retrieval gaps; no navigation or title text was promoted to a
training document.

The working papers `ecbwp204.pdf` and `ecbwp010.pdf`, and the 1997 annual report
`ar1997en.pdf`, reproduce pypdf's error for multiple embedded font streams in a
font descriptor. Independent `pdftotext` probes returned 118,111, 100,811 and
340,375 characters respectively. The first paper and annual-report openings are
readable; the `ecbwp010.pdf` opening is garbled, despite a successful exit status.
A generic fallback based only on nonempty text would therefore be unsound.
No production parser change or retry-history reset was applied. Those candidates
retain their exhausted per-item retries and unresolved status. A future recovery
needs both readable extraction evidence and a bounded, explicit retry mechanism.

These results describe five selected failures, not website exhaustion or an
exhaustive catalog investigation. Downloaded bodies stay under ignored artifacts.
