# ECB annual report translation enumeration — 2026-10-17

Implemented `annual_report_translations` in the existing ECB adapter. It starts
from the bounded English FOEDB annual report inventory (at most 1,000 URLs), then
inspects at most 100 English native HTML report pages. It accepts only explicit
anchor links with exactly the same canonical path and report identifier/hash,
changing only the language suffix. It does not synthesize links or recurse.
English and URLs already present in the base inventory are excluded. Records
retain the FOEDB publication identity, distinct URL-derived document identity,
URL language label, and seed URL as discovery provenance.

The cached Annual Report 2025 HTML produces **23 non-English candidate URLs**.
Two were fetched using the shared Collector with robots checks and five-second
host spacing on 17 October 2026:

| Version | HTTP | Extracted characters | HTML language | Title |
| --- | --- | ---: | --- | --- |
| German | 200 | 368,093 | de | Jahresbericht 2025 |
| French | 200 | 383,993 | fr | Rapport annuel 2025 de la BCE |

These are actual full report HTML samples, not metadata or inferred translated
text. Raw responses remain under ignored `artifacts/ecb-extra/translations-*`.
The companion JSON retains exact URLs, timestamps, hashes and retrieval results.
No corpus upload, mapping edit or deployment was performed by this investigation.

Tests cover foreign hosts, navigation, other reports and different hashes,
duplicate links, already inventoried URLs, English exclusion, stable publication
identity, output bounds, seed bounds and seed retrieval failure propagation.
All 28 ECB adapter tests pass; targeted Ruff checks pass.

Completeness describes only the inspected explicit-link inventory. Historical
PDF-only reports and translations with different document identifiers/paths are
excluded. The two samples establish that these particular versions contain
native translated text; other candidate languages still require body retrieval.
The entire new collection has not been enumerated live, and neither institutional
coverage nor website exhaustion is claimed. URL suffix language remains publisher
metadata; this enumeration does not perform statistical body language detection.
