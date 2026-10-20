# ECB technical, consultation and reference expansion

On 20 October ECB completed the preceding 69-document expansion through the
durable download queue: all 69 were downloaded and extracted, with zero failures,
bringing the source to **19,374 parsed rows**. The configured inventory was then
depleted. Neither that depletion nor the catalogue inventory is website exhaustion.

This expansion selects 28 additional exact types from all 81 cached official
FOEDB English data chunks in snapshot `1792213219/2sP68GoB`, with **139 distinct
candidate URLs**. It includes EP annual-report feedback, €STR methodology reviews
and error reports, strategy review documents, consultations and responses,
EBA/ECB reports, BPM6 and TARGET guides, SPACE and company cash studies, legal
acts, Governing Council and environmental statements, TARGET Services annual
reports, letters, digital-euro FAQs/governance, the bank treasurer survey,
Eurosystem oversight, macroprudential focus articles, IReF documentation, climate
indicator releases, and bank-lending questionnaires/glossaries. The adjacent JSON
records the exact type IDs, counts and native formats for every selected series.

All **62 selected newest/middle/oldest samples** from enabled series fetched and
parsed native full text (869–285,380 normalized characters). None exactly matches
the saved-text index at source revision
`75b7b2c7f2b18b2ba80ac38e31bff2f29ce6903c`. Candidate URLs have no exact
overlap with previously configured types or one another. Passage overlap has not
been measured. Explicit language variants are retained. Type 168 is a complete
conference programme, not proceedings, and is named accordingly; type 281 is a
complete meeting agenda. Short original tables and agendas use the unchanged
200-character extraction threshold.

Three additional podcast samples mechanically parsed 2,898 characters, but all
three episode URLs redirected to the same generic Euro Matters landing page,
with identical raw and text hashes and no episode transcript. **Podcast type 290
and its 23 candidates remain deferred**, despite passing a length-only test. All
65 sample outcomes and the actual redirect targets are retained. The earlier
deferred/broken types and exclusions for whole-report sections remain unchanged.

Discovery and collection were separated operationally: after confirming the
preceding ECB supervisor and descendants terminal, a single CAS commit saved the
28 reconstructed inventories in the private source repository. Every inventory
was read back, while rows, metrics and immutable shards remained unchanged.
Source write/readback revision:
`5fb9a1f94106ac8fb218a3fd8409b6e1a60011f4`. No website request or source-worker
dispatch occurred during that inventory write. Metadata states that cached native
rows were reconstructed; synthetic type-index URLs are not claimed as fresh
retrievals. See `ecb-inventory-queue-20261020.json` for paths and terminal proof.

Both mappings add the series at the existing default weight of one. Existing
weights, five-second pacing, private storage, single-writer ownership, retry
budgets/cooldowns, raw queue protocol, audit limits and collection/time budgets
are preserved. The existing controller owns subsequent worker dispatch. Only
successful parsed documents contribute to row/token totals; 139 links alone are
not 139 saved documents.

Validation: **435 tests passed**, plus Ruff lint/format and whitespace checks.
The new cases exercise each exact type through enumeration, native-format
preference and distinct language variants. Whole-website exhaustion, new-date
refresh, unindexed attachments, other language catalogues and unresolved source
errors remain coverage gaps.

The goal and manual apps were deployed. The existing controller launched
`goal-ecb-1792481512-b29f302ce6`, supervisor
`fc-01M5C3JMZ0K8RAM2GBV0J974XG`, with three rounds of at most 82 attempts
and the unchanged 3,000-second cooperative limit. A fresh worker adopted adapter
digest `42dd7f63b4d28313117839885a0e593dd5c85d7041e6353c2172c5919f61e115`.
At the 07:35 UTC pinned observation its first **20 originals were durably staged**
and parsing was underway; this is not a claim that all 139 were already parsed.

EMA independently adopted `durable_download_queue_v1` in round five of its
existing campaign. Its first **20 originals were staged and then all 20 parsed**
with zero failures: the raw-byte counter rose at 07:32:02 UTC, followed by parsed
rows rising from 23,961 to 23,981 at 07:32:59 UTC with raw bytes unchanged.
Both supervisors had live descendant trees and matching queue-module fingerprints.
See `ecb-expansion-live-20261020.json` for source/control revisions and counts.
