# ECB article filename recovery, 18 October 2026

The 07:58 source checkpoint contained 63 remaining HTTP 404 article candidates
outside the existing filename rules. Six bounded probes verified standalone
same-basename PDFs under the official `/pub/pdf/other/` path. Every response was
HTTP 200, had PDF magic, and contained the complete normalized inventory title in
its first two pages. The downloaded samples also passed the production validator
after the rule change. [Evidence](ecb-article-filename-recovery-20261018.json)
records candidate identities, source revision, receipts, hashes and opening text.

| Verified filename | Pages |
| --- | ---: |
| `p.11_16_mb199901en.pdf` | 6 |
| `p.29_43_mb199905en.pdf` | 15 |
| `eb201506_article01.en.pdf` | 16 |
| `art01_eb201502.en.pdf` | 13 |
| `ebart201705_01.en.pdf` | 19 |
| `ecb.ebart201803_01.en.pdf` | 30 |

The Monthly Bulletin rule adds the dotted `p.START_END_mb1999MMen.pdf` form,
restricted to 1999, matching year/month and a positive ordered page range.
The Economic Bulletin rule covers three article filename families, with an
optional `ecb.` prefix for `ebart`, restricted to 2015–2018, matching year,
issues 01–08 and positive two-digit article numbers. These are bounded proposals:
primary 404 is still required, and every inferred response must pass the existing
PDF and complete opening-title checks. There is no fuzzy title acceptance.
Whole issues, focus/box filenames, malformed ranges, queries and nested paths
remain excluded. Candidate IDs and original inventory URLs are preserved.

The new rules match **62 of the 63 saved article HTTP 404s**. This is an attempt
opportunity, not 62 verified documents or additional tokens. Sample selection
was non-random. No completed work, failure history, cooldown or campaign budget
was reset. The existing one-time opportunity for an untried alternate applies;
failed alternate URLs cannot loop indefinitely. Current workers can retain their
older image until a subsequent round.

The remaining `eb201501_focus04.en.pdf` candidate is a two-page box. Its catalog
title includes “Focus:”, while the PDF does not, so the complete-title check fails.
Its production normalized text SHA-256, however, already maps to saved box record
`9e2d5cebcc9957725020434492b9feddbcd09bf9a748387a8a36ea7f81bbcde6`
at pinned source revision `f31e2c76ba8ba339d2bd012b3164cddf71a8771a`.
That record was extracted on 17 October from the corresponding `focus/2015/pdf/`
inventory entry. This confirms existing exact text coverage; it does not justify
weakening the validator or counting the same text again. Its article candidate
remains an explicit failed metadata entry.

Validation: 314 tests passed. Regressions cover the new filename families,
malformed dates/pages, whole-issue and box exclusion, candidate nonmutation,
non-PDF bodies, mismatched or too-short titles, and a title appearing only on page
three. Ruff lint/format and whitespace checks passed. Shared collector, manual
and disabled-schedule recovery deployment receipts are retained under artifacts.
