# ECB Economic Bulletin box recovery — 17 October 2026

Pinned HF state `a8601db4e291ff7e481c5dbaf2dccd8f61d4423e` has 223 box failures: 222 HTTP 404 responses and one disconnect. Filename families: mb (73), eb (103), ebbox (44), ecb.ebbox (3).

Eight bounded alternate-path hypotheses succeeded with HTTP 200 application/pdf. Each PDF contains the complete normalized inventory title in its first two pages. They span Monthly Bulletin 2014, Economic Bulletin 2015–2016, and 2017–2018 box naming conventions. No full issue was substituted for a box.

Together these eight PDFs contain 92,714 extracted characters. This is research evidence, not additional uploaded corpus or token coverage.

The shared Collector enforced robots and the configured five-second minimum source delay. Only alternate URLs were requested: original 404 outcomes come from the pinned progress state. Full URLs, retrieval times, hashes, title excerpts and HTTP metadata are in the companion JSON.

| Inventory filename | PDF pages | Characters | Title match |
| --- | ---: | ---: | --- |
| mb201404_focus03.en.pdf | 4 | 12,134 | yes |
| ebbox201802_06.en.pdf | 4 | 10,463 | yes |
| eb201505_focus04.en.pdf | 4 | 11,665 | yes |
| mb201408_focus06.en.pdf | 4 | 9,487 | yes |
| ebbox201705_03.en.pdf | 5 | 14,235 | yes |
| eb201607_box02.en.pdf | 4 | 11,088 | yes |
| ecb.ebbox201803_01.en.pdf | 5 | 11,284 | yes |
| ecb.ebbox201803_02.en.pdf | 5 | 12,358 | yes |

## Integration recommendation

Safest immediate integration is an exact eight-URL map with recorded PDF hashes. A broader bounded fallback can be proposed only for explicitly recognized historical box filenames, matching issue year to the original inventory path, retaining the original document ID, and rejecting responses unless their opening pages contain the full normalized inventory title. Do not bypass validation merely because a filename fits. The eight successes do not prove a blanket directory replacement for all 223 failures.

No adapter code, source progress, dataset state, commits or deployments were changed by this investigation.
