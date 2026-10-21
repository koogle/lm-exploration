# Commission Press Corner — bounded live research

Observed 2026-10-16 UTC; eight GET requests. Machine-readable discovery works, but native full-document extraction needs a source adapter.

- `https://ec.europa.eu/commission/presscorner/api/rss?language=en` returned HTTP 200, `application/rss+xml`, with ten current item links, publication dates, categories, and short summaries. These summaries are **not full release text**.
- Home and a release detail URL returned HTTP 200 HTML. The sampled detail body contained an Angular application shell and zero paragraph elements. A generic text extractor must not accept the shell as the release.
- `https://ec.europa.eu/commission/presscorner/api/files/document/print/en/ip_26_2175/IP_26_2175_EN.pdf` returned HTTP 200 `application/pdf`, 82,742 bytes. This establishes a downloadable fallback; PDF text extraction was not tested in this probe.
- Guessed list and item API endpoints returned 400 and 404 respectively. The observed frontend JS was fetched but did not resolve the detail endpoint within this bounded investigation. Browser network inspection is the next discovery task.
- Robots returned HTTP 200. No Press Corner-specific restriction was found; `/archives/` is disallowed. Recheck current rules during collection.

Use RSS for incremental discovery. Resolve full text via the frontend's actual API or download the print PDF, preserving original release ID and language. Do not infer historical completeness from the ten-item RSS window. The first sampled RSS item was a statement, with speeches and press releases also present.

Exact observations and crawler-ready seed/pattern recommendations are in [commission-press.json](commission-press.json). Raw probe bodies remain in ignored artifacts. No token extrapolation is supported by this sample; eligibility is unresolved and this source is not registered for training.
