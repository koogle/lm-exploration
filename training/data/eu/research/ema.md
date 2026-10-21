# European Medicines Agency — bounded live research

Observed 2026-10-16 UTC; eight GET requests. EMA is a strong first collector: news has server-rendered article text, and documented JSON inventories expose a much larger document collection.

- `https://www.ema.europa.eu/en/news` returned HTTP 200 HTML with ten article links, including committee meeting highlights, regulatory announcements, and an executive-director speech.
- The sampled article, `https://www.ema.europa.eu/en/news/clinical-trial-applications-advice-pilot-becomes-permanent`, returned HTTP 200 HTML containing complete prose paragraphs in the response, without browser execution. Extract the article region rather than global navigation/language controls.
- `https://www.ema.europa.eu/en/documents/report/news-json-report_en.json` returned HTTP 200 `application/json`, 2,238,532 bytes. Its complete JSON reports **3,894 records**, timestamp `2026-10-16T06:02:24Z`. Each item exposes `news_url`, title, summary, categories, topics, and publication/update dates. It is an inventory, not full article text.
- The documentation page links a general document inventory, EPAR and non-EPAR inventories, medicine datasets, scientific guidelines, events, and other collections. `documents-output-non_epar_documents_json-report_en.json` returned HTTP 200 JSON whose header reports **50,525 records**, timestamp `2026-10-16T05:53:32Z`. The body was intentionally truncated at 3 MB; that reported total was not independently enumerated. Observed entries include agendas with `document_url` PDF links, stable IDs, and update dates.
- Robots explicitly allows `/*/documents/report/*.json$` and disallows search/admin/user paths. Chosen news and inventory paths showed no obstacle. The guessed old RSS documentation path returned 404; the actual `/en/news-events/rss-feeds` page returned 200.

Start with the HTML news listing for the generic collector. Add a JSON inventory adapter to enumerate `data[].news_url` and `data[].document_url`; choose HTML/native text where available and PDF extraction where necessary. Keep inventories separate from extracted corpus documents. Counts above are inventory records, not extracted full documents or token counts.

Exact observations and crawler-ready configuration are in [ema.json](ema.json). Raw responses remain in ignored artifacts. This is research, not a training eligibility or publication decision.
