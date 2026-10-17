# EUR-Lex / CELLAR: published laws

Probed **2026-10-16, 08:59 UTC** from the managed workspace using Python urllib.
Ten top-level GETs, with automatic redirects and a 3 MB read limit. Exact URLs,
headers, timestamps and responses are in [eurlex.json](eurlex.json). Downloaded
samples are local ignored artifacts, not a committed corpus.

## Finding

**Native full-text XHTML is retrievable through CELLAR, even though EUR-Lex's
front-end returned empty HTTP 202 responses in this environment.** This is a
working multilingual laws source, but a general HTML crawl does not yet give us
a working bulk inventory. The following sample is GDPR as originally published,
CELEX `32016R0679`; it is not a current consolidated-text guarantee.

| Request | Result | Interpretation |
| --- | --- | --- |
| EUR-Lex recent legislation collection | 202, HTML type, 0 bytes | No discovery links returned |
| EUR-Lex GDPR HTML, EN and FR | Both 202, HTML type, 0 bytes | No document text returned |
| EUR-Lex robots.txt | 202, HTML type, 0 bytes | Robots policy unresolved; not an allow-all result |
| CELEX resource with `?language=eng`, no format headers | 200, RDF/XML, read capped at 3 MB | Metadata graph, **not** legal full text |
| CELEX resource, `Accept: application/xhtml+xml`, `Accept-Language: eng` | 200, XHTML, 806,864 bytes | Full English legal text |
| Same negotiation, `Accept-Language: fra` | 200, XHTML, 876,858 bytes | Full French legal text |
| Direct HTTPS English CELLAR `DOC_1` URL | 200, XHTML, 806,864 bytes | Works without content-negotiation headers |
| CELLAR publications documentation | 200, HTML | Confirms content negotiation mechanism |
| CELLAR feeds documentation | 200, HTML | Describes changed publications/metadata notifications; actual feed not tested |

The empty 202 responses are consistent with a front-end access or bot-handling
issue; they do not establish the exact cause. No challenge bypass was attempted.
All findings above are local HTTP observations, not browser/search snippets.

## Working content retrieval

Request `https://publications.europa.eu/resource/celex/32016R0679` with:

```text
Accept: application/xhtml+xml
Accept-Language: eng
```

Use `fra` for the French version. The service redirected to HTTP content URLs;
the English equivalent was also independently tested over HTTPS:

- [English document](https://publications.europa.eu/resource/cellar/3e485e15-11bd-11e6-ba9a-01aa75ed71a1.0006.03/DOC_1)
- [French document](https://publications.europa.eu/resource/cellar/3e485e15-11bd-11e6-ba9a-01aa75ed71a1.0009.03/DOC_1) (HTTP redirect target tested; HTTPS form not separately requested)
- [Retrieval documentation](https://op.europa.eu/en/web/cellar/cellar-data/publications)
- [Update-feed documentation](https://op.europa.eu/en/web/cellar/cellar-data/rss-and-atom-feeds)

Both responses parse as XHTML. Joining body text and normalizing whitespace
produced 350,648 characters / 55,013 whitespace-separated words in English and
405,299 characters / 61,903 words in French. These are **not tokenizer counts**,
quality-filtered yields, or representative document-length estimates. Text
contains the regulation heading, recitals, articles, footnotes and journal
context; extraction should preserve their structure and avoid duplicate
footnote or paragraph rendering.

## Collector implications

The JSON source proposal supplies both the real but presently blocked collection
seed and a working direct document seed for a bounded generic crawler. A direct
sample proves extraction, not archive enumeration. Do not label this source
fully integrated or infer corpus coverage from these two translations.

Build a CELLAR-specific discovery adapter next: enumerate legal works and their
expressions/manifestations from the official metadata interfaces, resolve the
preferred full-text manifestation, then fetch its concrete content URL. Preserve
CELEX/work ID, CELLAR expression/manifestation ID, language, version/date,
source URL and content hash. A shared CELEX work ID makes translations related,
not duplicate text. Distinguish original acts, amendments and consolidations.

A generic HTML extractor must **reject RDF metadata as training text**. An XML
MIME type alone does not establish a legal document. Likewise the documentation
page is evidence about retrieval, not an eligible document seed for the corpus.

The documented notification service can eventually drive incremental updates;
its actual query parameters, retention window, cursor behavior and completeness
remain to be tested. A feed only covers changes, so a separate backfill inventory
is necessary. This probe establishes neither total laws coverage nor incremental
novelty versus existing EUR-Lex datasets. No total token estimate is inferred.

Public retrieval is verified; record-level reuse/provenance review and release
packaging remain separate. No training registry eligibility, Modal job,
Hugging Face upload, or scheduled refresh was enabled by this investigation.
