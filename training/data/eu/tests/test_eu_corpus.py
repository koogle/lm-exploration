import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eu_corpus import Collector, allowed, export, parse_content

SOURCE = {
    "id": "test",
    "name": "Test",
    "seed_urls": ["https://eu.test/index"],
    "allowed_hosts": ["eu.test"],
    "document_url_patterns": [r"/doc/"],
    "discovery_url_patterns": [r"/index", r"/feed"],
}


def collector(tmp_path, handler, **kwargs):
    elapsed = [0.0]

    def sleep(seconds):
        elapsed[0] += seconds

    result = Collector(
        tmp_path / "run",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=lambda: elapsed[0],
        sleep=sleep,
        **kwargs,
    )
    return result, elapsed


def records(path, name):
    return [json.loads(line) for line in (path / f"{name}.jsonl").read_text().splitlines()]


def test_bounds_classification_and_rate(tmp_path):
    calls = []

    def handle(request):
        calls.append(str(request.url))
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\nCrawl-delay: 2")
        if request.url.path == "/index":
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="""<html><main>Index
                <a href="/doc/1">one</a><a href="/doc/2">two</a><a href="https://other.test/doc/3">external</a>
                </main></html>""",
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<html><nav>menu</nav><main>" + "real text " * 40 + "</main></html>",
        )

    c, elapsed = collector(tmp_path, handle)
    result = c.crawl(SOURCE, max_pages=2)
    assert result["pages_attempted"] == 2
    assert len(calls) == 3  # robots plus bounded content pages
    docs = records(c.output, "documents")
    assert len(docs) == 1 and "menu" not in docs[0]["text"]
    assert elapsed[0] >= 4


def test_redirect_cannot_escape_allowlist(tmp_path):
    calls = []

    def handle(request):
        calls.append(request.url.host)
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(302, headers={"location": "https://private.test/doc/secret"})

    c, _ = collector(tmp_path, handle)
    data, record = c.fetch("https://eu.test/doc/1", SOURCE)
    assert data is None and "out_of_scope" in record["error"]
    assert set(calls) == {"eu.test"}


@pytest.mark.parametrize(
    "status,body",
    [(503, "down"), (200, "<html>challenge</html>"), (200, "User-agent: *\nDisallow: /")],
)
def test_robots_fail_closed(tmp_path, status, body):
    calls = []

    def handle(request):
        calls.append(request.url.path)
        return httpx.Response(status, text=body)

    c, _ = collector(tmp_path, handle)
    assert c.fetch("https://eu.test/doc/1", SOURCE)[0] is None
    assert calls == ["/robots.txt"]


def test_size_limit_and_immutable_runs(tmp_path):
    def handle(request):
        return (
            httpx.Response(404)
            if request.url.path == "/robots.txt"
            else httpx.Response(200, content=b"x" * 100)
        )

    c, _ = collector(tmp_path, handle, max_bytes=50)
    assert c.fetch("https://eu.test/doc/1", SOURCE)[1]["error"] == "body_exceeds_max_bytes"
    with pytest.raises(FileExistsError):
        Collector(c.output)


def test_feed_sitemap_metadata_and_bot_parsing():
    feed = b"<rss><channel><item><title>not a document</title><link>https://eu.test/doc/1</link></item></channel></rss>"
    result = parse_content(feed, "application/rss+xml", "https://eu.test/feed")
    assert result[0] == "" and result[3] == ["https://eu.test/doc/1"] and result[4] == "inventory"
    sitemap = b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://eu.test/doc/2</loc></url></urlset>'
    assert parse_content(sitemap, "text/xml", "https://eu.test/sitemap")[3] == [
        "https://eu.test/doc/2"
    ]
    assert (
        parse_content(b"<metadata><title>test</title></metadata>", "text/xml", "x")[4]
        == "xml_unclassified"
    )
    assert (
        parse_content(b"<html><title>Access denied</title></html>", "text/html", "x")[4]
        == "bot_or_error"
    )
    assert not allowed("https://eu.test:bad/doc/1", ["eu.test"])


def test_export_dedup_retains_aliases_and_empty_schema(tmp_path):
    import pyarrow.parquet as pq

    (tmp_path / "documents.jsonl").touch()
    export(tmp_path, tmp_path / "empty.parquet")
    empty = pq.read_table(tmp_path / "empty.parquet")
    assert empty.num_rows == 0 and "provenance_json" in empty.column_names
    base = {
        "source_id": "a",
        "title": "T",
        "language": "en",
        "text": "same text",
        "url": "https://a",
        "provenance": {"raw_sha256": "a"},
    }
    duplicate = {
        **base,
        "source_id": "b",
        "text": "same   text\n",
        "url": "https://b",
        "provenance": {"raw_sha256": "b"},
    }
    (tmp_path / "documents.jsonl").write_text(
        "\n".join(json.dumps(row) for row in [base, duplicate])
    )
    assert export(tmp_path, tmp_path / "docs.parquet")["unique_documents"] == 1
    row = pq.read_table(tmp_path / "docs.parquet").to_pylist()[0]
    assert row["urls"] == ["https://a", "https://b"]
    assert len(json.loads(row["provenance_json"])) == 2


def test_xhtml_cleanup_and_paragraphs():
    from eu_corpus import clean_text

    xhtml = b"""<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml" lang="fr">
    <head><title>Official title</title><style>body { color: red }</style></head>
    <body><nav>Navigation</nav><main><p>First paragraph.</p><p>Second paragraph.</p>
    <script>secret_code()</script></main></body></html>"""
    text, title, language, _, kind = parse_content(
        xhtml, "application/xhtml+xml", "https://eu.test/doc/1"
    )
    assert clean_text(text) == "First paragraph.\nSecond paragraph."
    assert title == "Official title" and language == "fr" and kind == "html"


def test_inline_markup_selectors_and_links():
    html = b"""<html><body><div>Outside</div><main><div class="report">
    <p>This is <em>important</em>: follow <a href="/doc/2">the link</a>.</p>
    <p>Next\n    paragraph.</p><!-- omitted comment --><div class="related"><a href="/doc/3">Related</a></div>
    </div></main></body></html>"""
    text, _, _, links, kind = parse_content(
        html,
        "text/html",
        "https://eu.test/doc/1",
        remove_selectors=[".related"],
        content_selector=".report",
    )
    assert text == "This is important: follow the link.\nNext paragraph."
    assert links == ["https://eu.test/doc/2", "https://eu.test/doc/3"]
    assert kind == "html"
    assert (
        parse_content(html, "text/html", "https://eu.test/doc/1", content_selector=".absent")[4]
        == "content_selector_missing"
    )


def test_source_delay_is_respected(tmp_path):
    def handle(request):
        return (
            httpx.Response(404)
            if request.url.path == "/robots.txt"
            else httpx.Response(200, text="text")
        )

    c, elapsed = collector(tmp_path, handle)
    assert c.fetch("https://eu.test/doc/1", {**SOURCE, "min_delay_seconds": 5})[0] == b"text"
    assert elapsed[0] == 5


def test_rate_limit_retains_retry_after(tmp_path):
    def handle(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(429, headers={"Retry-After": "1200"})

    c, _ = collector(tmp_path, handle)
    data, record = c.fetch("https://eu.test/doc/1", SOURCE)
    assert data is None
    assert record["error"] == "http_429"
    assert record["retry_after"] == "1200"
    assert records(c.output, "attempts")[-1]["retry_after"] == "1200"
