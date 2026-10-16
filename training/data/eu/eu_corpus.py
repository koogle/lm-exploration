"""Bounded EU source discovery pilot. Raw content is local research material."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup, Comment
from lxml import etree

USER_AGENT = "EUCorpusResearch/0.1"
SCHEMA_VERSION = 1
EXTRACTION_VERSION = "0.2.0"


def normalize_text(text):
    return " ".join(text.split())


def clean_text(text):
    """Preserve nonempty paragraph/line boundaries while trimming spacing."""
    return "\n".join(normalize_text(line) for line in text.splitlines() if line.strip())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def clean_url(url):
    return urldefrag(url)[0]


def allowed(url, hosts):
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname in hosts
        and not parsed.username
        and not parsed.password
        and port in {None, 80, 443}
    )


def matches(url, patterns):
    return any(re.search(pattern, url) for pattern in patterns)


def parse_content(data, content_type, url, remove_selectors=None, content_selector=None):
    """Return text, title, language, links, kind; feeds never become documents."""
    media = content_type.split(";")[0].lower()
    if media == "application/pdf" or data.startswith(b"%PDF"):
        try:
            from pypdf import PdfReader
        except ImportError:
            return "", "", "", [], "pdf_unsupported"
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages), "", "", [], "pdf"
    if media == "text/plain":
        return data.decode("utf-8", errors="replace"), "", "", [], "text"
    xml_like = media != "text/html" and (
        "xml" in media
        or data.lstrip().startswith(b"<?xml")
        or data.lstrip().startswith((b"<rss", b"<feed", b"<urlset", b"<sitemapindex"))
    )
    if xml_like:
        root = etree.fromstring(
            data, parser=etree.XMLParser(resolve_entities=False, no_network=True)
        )
        local = etree.QName(root).localname.lower()
        links = []
        for node in root.iter():
            if not isinstance(node.tag, str):
                continue
            tag = etree.QName(node).localname.lower()
            if tag in {"link", "loc", "enclosure"}:
                value = node.get("href") or node.get("url") or (node.text or "").strip()
                if value:
                    links.append(urljoin(url, value))
        if local in {"rss", "feed", "rdf", "urlset", "sitemapindex"}:
            return "", "", "", links, "inventory"
        if local in {"html", "xhtml"}:
            return parse_content(data, "text/html", url, remove_selectors, content_selector)
        # Only recognize full-document XML families; arbitrary API metadata is not text.
        if local not in {"html", "xhtml", "formex", "act", "document", "article", "tei"}:
            return "", "", "", links, "xml_unclassified"
        return (
            " ".join(root.itertext()),
            "",
            root.get("{http://www.w3.org/XML/1998/namespace}lang", ""),
            links,
            "xml",
        )
    if "html" in media or data.lstrip().lower().startswith((b"<!doctype html", b"<html")):
        soup = BeautifulSoup(data, "lxml")
        links = [
            urljoin(url, element["href"])
            for element in soup.select("a[href], link[rel='alternate'][href]")
        ]
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        language = soup.html.get("lang", "") if soup.html else ""
        for element in soup.select("script, style, nav, header, footer, noscript, form, aside"):
            element.decompose()
        for selector in remove_selectors or []:
            for element in soup.select(selector):
                element.decompose()
        if content_selector:
            body = soup.select_one(content_selector)
            if body is None:
                return "", title, language, links, "content_selector_missing"
        else:
            body = soup.find("main") or soup.find("article") or soup.body or soup
        # Source indentation is whitespace, whereas block elements create boundaries.
        # Concatenation preserves punctuation and words split across inline markup.
        for node in list(body.find_all(string=True)):
            if isinstance(node, Comment):
                node.extract()
                continue
            node.replace_with(re.sub(r"\s+", " ", str(node)))
        for element in body.select(
            "p, div, li, tr, h1, h2, h3, h4, h5, h6, section, article, blockquote, pre, br"
        ):
            element.insert_before("\n")
            element.insert_after("\n")
        text = clean_text(body.get_text("", strip=False))
        signals = " ".join([title.lower(), text[:1000].lower()])
        blocked = (
            "verify you are human",
            "checking your browser",
            "access denied",
            "just a moment",
            "enable javascript and cookies",
            "captcha",
        )
        if any(signal in signals for signal in blocked):
            return "", title, language, links, "bot_or_error"
        return text, title, language, links, "html"
    return "", "", "", [], "unsupported"


class Collector:
    def __init__(
        self,
        output,
        client=None,
        delay=1.0,
        max_bytes=20_000_000,
        min_chars=200,
        clock=time.monotonic,
        sleep=time.sleep,
    ):
        if delay < 1:
            raise ValueError("delay must be at least 1 second")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "raw").mkdir()
        for name in ("documents", "attempts", "discoveries", "provenance"):
            (self.output / f"{name}.jsonl").touch()
        self.client = client or httpx.Client(
            timeout=30, follow_redirects=False, headers={"User-Agent": USER_AGENT}
        )
        self.delay, self.max_bytes, self.min_chars = delay, max_bytes, min_chars
        self.clock, self.sleep = clock, sleep
        self.last_request, self.robots = {}, {}
        self.counts = {"documents": 0, "characters": 0, "words": 0, "attempts": 0}

    def log(self, name, record):
        with (self.output / f"{name}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def fetch(self, url, source, robots_request=False):
        original = url
        record = {
            "source_id": source["id"],
            "url": original,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "robots_request": robots_request,
        }
        self.counts["attempts"] += 1
        try:
            for _ in range(6):
                if not allowed(url, source["allowed_hosts"]):
                    raise ValueError("out_of_scope_url_or_redirect")
                if not robots_request and not self.permitted(url, source):
                    raise ValueError("robots_denied_or_unavailable")
                host = urlsplit(url).hostname
                interval = max(self.delay, float(source.get("min_delay_seconds", 0)))
                if not robots_request:
                    policy = self.robots.get(f"{urlsplit(url).scheme}://{urlsplit(url).netloc}")
                    if policy:
                        interval = max(
                            interval, policy.crawl_delay(USER_AGENT) or policy.crawl_delay("*") or 0
                        )
                wait = interval - (self.clock() - self.last_request.get(host, -1e20))
                if wait > 0:
                    self.sleep(wait)
                self.last_request[host] = self.clock()
                headers = {"User-Agent": USER_AGENT}
                if not robots_request:
                    headers.update(
                        {
                            key: value
                            for key, value in source.get("request_headers", {}).items()
                            if key.lower() in {"accept", "accept-language"}
                        }
                    )
                with self.client.stream("GET", url, headers=headers) as response:
                    record.update(final_url=url, http_status=response.status_code)
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise ValueError("redirect_without_location")
                        url = clean_url(urljoin(url, location))
                        continue
                    if response.status_code != 200:
                        raise ValueError(f"http_{response.status_code}")
                    chunks, size = [], 0
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > self.max_bytes:
                            raise ValueError("body_exceeds_max_bytes")
                        chunks.append(chunk)
                    data = b"".join(chunks)
                    raw_hash = digest(data)
                    (self.output / "raw" / raw_hash).write_bytes(data)
                    record.update(
                        status="fetched",
                        bytes=size,
                        raw_sha256=raw_hash,
                        content_type=response.headers.get("content-type", ""),
                        etag=response.headers.get("etag"),
                        last_modified=response.headers.get("last-modified"),
                    )
                    self.log("attempts", record)
                    self.log("provenance", record)
                    return data, record
            raise ValueError("redirect_limit")
        except (httpx.HTTPError, ValueError) as error:
            record.update(status="failed", error=str(error))
            self.log("attempts", record)
            return None, record

    def permitted(self, url, source):
        parsed = urlsplit(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self.robots:
            data, record = self.fetch(origin + "/robots.txt", source, robots_request=True)
            policy = RobotFileParser()
            if data is not None:
                # A successful HTML challenge is not a valid robots policy.
                if b"<html" in data[:1000].lower() or b"<!doctype html" in data[:1000].lower():
                    self.log(
                        "discoveries",
                        {
                            "source_id": source["id"],
                            "url": origin + "/robots.txt",
                            "status": "invalid_robots_html",
                        },
                    )
                    policy = None
                else:
                    policy.parse(data.decode("utf-8", errors="replace").splitlines())
            elif record.get("http_status") in {404, 410} and record.get("error") in {
                "http_404",
                "http_410",
            }:
                policy.parse(["User-agent: *", "Allow: /"])
            else:
                policy = None
            self.robots[origin] = policy
        policy = self.robots[origin]
        return policy is not None and policy.can_fetch(USER_AGENT, url)

    def crawl(self, source, max_pages=10, max_depth=2):
        queue = deque((clean_url(url), 0) for url in source["seed_urls"])
        seen, fetched = set(), 0
        while queue and fetched < max_pages:
            url, depth = queue.popleft()
            if url in seen:
                continue
            seen.add(url)
            if not allowed(url, source["allowed_hosts"]):
                self.log(
                    "discoveries", {"source_id": source["id"], "url": url, "status": "out_of_scope"}
                )
                continue
            fetched += 1
            data, provenance = self.fetch(url, source)
            if data is None:
                self.log(
                    "discoveries",
                    {
                        "source_id": source["id"],
                        "url": url,
                        "status": "fetch_failed",
                        "reason": provenance.get("error"),
                    },
                )
                continue
            final_url = provenance["final_url"]
            try:
                text, title, language, links, kind = parse_content(
                    data,
                    provenance["content_type"],
                    final_url,
                    source.get("remove_selectors"),
                    source.get("content_selector"),
                )
            except Exception as error:  # noqa: BLE001 -- isolate malformed documents; record every extraction failure
                self.log(
                    "discoveries",
                    {
                        "source_id": source["id"],
                        "url": url,
                        "status": "parse_failed",
                        "reason": str(error),
                    },
                )
                continue
            text = clean_text(text)
            is_document = any(
                matches(candidate, source.get("document_url_patterns", []))
                for candidate in (url, final_url)
            )
            accepted = (
                is_document
                and kind in {"html", "xml", "text", "pdf"}
                and len(text) >= self.min_chars
            )
            self.log(
                "discoveries",
                {
                    "source_id": source["id"],
                    "url": url,
                    "final_url": final_url,
                    "depth": depth,
                    "kind": kind,
                    "document_candidate": is_document,
                    "status": "extracted" if accepted else "discovery_or_rejected",
                    "characters": len(text),
                },
            )
            if accepted:
                row = {
                    "source_id": source["id"],
                    "institution": source.get("institution", source["name"]),
                    "url": final_url,
                    "requested_url": url,
                    "title": title,
                    "language": language or source.get("language", ""),
                    "text": text,
                    "text_sha256": digest(normalize_text(text).encode()),
                    "characters": len(text),
                    "words": len(text.split()),
                    "format": kind,
                    "provenance": provenance,
                    "schema_version": SCHEMA_VERSION,
                    "extraction_version": EXTRACTION_VERSION,
                    "license_notes": source.get(
                        "license_notes", "Unreviewed; source metadata retained for research."
                    ),
                }
                self.log("documents", row)
                self.counts["documents"] += 1
                self.counts["characters"] += row["characters"]
                self.counts["words"] += row["words"]
            for link in dict.fromkeys(clean_url(link) for link in links):
                in_scope = allowed(link, source["allowed_hosts"])
                relevant = matches(
                    link,
                    source.get("document_url_patterns", [])
                    + source.get("discovery_url_patterns", []),
                )
                enqueue = in_scope and relevant and depth < max_depth and link not in seen
                self.log(
                    "discoveries",
                    {
                        "source_id": source["id"],
                        "url": link,
                        "parent_url": final_url,
                        "depth": depth + 1,
                        "status": "queued" if enqueue else "not_queued",
                    },
                )
                if enqueue:
                    queue.append((link, depth + 1))
        return {
            "source_id": source["id"],
            "pages_attempted": fetched,
            "remaining_queue_entries": len(queue),
        }


def export(input_dir, output):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = {}
    for line in (Path(input_dir) / "documents.jsonl").read_text().splitlines():
        document = json.loads(line)
        text = clean_text(document["text"])
        key = digest(normalize_text(text).encode())
        if key not in rows:
            rows[key] = {
                "id": key,
                "text": text,
                "title": document["title"],
                "language": document["language"],
                "characters": len(text),
                "words": len(text.split()),
                "urls": [],
                "source_ids": [],
                "provenance": [],
            }
        row = rows[key]
        for url in (document["url"], document.get("requested_url", document["url"])):
            if url not in row["urls"]:
                row["urls"].append(url)
        if document["source_id"] not in row["source_ids"]:
            row["source_ids"].append(document["source_id"])
        row["provenance"].append({key: value for key, value in document.items() if key != "text"})
    fields = [
        ("id", pa.string()),
        ("text", pa.string()),
        ("title", pa.string()),
        ("language", pa.string()),
        ("characters", pa.int64()),
        ("words", pa.int64()),
        ("urls", pa.list_(pa.string())),
        ("source_ids", pa.list_(pa.string())),
        ("provenance_json", pa.string()),
    ]
    output_rows = []
    for row in rows.values():
        row["provenance_json"] = json.dumps(row.pop("provenance"), ensure_ascii=False)
        output_rows.append(row)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(output_rows, schema=pa.schema(fields)), output, compression="zstd"
    )
    return {
        "unique_documents": len(rows),
        "characters": sum(r["characters"] for r in output_rows),
        "words": sum(r["words"] for r in output_rows),
        "output": str(output),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    crawl = sub.add_parser("crawl")
    crawl.add_argument("--sources", type=Path, default=Path("sources.json"))
    crawl.add_argument("--source", action="append", required=True)
    crawl.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Per-source content attempts; robots and redirects are additional",
    )
    crawl.add_argument("--max-depth", type=int, default=2)
    crawl.add_argument("--delay", type=float, default=1.0)
    crawl.add_argument("--max-bytes", type=int, default=20_000_000)
    crawl.add_argument("--min-chars", type=int, default=200)
    crawl.add_argument("--output", type=Path, required=True)
    exp = sub.add_parser("export")
    exp.add_argument("--input", type=Path, required=True)
    exp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        print(json.dumps(export(args.input, args.output), indent=2))
        return
    if args.max_pages < 1 or args.max_depth < 0 or args.max_bytes < 1 or args.min_chars < 1:
        parser.error("Bounds must be positive (max-depth may be zero)")
    manifest = json.loads(args.sources.read_text())
    if manifest.get("schema_version") != 1:
        parser.error("Expected source manifest schema_version 1")
    sources = {source["id"]: source for source in manifest["sources"]}
    unknown = set(args.source) - sources.keys()
    if unknown:
        parser.error(f"Unknown sources: {sorted(unknown)}")
    collector = Collector(
        args.output, delay=args.delay, max_bytes=args.max_bytes, min_chars=args.min_chars
    )
    run = {
        "schema_version": SCHEMA_VERSION,
        "extraction_version": EXTRACTION_VERSION,
        "collector_sha256": digest(Path(__file__).read_bytes()),
        "created_at": datetime.now(UTC).isoformat(),
        "user_agent": USER_AGENT,
        "sources": [sources[key] for key in dict.fromkeys(args.source)],
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "refresh_policy": "Immutable runs; a new output directory is required for every crawl. No conditional HTTP refresh yet.",
    }
    (args.output / "run.json").write_text(json.dumps(run, indent=2))
    try:
        source_results = [
            collector.crawl(source, args.max_pages, args.max_depth) for source in run["sources"]
        ]
    finally:
        collector.client.close()
    summary = {
        **collector.counts,
        "sources": source_results,
        "counting_note": "Counts are before cross-document deduplication. Words are whitespace units, not model tokens.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
