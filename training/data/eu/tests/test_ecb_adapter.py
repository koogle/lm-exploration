import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.ecb import (
    COLLECTIONS,
    DATABASE,
    _representations,
    canonical_url,
    enumerate_collection,
)


def fixture_fetch(rows):
    header = ["id", "pub_timestamp", "type", "documentTypes", "publicationProperties"]
    snapshot = DATABASE + "v/h/"
    payloads = {
        DATABASE + "versions.json": [{"version": "v", "hash": "h"}],
        snapshot + "metadata.json": {
            "header": header,
            "total_records": len(rows),
            "chunk_size": 1,
            "chunk_group_size": 1,
        },
        snapshot + "indexes/type/index.json": [
            {"value": "19", "total_records": len(rows), "first_sort_id_per_chunk": [0]}
        ],
        snapshot + "indexes/type/0/chunk_0.json": list(range(len(rows))),
    }
    for i, row in enumerate(rows):
        payloads[snapshot + f"data/{i}/chunk_{i}.json"] = row
    calls = []

    def fetch(url):
        calls.append(url)
        return json.dumps(payloads[url]).encode(), {}

    return fetch, calls


def row(n, files):
    return [n, 1700000000, 19, files, {"Title": f"Speech {n}"}]


def test_native_preference_translations_and_canonicalization():
    fetch, _ = fixture_fetch(
        [
            row(
                1,
                [
                    "//www.ecb.europa.eu//press/key/a.en.pdf",
                    "/press/key/a.en.html",
                    "/press/key/a.fr.pdf",
                    "https://evil.test/a.en.html",
                ],
            )
        ]
    )
    result = enumerate_collection("speeches", fetch)
    assert result["enumeration"]["complete"] is True
    assert result["enumeration"]["reported_total"] == 1
    assert [(r["language"], r["format"]) for r in result["records"]] == [
        ("en", "html"),
        ("fr", "pdf"),
    ]
    assert result["records"][0]["url"] == "https://www.ecb.europa.eu/press/key/a.en.html"


def test_limit_does_not_claim_complete_and_higher_limit_backfills():
    rows = [row(1, ["/press/key/a.en.html"]), row(2, ["/press/key/b.en.html"])]
    fetch, _ = fixture_fetch(rows)
    limited = enumerate_collection("speeches", fetch, limit=1)
    assert not limited["enumeration"]["complete"]
    fetch, calls = fixture_fetch(rows)
    full = enumerate_collection("speeches", fetch, limit=2)
    assert full["enumeration"]["complete"]
    assert len(full["records"]) == 2
    assert full["records"][0]["document_id"] == limited["records"][0]["document_id"]
    assert any("/data/1/chunk_1.json" in url for url in calls)


def test_fetch_failure_propagates():
    def broken(url):
        raise RuntimeError("network interruption")

    with pytest.raises(RuntimeError, match="interruption"):
        enumerate_collection("speeches", broken)


def test_canonical_external_host_and_content_fragment():
    assert canonical_url("https://elsewhere.test/x") is None
    fetch, _ = fixture_fetch([row(1, ["/press/key/a_content.en.html"])])
    assert enumerate_collection("speeches", fetch)["records"] == []


def test_verified_research_series_and_historical_pdf_forms():
    assert COLLECTIONS["working_papers"]["type_ids"] == ["2"]
    assert COLLECTIONS["occasional_papers"]["type_ids"] == ["3"]
    assert COLLECTIONS["economic_bulletin_articles"]["type_ids"] == ["24"]
    assert COLLECTIONS["economic_bulletin_boxes"]["type_ids"] == ["25"]
    urls = [
        "/pub/pdf/scpwps/ecbwp001.pdf",
        "/press/economic-bulletin/articles/1999/pdf/p.11_16_mb199901en.pdf",
        "/pub/pdf/not-a-document.xlsx",
    ]
    records = _representations({"documentTypes": urls})
    assert len(records) == 2
    assert all(language == "und" and fmt == "pdf" for _, language, fmt in records)
    assert {url for url, _, _ in records} == {
        "https://www.ecb.europa.eu" + path for path in urls[:2]
    }


def test_alternatives_only_for_verified_standalone_articles():
    from adapters.ecb import VERIFIED_HISTORICAL_ARTICLES, alternate_document_urls

    for original, resolved_url in VERIFIED_HISTORICAL_ARTICLES.items():
        record = {
            "url": original,
            "document_id": "stable",
            "collection_id": "economic_bulletin_articles",
        }
        assert alternate_document_urls(record) == [resolved_url]
        assert record["url"] == original
        assert record["document_id"] == "stable"
        assert alternate_document_urls({**record, "collection_id": "other"}) == []
    assert (
        alternate_document_urls(
            {
                "url": "https://www.ecb.europa.eu/press/economic-bulletin/articles/2014/pdf/unverified.pdf",
                "collection_id": "economic_bulletin_articles",
            }
        )
        == []
    )


def test_failed_probe_is_only_inferred_not_trusted():
    from adapters.ecb import VERIFIED_HISTORICAL_ARTICLES, alternate_document_urls

    url = "https://www.ecb.europa.eu/press/economic-bulletin/articles/2014/pdf/art1_mb201411_pp79-90.en.pdf"
    record = {"url": url, "collection_id": "economic_bulletin_articles"}
    assert url not in VERIFIED_HISTORICAL_ARTICLES
    assert alternate_document_urls(record) == [
        "https://www.ecb.europa.eu/pub/pdf/other/art1_mb201411_pp79-90.en.pdf"
    ]


@pytest.mark.parametrize(
    "name",
    [
        "art2_mb201201en_pp75-86en.pdf",
        "art1_mb201201.en.pdf",
        "pp75_86_mb201201en.pdf",
        "p75-86_mb201201en.pdf",
        "mb201201en_pp75-86en.pdf",
        "mb201201_75-86en.pdf",
    ],
)
def test_monthly_filename_families(name):
    from adapters.ecb import alternate_document_urls

    candidate = {
        "url": f"https://www.ecb.europa.eu/press/economic-bulletin/articles/2012/pdf/{name}",
        "collection_id": "economic_bulletin_articles",
    }
    assert alternate_document_urls(candidate) == ["https://www.ecb.europa.eu/pub/pdf/other/" + name]


@pytest.mark.parametrize(
    "path",
    [
        "2012/pdf/art2_mb201101en_pp75-86en.pdf",  # year mismatch
        "2012/pdf/art2_mb201213en_pp75-86en.pdf",  # bad month
        "2012/pdf/art2_mb201201en_pp86-75en.pdf",  # reversed pages
        "2012/pdf/art2_mb201201en_pp0-75en.pdf",  # zero page
        "2015/pdf/art2_mb201501en_pp75-86en.pdf",  # after Monthly Bulletin
        "2012/pdf/mb201201en.pdf",  # whole issue
        "2012/pdf/art2_mb201201en_pp75-86en.pdf?x=1",  # query not allowed
        "2012/pdf/../art2_mb201201en_pp75-86en.pdf",  # nested path
    ],
)
def test_inferred_rule_exclusions(path):
    from adapters.ecb import alternate_document_urls

    assert (
        alternate_document_urls(
            {
                "url": "https://www.ecb.europa.eu/press/economic-bulletin/articles/" + path,
                "collection_id": "economic_bulletin_articles",
            }
        )
        == []
    )


def test_inferred_pdf_requires_title_in_first_two_pages(monkeypatch):
    from types import SimpleNamespace

    from adapters import ecb

    candidate = {
        "url": "https://www.ecb.europa.eu/press/economic-bulletin/articles/2012/pdf/art2_mb201201en_pp75-86en.pdf",
        "collection_id": "economic_bulletin_articles",
        "title": "The euro-area credit market",
    }
    pages = ["Cover", "THE EURO AREA\nCREDIT MARKET", "third page"]
    monkeypatch.setattr(
        ecb,
        "PdfReader",
        lambda data: SimpleNamespace(
            pages=[SimpleNamespace(extract_text=lambda text=text: text) for text in pages]
        ),
    )
    ecb.validate_alternate(candidate, b"%PDF-1.4 fake fixture")
    pages[:] = ["Cover", "Wrong article", candidate["title"]]
    with pytest.raises(ValueError, match="title mismatch"):
        ecb.validate_alternate(candidate, b"%PDF-1.4 fake fixture")
    with pytest.raises(ValueError, match="not a PDF"):
        ecb.validate_alternate(candidate, b"<html>error</html>")
    with pytest.raises(ValueError, match="too short"):
        ecb.validate_alternate({**candidate, "title": "Credit"}, b"%PDF-1.4")
    # Exact mappings use already recorded evidence, not this inferred-body check.
    ecb.validate_alternate({**candidate, "url": next(iter(ecb.VERIFIED_HISTORICAL_ARTICLES))}, b"")
