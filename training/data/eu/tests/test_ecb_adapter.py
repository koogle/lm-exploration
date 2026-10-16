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
