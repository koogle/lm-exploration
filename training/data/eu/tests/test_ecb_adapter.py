import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.ecb import DATABASE, canonical_url, enumerate_collection


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
