import json
from urllib.parse import parse_qs, urlsplit

import pytest

from adapters import eurlex


def row(number, language="ENG"):
    work = "http://publications.europa.eu/resource/cellar/00000000-0000-0000-0000-000000000001"
    expression = f"{work}.{number:04d}"
    values = {
        "work": work,
        "e": expression,
        "m": expression + ".03",
        "item": expression + ".03/DOC_1",
        "celex": "32024R0001",
        "date": "2024-01-01",
        "lang": "http://publications.europa.eu/resource/authority/language/" + language,
    }
    return {key: {"value": value} for key, value in values.items()}


def pages_fetch(pages, queries):
    def fetch(url):
        queries.append(parse_qs(urlsplit(url).query)["query"][0])
        return json.dumps({"results": {"bindings": pages.pop(0)}}).encode(), {"url": url}

    return fetch


def test_keyset_pagination_preserves_languages_and_concrete_items(monkeypatch):
    monkeypatch.setattr(eurlex, "PAGE_SIZE", 2)
    queries = []
    result = eurlex.enumerate_collection(
        "legislation_2024", pages_fetch([[row(1), row(2, "FRA")], [row(3, "BUL")]], queries), 5
    )
    assert result["enumeration"]["complete"]
    assert result["enumeration"]["last_item_cursor"] == row(3)["item"]["value"]
    assert [r["language"] for r in result["records"]] == ["en", "fr", "bg"]
    assert 'FILTER(STR(?item) > "' in queries[1]
    assert all(
        r["url"].startswith("https://publications.europa.eu/resource/cellar/")
        for r in result["records"]
    )
    assert len({r["document_id"] for r in result["records"]}) == 3
    assert all(r["format"] == "html" for r in result["records"])


def test_limit_never_claims_exhaustion():
    result = eurlex.enumerate_collection("legislation_2024", pages_fetch([[row(1)]], []), 1)
    assert not result["enumeration"]["complete"]
    assert result["enumeration"]["enumerated_count"] == 1


def test_repeated_page_rejected(monkeypatch):
    monkeypatch.setattr(eurlex, "PAGE_SIZE", 1)
    with pytest.raises(ValueError, match="advance"):
        eurlex.enumerate_collection("legislation_2024", pages_fetch([[row(1)], [row(1)]], []), 2)


def test_external_item_rejected():
    bad = row(1)
    bad["item"]["value"] = "https://example.com/metadata.xml"
    with pytest.raises(ValueError, match="unexpected content"):
        eurlex.enumerate_collection("legislation_2024", pages_fetch([[bad]], []), 1)


def test_zero_bound_and_invalid_arguments():
    result = eurlex.enumerate_collection("legislation_2024", lambda _: pytest.fail("network"), 0)
    assert result["records"] == []
    assert not result["enumeration"]["complete"]
    with pytest.raises(ValueError):
        eurlex.enumerate_collection("invalid", None)
    with pytest.raises(ValueError):
        eurlex.enumerate_collection("legislation_2024", None, -1)
