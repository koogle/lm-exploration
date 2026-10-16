import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters import ema


def inventory(items, total=None):
    payload = {"meta": {"total_records": len(items) if total is None else total}, "data": items}
    return lambda url: (json.dumps(payload).encode(), {"url": url})


def document(kind="agenda", id="42", **kwargs):
    return {
        "id": id,
        "name": "A committee document",
        "type": kind,
        "document_url": f"https://www.ema.europa.eu/en/documents/{kind}/example-{id}_en.pdf",
        "first_published_date": "2024-01-01T00:00:00Z",
        **kwargs,
    }


def test_partition_and_translations():
    items = [
        document(
            translations={"fr": "https://www.ema.europa.eu/fr/documents/agenda/example_fr.pdf"}
        ),
        document("scientific-guideline", "43"),
        document("report", "44"),
    ]
    counts = {}
    for collection in ["committee_documents", "scientific_guidelines", "non_epar_documents"]:
        result = ema.enumerate_collection(collection, inventory(items))
        counts[collection] = len(result["records"])
        assert result["enumeration"]["complete"]
    assert counts == {"committee_documents": 2, "scientific_guidelines": 1, "non_epar_documents": 1}
    records = ema.enumerate_collection("committee_documents", inventory(items))["records"]
    assert {r["document_id"] for r in records} == {"ema:42:en", "ema:42:fr"}


def test_limit_latest_dates_and_stable_news_ids():
    items = [
        {
            "news_url": f"https://www.ema.europa.eu/en/news/article-{i}",
            "title": f"Article {i}",
            "first_published_date": date,
            "press_release": "Yes",
        }
        for i, date in enumerate(["31/12/2024", "01/01/2025"])
    ]
    result = ema.enumerate_collection("news", inventory(items), limit=1)
    record = result["records"][0]
    assert record["published_at"] == "2025-01-01"
    assert record["document_type"] == "press-release"
    assert record["format"] == "html"
    assert not result["enumeration"]["complete"]
    assert result["enumeration"]["available_document_language_records"] == 2
    assert (
        record["document_id"]
        == ema.enumerate_collection("news", inventory(list(reversed(items))), limit=1)["records"][
            0
        ]["document_id"]
    )


def test_duplicate_urls_and_invalid_host_are_explicit():
    doc = document()
    result = ema.enumerate_collection(
        "committee_documents",
        inventory([doc, doc, document("agenda", "43", document_url="https://evil.invalid/a.pdf")]),
    )
    assert len(result["records"]) == 1
    assert result["enumeration"]["skipped_urls"] == 1
    assert not result["enumeration"]["complete"]


def test_truncated_inventory_and_bad_input_fail():
    with pytest.raises(ValueError, match="count mismatch"):
        ema.enumerate_collection("news", inventory([], total=10))
    with pytest.raises(ValueError, match="Unknown"):
        ema.enumerate_collection("missing", inventory([]))
    with pytest.raises(ValueError, match="non-negative"):
        ema.enumerate_collection("news", inventory([]), limit=-1)
