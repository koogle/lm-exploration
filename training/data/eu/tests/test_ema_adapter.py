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


@pytest.mark.parametrize("collection,kind", list(ema.EPAR_SERIES.items()))
def test_epar_series_exact_filter_preserves_identity(collection, kind):
    selected = document(
        kind,
        "812",
        medicine_name="Example medicine",
        ema_product_number="EMEA/H/C/000001",
        translations={"fr": f"https://www.ema.europa.eu/fr/documents/{kind}/example_fr.pdf"},
    )
    items = [selected, document("variation-report", "813")]
    result = ema.enumerate_collection(collection, inventory(items), limit=100000)
    records = result["records"]
    assert len(records) == 2
    assert {r["language"] for r in records} == {"en", "fr"}
    assert {r["document_type"] for r in records} == {kind}
    assert records[0]["ema_product_number"] == "EMEA/H/C/000001"
    assert records[0]["medicine_name"] == "Example medicine"
    assert result["enumeration"]["complete"]
    assert result["enumeration"]["overlaps_collections"] == ["epar_documents"]
    umbrella = ema.enumerate_collection("epar_documents", inventory(items))["records"]
    assert {r["document_id"] for r in records} <= {r["document_id"] for r in umbrella}


def test_epar_bound_is_machine_readable():
    result = ema.enumerate_collection(
        "epar_overviews", inventory([document("overview", "1"), document("overview", "2")]), limit=1
    )
    summary = result["enumeration"]
    assert summary["selection_truncated"]
    assert summary["unselected_count"] == 1
    assert summary["selection_limit"] == 1
    assert not summary["complete"]


@pytest.mark.parametrize("extension", ["xlsx", "xls", "docx", "doc", "zip", "unknown"])
def test_unsupported_document_formats_are_coverage_gaps(extension):
    rows = [
        document("agenda", "1"),
        document(
            "agenda",
            "2",
            document_url=(
                f"https://www.ema.europa.eu/en/documents/agenda/attachment_en.{extension}"
            ),
        ),
    ]
    result = ema.enumerate_collection("committee_documents", inventory(rows), limit=100000)
    assert len(result["records"]) == 1
    summary = result["enumeration"]
    assert summary["skipped_unsupported_formats"] == {extension: 1}
    assert summary["skipped_unsupported_format_count"] == 1
    assert summary["matching_inventory_records"] == 2
    assert not summary["complete"]
    assert not summary["selection_truncated"]


@pytest.mark.parametrize("extension", ["pdf", "txt", "xml", "html", "htm", "xhtml"])
def test_native_document_format_survives(extension):
    result = ema.enumerate_collection(
        "committee_documents",
        inventory(
            [document(document_url=f"https://www.ema.europa.eu/en/documents/agenda/a.{extension}")]
        ),
    )
    assert result["records"][0]["format"] == extension
    assert result["enumeration"]["complete"]


@pytest.mark.parametrize("suffix", ["pdf-0", "pdf-1", "pdf-23"])
def test_drupal_pdf_collision_suffixes_are_retained(suffix):
    result = ema.enumerate_collection(
        "committee_documents",
        inventory(
            [document(document_url=f"https://www.ema.europa.eu/en/documents/agenda/a.{suffix}")]
        ),
    )
    assert result["records"][0]["format"] == "pdf"
    assert result["enumeration"]["complete"]


@pytest.mark.parametrize(
    "url,supported",
    [
        ("https://www.ema.europa.eu/en/news/a-story", True),
        ("https://www.ema.europa.eu/fr/documents/overview/a.pdf-0", True),
        ("https://www.ema.europa.eu/en/documents/minutes/a.xlsx", False),
        ("https://www.ema.europa.eu/en/documents/minutes/a.docx", False),
        ("https://www.ema.europa.eu/en/documents/minutes/a.zip", False),
        ("https://www.ema.europa.eu/en/documents/minutes/unknown", False),
        ("https://invalid.example/a.pdf", False),
    ],
)
def test_old_manifest_supported_record_ignores_legacy_format(url, supported):
    assert ema.supported_record({"url": url, "format": "html"}) is supported
