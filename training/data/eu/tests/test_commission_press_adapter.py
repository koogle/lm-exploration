import json
from urllib.parse import parse_qs, urlsplit

import pytest

from adapters import commission_press as cp


def row(reference="IP/24/1", language="EN", translated=False):
    return {
        "ky": reference,
        "refCode": reference,
        "languageCode": language,
        "isETranslated": translated,
        "title": "Test release",
        "eventDate": "2024-01-01",
        "docutype": {"code": "IP"},
    }


def detail(language="EN", translated=False):
    return {
        "refCd": "IP/24/1",
        "ky": 123,
        "originalLanguage": "EN",
        "docuLanguageResource": {
            "language": language,
            "etranslated": translated,
            "title": "Test release",
            "htmlContent": "<p>Full release body.</p>",
            "dolaQuoteResources": [
                {"extract": "Official quote", "author": {"authorName": "Commissioner"}}
            ],
        },
    }


def candidate():
    return {"reference_number": "IP/24/1", "language": "en"}


def test_search_language_is_not_document_language():
    def fetch(url):
        return json.dumps(
            {
                "totalNumber": 4,
                "pageNumber": 0,
                "pageSize": 100,
                "docuLanguageListResources": [
                    row(),
                    row("IP/24/2", "FR"),
                    row("IP/24/3", "EN", True),
                    row("IP/24/4", "EN", None),
                ],
            }
        ).encode(), {}

    result = cp.enumerate_collection("releases_en", fetch)
    assert len(result["records"]) == 1
    record = result["records"][0]
    assert parse_qs(urlsplit(record["url"]).query) == {
        "reference": ["IP/24/1"],
        "language": ["en"],
        "etrans": ["false"],
    }
    assert result["enumeration"]["skipped_language_fallback"] == 1
    assert result["enumeration"]["skipped_machine_translation_or_unknown"] == 2
    assert result["enumeration"]["complete"] is False


def test_request_and_response_page_index_differ(monkeypatch):
    monkeypatch.setattr(cp, "MAX_PAGES", 2)
    calls = []

    def fetch(url):
        page = int(parse_qs(urlsplit(url).query)["pagenumber"][0])
        calls.append(page)
        return json.dumps(
            {
                "totalNumber": 300,
                "pageNumber": page - 1,
                "pageSize": 100,
                "docuLanguageListResources": [row(f"IP/24/{page}")],
            }
        ).encode(), {}

    result = cp.enumerate_collection("releases_en", fetch, limit=1000)
    assert calls == [1, 2]
    assert len(result["records"]) == 2
    assert result["enumeration"]["selection_truncated"]


def test_repeated_pages_fail_instead_of_false_completion(monkeypatch):
    monkeypatch.setattr(cp, "MAX_PAGES", 2)

    def fetch(url):
        page = int(parse_qs(urlsplit(url).query)["pagenumber"][0])
        return json.dumps(
            {
                "totalNumber": 300,
                "pageNumber": page - 1,
                "pageSize": 100,
                "docuLanguageListResources": [row()],
            }
        ).encode(), {}

    with pytest.raises(ValueError, match="repeated"):
        cp.enumerate_collection("releases_en", fetch)


def test_extract_only_full_body_and_quotes():
    payload = detail()
    payload["leadText"] = "Summary must not enter text"
    payload["contactsResource"] = [{"mail": "contact@example.org"}]
    body, metadata = cp.extract_document(candidate(), json.dumps(payload).encode())
    assert b"Full release body" in body and b"Official quote" in body
    assert b"Summary must not" not in body and b"contact@example.org" not in body
    assert metadata["reference_number"] == "IP/24/1"
    assert metadata["is_machine_translated"] is False


@pytest.mark.parametrize("language,translated", [("FR", False), ("EN", True), ("EN", None)])
def test_reject_fallback_or_machine_translation(language, translated):
    with pytest.raises(ValueError):
        cp.extract_document(candidate(), json.dumps(detail(language, translated)).encode())


def test_reject_wrong_reference_and_missing_body():
    payload = detail()
    payload["refCd"] = "IP/24/2"
    with pytest.raises(ValueError, match="reference"):
        cp.extract_document(candidate(), json.dumps(payload).encode())
    payload = detail()
    payload["docuLanguageResource"].pop("htmlContent")
    payload["docuLanguageResource"]["leadText"] = "This is only a summary"
    with pytest.raises(ValueError, match="htmlContent"):
        cp.extract_document(candidate(), json.dumps(payload).encode())


def test_zero_limit_does_not_fetch():
    assert (
        cp.enumerate_collection("releases_en", lambda _: pytest.fail("unexpected fetch"), 0)[
            "records"
        ]
        == []
    )
