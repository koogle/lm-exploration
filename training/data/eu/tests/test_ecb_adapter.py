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


def fixture_fetch(rows, type_id="19"):
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
            {"value": type_id, "total_records": len(rows), "first_sort_id_per_chunk": [0]}
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


@pytest.mark.parametrize(
    ("collection_id", "type_id"),
    [
        ("annual_reports", "132"),
        ("convergence_reports", "22"),
        ("financial_stability_reviews", "137"),
        ("statistics_papers", "5"),
        ("research_bulletins", "61"),
    ],
)
def test_full_report_types_keep_native_text_and_distinct_translations(collection_id, type_id):
    fetch, _ = fixture_fetch(
        [
            [
                42,
                1700000000,
                int(type_id),
                [
                    "/pub/annual/html/ecb.ar2024.en.html",
                    "/pub/pdf/annrep/ecb.ar2024.en.pdf",
                    "/pub/pdf/annrep/ecb.ar2024.fr.pdf",
                ],
                {"Title": "Report 2024"},
            ]
        ],
        type_id=type_id,
    )
    result = enumerate_collection(collection_id, fetch)
    assert result["enumeration"]["complete"]
    assert result["enumeration"]["reported_total"] == 1
    assert [(r["format"], r["language"]) for r in result["records"]] == [
        ("html", "en"),
        ("pdf", "fr"),
    ]
    assert all(r["document_type"] == collection_id for r in result["records"])
    assert all(r["source_document_id"] == "42" for r in result["records"])
    assert COLLECTIONS[collection_id]["type_ids"] == [type_id]


def translation_fixture(pages, extra_files=()):
    rows = [
        [i, 1700000000, 132, [url, *extra_files], {"Title": "Report"}]
        for i, url in enumerate(pages)
    ]
    fetch, calls = fixture_fetch(rows, type_id="132")

    def translated_fetch(url):
        if url in pages:
            calls.append(url)
            return pages[url].encode(), {}
        return fetch(url)

    return translated_fetch, calls


def test_report_language_links_reject_navigation_other_reports_and_offhost():
    from adapters.ecb import report_translation_links

    seed = "https://www.ecb.europa.eu/pub/annual/html/ecb.ar2025~abc.en.html"
    links = [
        "ecb.ar2025~abc.de.html",
        "ecb.ar2025~abc.fr.html#section",
        "ecb.ar2025~abc.de.html",
        "ecb.ar2025~abc.en.html",
        "ecb.ar2024~abc.fr.html",
        "ecb.ar2025~def.fr.html",
        "/home.fr.html",
        "https://evil.test/pub/annual/html/ecb.ar2025~abc.fr.html",
        "ecb.ar2025~abc_content.fr.html",
        "ecb.ar2025~abc.fr.pdf",
    ]
    body = "".join(f'<a href="{url}">Language</a>' for url in links).encode()
    result = report_translation_links(body, seed)
    assert [language for _, language in result] == ["de", "fr"]
    assert report_translation_links(body, seed.replace(".en.", ".de.")) == []


def test_translation_enumeration_preserves_identity_and_excludes_inventory_urls():
    seed = "https://www.ecb.europa.eu/pub/annual/html/ecb.ar2025.en.html"
    fetch, _ = translation_fixture(
        {seed: '<a href="ecb.ar2025.fr.html">FR</a><a href="ecb.ar2025.de.html">DE</a>'},
        [seed.replace(".en.", ".fr.")],
    )
    result = enumerate_collection("annual_report_translations", fetch)
    assert result["enumeration"]["complete"]
    assert len(result["records"]) == 1
    record = result["records"][0]
    assert record["language"] == "de"
    assert record["source_document_id"] == "0"
    assert record["discovery_url"] == seed
    assert record["url"] == seed.replace(".en.", ".de.")


def test_translation_output_and_seed_bounds():
    from adapters.ecb import enumerate_report_translations

    seed = "https://www.ecb.europa.eu/pub/annual/html/ecb.ar2025.en.html"
    older = seed.replace("2025", "2024")
    pages = {
        seed: '<a href="ecb.ar2025.de.html">DE</a><a href="ecb.ar2025.fr.html">FR</a>',
        older: '<a href="ecb.ar2024.fr.html">FR</a>',
    }
    fetch, calls = translation_fixture(pages)
    result = enumerate_report_translations(fetch, limit=1)
    assert len(result["records"]) == 1
    assert not result["enumeration"]["complete"]
    assert older not in calls
    fetch, calls = translation_fixture(pages)
    result = enumerate_report_translations(fetch, seed_limit=1)
    assert len(result["records"]) == 2
    assert not result["enumeration"]["complete"]
    assert older not in calls


def test_translation_seed_fetch_failure_is_not_exhaustion():
    from adapters.ecb import enumerate_report_translations

    seed = "https://www.ecb.europa.eu/pub/annual/html/ecb.ar2025.en.html"
    fetch, _ = translation_fixture({seed: ""})

    def failing(url):
        if url == seed:
            raise RuntimeError("seed unavailable")
        return fetch(url)

    with pytest.raises(RuntimeError, match="seed unavailable"):
        enumerate_report_translations(failing)


def test_convergence_alternates_are_exact_and_hash_pinned(monkeypatch):
    import hashlib

    from adapters import ecb

    old = ecb.BASE + "/press/other-publications/convergence/pdf/cr2006en.pdf"
    record = {"collection_id": "convergence_reports", "url": old}
    assert ecb.alternate_document_urls(record) == [ecb.BASE + "/pub/pdf/conrep/cr2006en.pdf"]
    assert ecb.alternate_document_urls({**record, "url": old.replace("2006", "2008")}) == []
    with pytest.raises(ValueError, match="verified PDF identity"):
        ecb.validate_alternate(record, b"wrong document")
    body = b"%PDF-test identity fixture"
    monkeypatch.setitem(
        ecb.VERIFIED_CONVERGENCE_REPORTS,
        old,
        {
            "url": ecb.BASE + "/pub/pdf/conrep/cr2006en.pdf",
            "sha256": hashlib.sha256(body).hexdigest(),
        },
    )
    ecb.validate_alternate(record, body)


@pytest.mark.parametrize(
    "name",
    [
        "mb201212_focus08.en.pdf",
        "mb201312_focus07.en.pdf",
        "mb201404_focus03.en.pdf",
        "eb201505_focus04.en.pdf",
        "eb201607_box02.en.pdf",
        "ebbox201802_06.en.pdf",
        "ecb.ebbox201803_01.en.pdf",
    ],
)
def test_box_recovery_preserves_recognized_basename(name):
    import re

    from adapters.ecb import alternate_document_urls

    year = re.search(r"20\d{2}", name)[0]
    record = {
        "collection_id": "economic_bulletin_boxes",
        "url": f"https://www.ecb.europa.eu/press/economic-bulletin/focus/{year}/pdf/{name}",
    }
    assert alternate_document_urls(record) == [f"https://www.ecb.europa.eu/pub/pdf/other/{name}"]


@pytest.mark.parametrize(
    "path",
    [
        "2011/pdf/mb201112_focus13.en.pdf",
        "2012/pdf/eb201202_box02.en.pdf",
        "2013/pdf/ebbox201302_01.en.pdf",
        "2012/pdf/mb201312_focus07.en.pdf",
        "2015/pdf/mb201404_focus03.en.pdf",
        "2014/pdf/mb201413_focus03.en.pdf",
        "2015/pdf/eb201509_box02.en.pdf",
        "2018/pdf/ebbox201802_00.en.pdf",
        "2020/pdf/ebbox202002_01.en.pdf",
        "2014/pdf/mb201404.en.pdf",
    ],
)
def test_box_recovery_rejects_out_of_scope_names(path):
    from adapters.ecb import alternate_document_urls

    record = {
        "collection_id": "economic_bulletin_boxes",
        "url": f"https://www.ecb.europa.eu/press/economic-bulletin/focus/{path}",
    }
    assert alternate_document_urls(record) == []


@pytest.mark.parametrize("year", [2012, 2013, 2014])
def test_box_recovery_requires_title_and_rejects_full_issue(monkeypatch, year):
    from types import SimpleNamespace

    import adapters.ecb as adapter

    record = {
        "collection_id": "economic_bulletin_boxes",
        "title": "A verified historical box title",
        "url": f"https://www.ecb.europa.eu/press/economic-bulletin/focus/{year}/pdf/mb{year}04_focus03.en.pdf",
    }
    page = SimpleNamespace(extract_text=lambda: record["title"])
    monkeypatch.setattr(adapter, "PdfReader", lambda _: SimpleNamespace(pages=[page]))
    adapter.validate_alternate(record, b"%PDF-fixture")
    with pytest.raises(ValueError, match="title mismatch"):
        adapter.validate_alternate(
            {**record, "title": "A completely different official title"}, b"%PDF-fixture"
        )
    with pytest.raises(ValueError, match="not a PDF"):
        adapter.validate_alternate(record, b"HTML response")
    monkeypatch.setattr(adapter, "PdfReader", lambda _: SimpleNamespace(pages=[page] * 21))
    with pytest.raises(ValueError):
        adapter.validate_alternate(record, b"%PDF-fixture")
