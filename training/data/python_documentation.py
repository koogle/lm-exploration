"""Licensed Python explanatory Q&A supplement, with immutable source provenance."""

import re

REPOSITORY = "common-pile/stackexchange"
REVISION = "5ec3aa2e65b7c61b7c18f1ff18ccbc4d3279a23c"
TOPIC = re.compile(
    r"\b(?:python|pandas|numpy|scipy|pytest|pytorch|django|flask|pydantic|fastapi|"
    r"tensorflow|matplotlib|sqlalchemy|jupyter|scikit-learn)\b",
    re.IGNORECASE,
)
PYTHON_CODE = re.compile(r"(?m)^\s*(?:>>> |(?:async )?def \w+\(|from [\w.]+ import |import \w+)")
OTHER_CODE = re.compile(
    r"(?m)^\s*(?:#include\s*[<\"]|public static void |package [\w.]+;|using System;|const \w+ = require\()"
)
LICENSES = {f"https://creativecommons.org/licenses/by-sa/{v}/" for v in ["2.5", "3.0", "4.0"]}


def python2_specific(text):
    title = text.splitlines()[0]
    return bool(re.search(r"\bpython\s*2(?:\.\d+)?\b", title, re.IGNORECASE)) and not bool(
        re.search(r"\bpython\s*3(?:\.\d+)?\b", title, re.IGNORECASE)
    )


def select_document(record):
    """Keep whole explanatory threads; never treat a name as a quality guarantee."""
    if not isinstance(record, dict):
        return None, "malformed_record"
    text, meta = record.get("text"), record.get("metadata")
    if not isinstance(text, str) or not isinstance(meta, dict):
        return None, "malformed_record"
    if not 800 <= len(text) <= 60_000 or "\x00" in text:
        return None, "length_or_encoding"
    if python2_specific(text):
        return None, "python2_specific_documentation"
    if not TOPIC.search(text.splitlines()[0]) or not PYTHON_CODE.search(text):
        return None, "not_python_instruction"
    if OTHER_CODE.search(text):
        return None, "mixed_language_code"
    licenses = meta.get("all_licenses", [meta.get("license", "")])
    if (
        not isinstance(licenses, list)
        or not licenses
        or not all(
            isinstance(value, str) and any(value.endswith(url) for url in LICENSES)
            for value in licenses
        )
    ):
        return None, "unresolved_license"
    url = meta.get("url", "")
    if not isinstance(url, str) or not re.fullmatch(
        r"https://stackoverflow.com/questions/\d+", url
    ):
        return None, "missing_thread_identity"
    if not meta.get("authors"):
        return None, "missing_attribution"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Require explanatory prose, rather than only error logs or pasted snippets.
    prose = sum(
        len(line)
        for line in lines
        if len(line.split()) >= 12 and not line.startswith((">", "File ", "Traceback"))
    )
    if prose < 400 or max(map(len, lines)) > 2000 or len(set(lines)) / len(lines) < 0.5:
        return None, "insufficient_explanation_or_boilerplate"
    parent = "stackoverflow:" + url.rsplit("/", 1)[1]
    return (
        text,
        parent,
        {
            "origin": "python_explanatory_qa",
            "dataset": REPOSITORY,
            "revision": REVISION,
            "document_id": record.get("id"),
            "created": record.get("created"),
            "attribution": meta,
        },
    ), None
