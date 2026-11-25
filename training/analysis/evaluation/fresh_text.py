"""Recent arXiv abstracts as text no released model has trained on.

Run from training/:
    python -m analysis.evaluation.fresh_text --since 2026-11-16 --until 2026-11-24 \
        --count 1000 --output artifacts/evaluation/arxiv-2026-11-16.jsonl

Writes one `{"id", "published", "text"}` row per paper first submitted in the date range
(the arXiv API's `submittedDate`, newest first): the title, a blank line and the abstract,
with the API's line wrapping removed. Upload the file to the runs volume under
`benchmarks/texts/` for `model_hybrid.modal_benchmarks --suite perplexity --text`.
"""

import argparse
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

API = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
PAGE = 200
# The API asks for three seconds between requests.
PAUSE_SECONDS = 3


def rows(since, until, count):
    window = f"submittedDate:[{since.replace('-', '')}0000 TO {until.replace('-', '')}2359]"
    found = {}
    for start in range(0, count, PAGE):
        query = urllib.parse.urlencode(
            {
                "search_query": window,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "start": start,
                "max_results": min(PAGE, count - start),
            }
        )
        with urllib.request.urlopen(f"{API}?{query}", timeout=120) as response:
            entries = ET.fromstring(response.read()).findall(f"{ATOM}entry")
        for entry in entries:
            title, abstract = (
                " ".join(entry.findtext(f"{ATOM}{field}").split()) for field in ("title", "summary")
            )
            identifier = entry.findtext(f"{ATOM}id").rsplit("/", 1)[-1]
            found[identifier] = {
                "id": identifier,
                "published": entry.findtext(f"{ATOM}published"),
                "text": f"{title}\n\n{abstract}",
            }
        if len(entries) < PAGE:
            break
        time.sleep(PAUSE_SECONDS)
    return list(found.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", required=True)
    parser.add_argument("--until", required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    found = rows(args.since, args.until, args.count)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in found:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    size = sum(len(row["text"].encode()) for row in found)
    print(json.dumps({"output": str(output), "rows": len(found), "bytes": size}))


if __name__ == "__main__":
    main()
