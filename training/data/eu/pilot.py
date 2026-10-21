"""Run a bounded slice of the source registry and emit follow-up agent tasks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from eu_corpus import export


def task_for(source, pages):
    return {
        "source_id": source["id"],
        "institution": source["institution"],
        "seed_urls": source["seed_urls"],
        "allowed_hosts": source["allowed_hosts"],
        "suggested_request_budget": pages,
        "report_paths": [f"research/{source['id']}.md", f"research/{source['id']}.json"],
        "instructions": (
            "Read AGENTS.md and README.md. Investigate this collection with bounded public "
            "requests. Find native full text, inventories, pagination, language versions and "
            "update mechanisms. Follow subgroups, meetings and attachments where relevant. "
            "Record exact URLs, UTC times, HTTP outcomes and evidence. Distinguish inventory "
            "metadata and summaries from full documents. Keep raw files in artifacts/. "
            "Report a tested source configuration and unresolved gaps; do not invent totals. "
            "Do not deploy jobs, publish data, or commit; parent integration owns Git."
        ),
        "expansion_targets": source.get("expansion", []),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path("sources.json"))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-pages", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.offset < 0 or args.limit < 1 or args.max_pages < 1 or args.max_depth < 0:
        parser.error("offset/depth must be nonnegative; limit/max-pages must be positive")
    manifest = json.loads(args.sources.read_text())
    if manifest.get("schema_version") != 1:
        parser.error("Expected registry schema_version 1")
    sources = sorted(manifest["sources"], key=lambda source: source["priority"])
    selected = sources[args.offset : args.offset + args.limit]
    if not selected:
        parser.error("Selected slice is empty")
    tasks = [task_for(source, args.max_pages) for source in selected]
    if args.plan_only:
        for task in tasks:
            print(json.dumps(task, ensure_ascii=False))
        return
    output = args.output or Path("artifacts") / datetime.now(UTC).strftime(
        "pilot-%Y%m%dT%H%M%S.%fZ"
    )
    command = [sys.executable, str(Path(__file__).with_name("eu_corpus.py")), "crawl"]
    command += ["--sources", str(args.sources), "--output", str(output)]
    command += ["--max-pages", str(args.max_pages), "--max-depth", str(args.max_depth)]
    for source in selected:
        command += ["--source", source["id"]]
    subprocess.run(command, check=True)
    exported = export(output, output / "documents.parquet")
    (output / "export-summary.json").write_text(json.dumps(exported, indent=2) + "\n")
    with (output / "agent-tasks.jsonl").open("w") as handle:
        for task in tasks:
            task["crawl_artifacts"] = str(output)
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "export": exported}, indent=2))


if __name__ == "__main__":
    main()
