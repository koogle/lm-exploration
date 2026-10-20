"""Derive local edit SFT from exact private collection buffers; never execute code."""

import argparse
import difflib
import gzip
import hashlib
import json
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

from training_data.commitpack_sft import fingerprint
from training_data.repair_sft import encode, prepare
from training_data.subword import digest, load_frozen

LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".java": "java",
    ".sh": "bash",
    ".bash": "bash",
    ".zig": "zig",
    ".lua": "lua",
    ".swift": "swift",
    ".m": "objective-c",
    ".mm": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".sql": "sql",
}
GENERATED = re.compile(
    r"(^|/)(vendor|node_modules|dist|third_party|third-party|generated|target)/|\.min\.|\.generated\.",
    re.IGNORECASE,
)
SECRET = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(?:gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}|hf_[A-Za-z0-9]{30,})\b"
)


def commit_split(repository, sha):
    key = f"{repository.lower()}@{sha.lower()}"
    return (
        "validation" if int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 20 == 0 else "train"
    )


def blob_matches(text, oid):
    if not isinstance(oid, str) or not re.fullmatch("[a-f0-9]{40}", oid):
        return False
    data = text.encode("utf-8")
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == oid


def excerpts(before, after):
    """Exact aligned old/new regions, including six unchanged context lines."""
    old, new = before.splitlines(keepends=True), after.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(a=old, b=new, autojunk=True)
    for group in matcher.get_grouped_opcodes(n=6):
        a, b = group[0][1], group[-1][2]
        c, d = group[0][3], group[-1][4]
        left, right = "".join(old[a:b]), "".join(new[c:d])
        changed = sum(max(j - i, l - k) for tag, i, j, k, l in group if tag != "equal")
        if min(len(left.strip()), len(right.strip())) < 15 or fingerprint(left) == fingerprint(
            right
        ):
            continue
        yield a, b, c, d, left, right, changed


def instruction(message):
    if not isinstance(message, str):
        return None
    text = message.strip()
    subject = text.splitlines()[0] if text else ""
    if not 12 <= len(subject) <= 500 or re.match(
        r"^(merge|revert|bump|release)\b", subject, re.IGNORECASE
    ):
        return None
    # Do not include author emails, trailers or attribution in training prompts.
    parts = re.split(r"\n\s*\n", text)
    body = parts[1].strip() if len(parts) > 1 else ""
    if len(body) > 1200 or re.search(r"@|signed-off-by:|co-authored-by:", body, re.IGNORECASE):
        body = ""
    return subject + ("\n" + body if body else "")


def read_sources(path, expected):
    with open(path) as handle:
        count, last = 0, time.monotonic()
        while True:
            line = handle.readline()
            if line:
                count += 1
                last = time.monotonic()
                yield json.loads(line)
            elif expected is None or count >= expected:
                return
            elif time.monotonic() - last > 1800:
                raise TimeoutError("Source downloader stopped producing shards")
            else:
                time.sleep(1)


def build(
    manifest,
    output,
    reserved_paths=(),
    context=2048,
    train_cap=180000,
    expected_shards=None,
    replay_path=None,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = load_frozen()
    db = sqlite3.connect(output / "candidates.sqlite")
    db.execute(
        "CREATE TABLE candidates (id TEXT PRIMARY KEY, split TEXT, lang TEXT, repo TEXT, before_hash TEXT, after_hash TEXT, body TEXT)"
    )
    db.execute("CREATE TABLE blocked (hash TEXT PRIMARY KEY)")
    reserved_repos = set()
    for path in reserved_paths:
        with open(path) as handle:
            for line in handle:
                row = json.loads(line)
                if row["split"] != "validation":
                    continue
                reserved_repos.update(r.lower() for r in row.get("repositories", []))
                for key in ("broken", "fixed"):
                    db.execute("INSERT OR IGNORE INTO blocked VALUES (?)", (fingerprint(row[key]),))
    notices = output / "license-evidence"
    notices.mkdir()
    rejected, candidates, languages = Counter(), Counter(), Counter()
    downloaded = []
    seen_commits = set()
    for shard_index, source in enumerate(read_sources(manifest, expected_shards)):
        downloaded.append(source)
        if "error" in source:
            rejected["source_download_failure"] += 1
            continue
        if digest(source["local_path"]) != source["sha256"]:
            raise ValueError("Downloaded shard hash mismatch")
        with gzip.open(source["local_path"], "rt") as handle:
            for line in handle:
                raw = json.loads(line)
                repo, sha = raw["repository"].lower(), raw["sha"].lower()
                if repo in reserved_repos:
                    rejected["previous_validation_repository"] += 1
                    continue
                if (repo, sha) in seen_commits:
                    continue
                seen_commits.add((repo, sha))
                task = instruction(raw.get("message"))
                if not task or SECRET.search(task):
                    rejected["instruction"] += 1
                    continue
                evidence = {
                    k: raw.get(k) for k in ("license_before", "license_after", "repository_license")
                }
                if not evidence["license_before"] or not evidence["license_after"]:
                    rejected["missing_license_evidence"] += 1
                    continue
                evidence_text = json.dumps(evidence, sort_keys=True)
                evidence_hash = hashlib.sha256(evidence_text.encode()).hexdigest()
                evidence_path = notices / (evidence_hash + ".json")
                if not evidence_path.exists():
                    evidence_path.write_text(evidence_text)
                license_name = raw.get("repository_license") or (
                    "MIT" if "mit-claude" in source["dataset"] else "see-retained-notices"
                )
                split = commit_split(repo, sha)
                for file in raw.get("files", []):
                    path = file.get("path", "")
                    lang = LANGUAGES.get(Path(path).suffix.lower())
                    before, after = file.get("before"), file.get("after")
                    if not lang or GENERATED.search(path):
                        rejected["path"] += 1
                        continue
                    if (
                        not isinstance(before, str)
                        or not isinstance(after, str)
                        or not before
                        or not after
                    ):
                        rejected["missing_buffer"] += 1
                        continue
                    if max(len(before), len(after)) > 200000:
                        rejected["large_file"] += 1
                        continue
                    if not blob_matches(before, file.get("before_blob")) or not blob_matches(
                        after, file.get("after_blob")
                    ):
                        rejected["blob_hash"] += 1
                        continue
                    if SECRET.search(before) or SECRET.search(after):
                        rejected["credential_pattern"] += 1
                        continue
                    for a, b, c, d, left, right, changed in excerpts(before, after):
                        key = f"{repo}@{sha}:{path}:{a}:{b}:{c}:{d}"
                        row = {
                            "id": "private-patch-" + hashlib.sha256(key.encode()).hexdigest(),
                            "group": f"github:{repo}@{sha}",
                            "split": split,
                            "language": lang,
                            "instruction": f"Apply the relevant part of this change to the code excerpt from {path}. Return the complete corrected excerpt, preserving its surrounding context.\nChange: {task}",
                            "broken": left,
                            "fixed": right,
                            "source": f"https://github.com/{repo}/commit/{sha}",
                            "license": license_name,
                            "license_evidence_sha256": evidence_hash,
                            "dataset": source["dataset"],
                            "dataset_revision": source["revision"],
                            "source_shard": source["file"],
                            "source_shard_sha256": source["sha256"],
                            "repository": repo,
                            "commit": sha,
                            "parent": raw.get("parent"),
                            "path": path,
                            "before_blob": file["before_blob"],
                            "after_blob": file["after_blob"],
                            "before_lines": [a + 1, b],
                            "after_lines": [c + 1, d],
                            "changed_lines": changed,
                            "scope": "excerpt",
                            "verification": "Git blob hashes and exact buffer slices verified; project tests not rerun",
                        }
                        try:
                            encoded = encode(row, tokenizer, context)
                        except ValueError:
                            rejected["context"] += 1
                            continue
                        row["input_tokens"] = len(encoded["x"])
                        row["answer_tokens"] = sum(v != -100 for v in encoded["y"])
                        old_hash, new_hash = fingerprint(left), fingerprint(right)
                        if split == "validation":
                            db.executemany(
                                "INSERT OR IGNORE INTO blocked VALUES (?)",
                                [(old_hash,), (new_hash,)],
                            )
                        db.execute(
                            "INSERT OR IGNORE INTO candidates VALUES (?,?,?,?,?,?,?)",
                            (row["id"], split, lang, repo, old_hash, new_hash, json.dumps(row)),
                        )
                        candidates[split] += 1
                        languages[lang] += 1
        db.commit()
        if (shard_index + 1) % 20 == 0:
            print(
                json.dumps(
                    {
                        "shards": shard_index + 1,
                        "of": expected_shards,
                        "candidates": dict(candidates),
                    }
                ),
                flush=True,
            )
    db.execute("CREATE INDEX split_id ON candidates(split, id)")
    counts, repo_counts, tasks, selected_languages = Counter(), Counter(), Counter(), set()
    seen_pairs = set()
    answer_tokens, input_tokens = 0, 0
    destination = output / "examples.jsonl"
    with destination.open("w") as handle:
        for split in ("validation", "train"):
            query = "SELECT id,lang,repo,before_hash,after_hash,body FROM candidates WHERE split=? ORDER BY id"
            for _, lang, repo, old_hash, new_hash, body in db.execute(query, (split,)):
                if split == "validation" and counts[(split, lang)] >= 128:
                    continue
                if split == "train":
                    if (
                        lang not in selected_languages
                        or counts[(split, lang)] >= 30000
                        or repo_counts[(split, repo)] >= 12000
                        or tasks[split] >= train_cap
                    ):
                        continue
                    if db.execute(
                        "SELECT 1 FROM blocked WHERE hash IN (?,?) LIMIT 1", (old_hash, new_hash)
                    ).fetchone():
                        rejected["heldout_overlap"] += 1
                        continue
                if (old_hash, new_hash) in seen_pairs:
                    rejected["duplicate_pair"] += 1
                    continue
                seen_pairs.add((old_hash, new_hash))
                row = json.loads(body)
                handle.write(body + "\n")
                counts[(split, lang)] += 1
                repo_counts[(split, repo)] += 1
                tasks[split] += 1
                if split == "validation":
                    selected_languages.add(lang)
                else:
                    answer_tokens += row["answer_tokens"]
                    input_tokens += row["input_tokens"]
    private_totals = dict(tasks)
    replay_counts = Counter()
    if replay_path:
        # Preserve language coverage with a bounded slice of the previous corpus.
        # Exclude replay repositories used by private validation as an extra guard.
        private_validation_repos = {r for (sp, r) in repo_counts if sp == "validation"}
        with open(replay_path) as source, destination.open("a") as handle:
            for line in source:
                row = json.loads(line)
                split, lang = row["split"], row["language"]
                if split == "train":
                    if replay_counts[(split, lang)] >= 2000:
                        continue
                    if private_validation_repos & set(row.get("repositories", [])):
                        continue
                    hashes = [fingerprint(row[k]) for k in ("broken", "fixed")]
                    if db.execute(
                        "SELECT 1 FROM blocked WHERE hash IN (?,?) LIMIT 1", hashes
                    ).fetchone():
                        continue
                handle.write(json.dumps(row) + "\n")
                replay_counts[(split, lang)] += 1
                counts[(split, lang)] += 1
                tasks[split] += 1
                if split == "train":
                    answer_tokens += row["answer_tokens"]
                    input_tokens += row["input_tokens"]
    db.close()
    # Reuse the actual trainer's strict split, duplicate and tokenizer validation.
    splits, data_hash = prepare(destination, tokenizer, context)
    del splits
    report = {
        "dataset_sha256": data_hash,
        "builder_sha256": digest(__file__),
        "context": context,
        "counts": {
            s: {l: n for (sp, l), n in sorted(counts.items()) if sp == s}
            for s in ("train", "validation")
        },
        "total": dict(tasks),
        "private_patch_total": private_totals,
        "replay_counts": {
            s: {l: n for (sp, l), n in sorted(replay_counts.items()) if sp == s}
            for s in ("train", "validation")
        },
        "replay_sha256": digest(replay_path) if replay_path else None,
        "input_tokens": input_tokens,
        "answer_tokens": answer_tokens,
        "repositories": {
            s: len({r for (sp, r) in repo_counts if sp == s}) for s in ("train", "validation")
        },
        "candidate_counts": dict(candidates),
        "rejections": dict(rejected),
        "sources": [{k: v for k, v in s.items() if k != "local_path"} for s in downloaded],
        "split_policy": "Commit-disjoint within repositories; all eligible validation excerpt hashes blocked from training. Prior CommitPack validation repositories excluded.",
        "limits": {
            "max_train": train_cap,
            "train_per_language": 30000,
            "train_per_repository": 12000,
            "validation_per_language": 128,
        },
        "limitations": [
            "Local excerpt edits, not complete repository repair.",
            "Git blobs and slices verified; project tests not rerun.",
            "Commit-disjoint, not repository-disjoint. No semantic clone audit or pretraining decontamination.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "sources"}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reserved", action="append", default=[])
    parser.add_argument("--expected-shards", type=int)
    parser.add_argument("--replay")
    args = parser.parse_args()
    build(
        args.manifest,
        args.output,
        args.reserved,
        expected_shards=args.expected_shards,
        replay_path=args.replay,
    )
