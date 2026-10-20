"""Bounded Python corpus preparation; private provenance stays on the data volume."""

import ast
import hashlib
import json
import re
import sqlite3
import subprocess
import time
import zlib
from collections import Counter
from pathlib import Path

from .overnight_corpus import SCHEMA
from .subword import FIRST_CONTENT_TOKEN_ID, TOKENIZER_SHA256, load_frozen

TARGETS = {
    "python_implementation": 3_250_000_000,
    "python_tests_examples": 750_000_000,
    "python_documentation": 500_000_000,
    "cs_math": 250_000_000,
    "general_education": 150_000_000,
    "books": 100_000_000,
}
EXCLUDED = {
    "vendor",
    "vendored",
    "third_party",
    "third-party",
    "node_modules",
    "build",
    "dist",
    "generated",
    "_generated",
    ".git",
    "site-packages",
    ".venv",
    "migrations",
    "__pycache__",
    "fixtures",
    "testdata",
}
GENERATED = re.compile(
    r"auto.?generated|automatically generated|generated (by|from)|do not edit", re.IGNORECASE
)


def canonical_repo(value):
    value = str(value).lower().removeprefix("code_python:")
    value = value.removeprefix("https://github.com/").removesuffix(".git").strip("/")
    return value


def split_for(parent):
    return (
        "validation"
        if int(hashlib.sha256(("python-5b-v1:" + parent).encode()).hexdigest()[:8], 16) % 20 == 0
        else "train"
    )


def category(path, text, python_repo=False):
    """Exclusive path/content classification without rewriting any source text."""
    if not isinstance(path, str) or not isinstance(text, str):
        return None, "malformed_record"
    p = Path(path.lower())
    if EXCLUDED.intersection(p.parts) or any(x.endswith(".egg-info") for x in p.parts):
        return None, "excluded_path"
    if not isinstance(text, str) or not 64 <= len(text) <= 2_000_000 or "\x00" in text:
        return None, "malformed_or_size"
    if GENERATED.search(text[:2000]):
        return None, "generated"
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    if not lines or max(map(len, lines)) > 2000:
        return None, "minified_or_long_line"
    if len(lines) > 40 and len(set(lines)) / len(lines) < 0.2:
        return None, "boilerplate"
    if p.suffix == ".py":
        try:
            ast.parse(text)
        except (SyntaxError, ValueError, RecursionError):
            return None, "invalid_python3"
        test = (
            p.name.startswith("test_")
            or p.name.endswith("_test.py")
            or any(
                x
                in {
                    "test",
                    "tests",
                    "example",
                    "examples",
                    "demo",
                    "demos",
                    "tutorial",
                    "tutorials",
                }
                for x in p.parts
            )
        )
        return ("python_tests_examples" if test else "python_implementation"), None
    if p.suffix in {".md", ".rst", ".txt"} and (python_repo or "python" in text[:10000].lower()):
        if p.name.startswith(("license", "copying", "authors", "changelog", "changes")):
            return None, "notice_or_changelog"
        if any(
            x in {"doc", "docs", "documentation", "tutorial", "tutorials"} for x in p.parts
        ) or p.name.startswith("readme"):
            return "python_documentation", None
    return None, "not_python_material"


def git(folder, *args):
    return subprocess.check_output(
        ["git", "-c", "safe.directory=*", "--git-dir", str(folder), *args], timeout=90
    )


def tree_files(folder, files):
    """Read complete pinned blobs in one Git process, including export-ignored tests."""
    process = subprocess.Popen(
        ["git", "-c", "safe.directory=*", "--git-dir", str(folder), "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    try:
        for path, blob in files:
            if Path(path).suffix.lower() not in {".py", ".md", ".rst", ".txt"}:
                continue
            if EXCLUDED.intersection(Path(path.lower()).parts):
                continue
            process.stdin.write((blob + "\n").encode())
            process.stdin.flush()
            header = process.stdout.readline().decode().split()
            if len(header) != 3 or header[0] != blob or header[1] != "blob":
                raise ValueError("Pinned Git blob missing")
            size = int(header[2])
            if not 64 <= size <= 2_000_000:
                remaining = size + 1
                while remaining:
                    chunk = process.stdout.read(min(remaining, 65536))
                    if not chunk:
                        raise ValueError("Truncated Git blob")
                    remaining -= len(chunk)
                continue
            raw = process.stdout.read(size)
            if len(raw) != size or process.stdout.read(1) != b"\n":
                raise ValueError("Truncated Git blob")
            try:
                yield path, raw.decode("utf-8")
            except UnicodeDecodeError:
                yield path, None
        process.stdin.close()
        if process.wait(timeout=90):
            raise RuntimeError("Pinned Git blob reader failed")
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdin.close()
        process.stdout.close()


def license_kind(text):
    t = text.lower()
    if "permission is hereby granted" in t and "without restriction" in t:
        return "MIT"
    if "apache license" in t and "version 2.0" in t:
        return "Apache-2.0"
    if "redistribution and use in source and binary forms" in t and "neither the name" in t:
        return "BSD-3-Clause"
    if "redistribution and use in source and binary forms" in t:
        return "BSD-2-Clause"
    if "python software foundation license" in t:
        return "PSF-2.0"
    if "permission to use, copy, modify, and/or distribute" in t:
        return "ISC"
    return None


class Corpus:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.folder / "raw.sqlite")
        self.db.execute("PRAGMA cache_size=-1048576")
        self.db.executescript(
            SCHEMA
            + """
          CREATE TABLE IF NOT EXISTS repositories (parent TEXT PRIMARY KEY, split TEXT, revision TEXT);
          CREATE TABLE IF NOT EXISTS completed (name TEXT PRIMARY KEY);
          CREATE TABLE IF NOT EXISTS rejections (reason TEXT PRIMARY KEY, count INTEGER);
          CREATE TABLE IF NOT EXISTS excluded_hashes (hash TEXT PRIMARY KEY);
          CREATE TABLE IF NOT EXISTS excluded_repositories (parent TEXT PRIMARY KEY);
          CREATE TABLE IF NOT EXISTS supplement_cursors (name TEXT PRIMARY KEY, hash TEXT);
          CREATE INDEX IF NOT EXISTS capacity_tokens ON documents(source,split,conflict,loss_tokens);
          CREATE INDEX IF NOT EXISTS repository_tokens ON documents(parent,source,split,conflict,loss_tokens);
        """
        )
        self.tokenizer = load_frozen()
        self.stats = Counter()
        self.repo_counts = Counter(
            {
                (p, s): n
                for p, s, n in self.db.execute(
                    "SELECT parent,source,SUM(loss_tokens) FROM documents WHERE split='train' AND conflict=0 GROUP BY parent,source"
                )
            }
        )

    def reject(self, reason):
        self.stats[reason] += 1

    def add(self, text, source, parent, provenance, split=None, verified_count=None):
        parent = canonical_repo(parent)
        row = self.db.execute("SELECT split FROM repositories WHERE parent=?", (parent,)).fetchone()
        split = row[0] if row else (split or split_for(parent))
        key = hashlib.sha256(text.encode()).hexdigest()
        if (
            split == "train"
            and self.db.execute("SELECT 1 FROM excluded_hashes WHERE hash=?", (key,)).fetchone()
        ):
            self.reject("heldout_hash_exclusion")
            return
        if (
            split == "train"
            and self.db.execute(
                "SELECT 1 FROM excluded_repositories WHERE parent=?", (parent,)
            ).fetchone()
        ):
            self.reject("heldout_repository_exclusion")
            return
        prior = self.db.execute("SELECT split FROM documents WHERE hash=?", (key,)).fetchone()
        if prior:
            if prior[0] != split:
                self.db.execute("UPDATE documents SET conflict=1 WHERE hash=?", (key,))
            self.reject("exact_duplicate")
            return
        if verified_count is None:
            ids = self.tokenizer.encode(text, add_special_tokens=False).ids
            if not ids or any(t < FIRST_CONTENT_TOKEN_ID for t in ids):
                self.reject("reserved_or_empty_encoding")
                return
            count = len(ids) + 1  # BOS predicts first token; EOS is one target.
        else:
            if type(verified_count) is not int or verified_count < 2:
                raise ValueError("Invalid inherited exact token count")
            count = verified_count
        if (
            split == "train"
            and source.startswith("python_")
            and self.repo_counts[parent, source] + count > TARGETS[source] // 100
        ):
            self.reject("repository_one_percent_cap")
            return
        self.db.execute(
            "INSERT INTO documents VALUES(?,?,?,?,?,?,?,0)",
            (
                key,
                source,
                split,
                parent,
                json.dumps(provenance, sort_keys=True),
                zlib.compress(text.encode()),
                count,
            ),
        )
        if split == "train":
            self.repo_counts[parent, source] += count
        return count

    def report(self, stage):
        with self.db:
            for k, n in self.stats.items():
                self.db.execute(
                    "INSERT INTO rejections VALUES(?,?) ON CONFLICT(reason) DO UPDATE SET count=count+excluded.count",
                    (k, n),
                )
            self.stats.clear()
        categories = {}
        for name, target in TARGETS.items():
            counts = dict(
                self.db.execute(
                    "SELECT split,SUM(loss_tokens) FROM documents WHERE source=? AND conflict=0 GROUP BY split",
                    (name,),
                )
            )
            unique = counts.get("train", 0)
            categories[name] = {
                "target": target,
                "unique_train_targets": unique,
                "validation_targets": counts.get("validation", 0),
                "required_passes": target / unique if unique else None,
                "two_pass_shortfall": max(0, target - 2 * unique),
                "one_pass_shortfall": max(0, target - unique),
            }
        result = {
            "stage": stage,
            "updated_unix": time.time(),
            "tokenizer_sha256": TOKENIZER_SHA256,
            "categories": categories,
            "capacity_sufficient": all(
                v["two_pass_shortfall"] == 0 and v["validation_targets"] > 0
                for v in categories.values()
            ),
            "rejections": dict(self.db.execute("SELECT * FROM rejections")),
            "repository_count": self.db.execute("SELECT COUNT(*) FROM repositories").fetchone()[0],
        }
        (self.folder / "capacity.json").write_text(json.dumps(result, indent=2) + "\n")
        table = "| Category | Target | Unique train targets | Validation targets | Required passes | Two-pass shortfall |\n|---|---:|---:|---:|---:|---:|\n"
        for n, v in categories.items():
            passes = (
                f"{v['required_passes']:.3f}" if v["required_passes"] is not None else "unavailable"
            )
            table += f"| {n} | {v['target']:,} | {v['unique_train_targets']:,} | {v['validation_targets']:,} | {passes} | {v['two_pass_shortfall']:,} |\n"
        (self.folder / "capacity.md").write_text(table)
        return result

    def repositories(self, roots, commit=lambda: None):
        repos = []
        for root in roots:
            for p in sorted(Path(root).iterdir()):
                try:
                    parent = p.name.removesuffix("-10k").replace("--", "/", 1).lower()
                    pinned = self.db.execute(
                        "SELECT revision FROM repositories WHERE parent=?", (parent,)
                    ).fetchone()
                    sha = pinned[0] if pinned else git(p, "rev-parse", "HEAD").decode().strip()
                    date = int(git(p, "show", "-s", "--format=%ct", sha))
                    repos.append((date, parent, sha, p))
                except (subprocess.SubprocessError, ValueError):
                    self.reject("invalid_git_snapshot")
        # Pin all repository splits before reading any file or making windows.
        with self.db:
            for date, parent, sha, p in repos:
                self.db.execute(
                    "INSERT OR IGNORE INTO repositories VALUES(?,?,?)",
                    (parent, split_for(parent), sha),
                )
        for date, parent, sha, p in sorted(repos, reverse=True):
            marker = "git:" + parent + ":" + sha
            if self.db.execute("SELECT 1 FROM completed WHERE name=?", (marker,)).fetchone():
                continue
            tree = git(p, "ls-tree", "-r", "-z", sha).split(b"\0")
            files = []
            for item in tree:
                if not item:
                    continue
                meta, path = item.split(b"\t", 1)
                mode, kind, blob = meta.decode().split()
                if mode not in {"100644", "100755"} or kind != "blob":
                    continue
                try:
                    path = path.decode("utf-8")
                except UnicodeDecodeError:
                    self.reject("non_utf8_path")
                    continue
                files.append((path, blob))
            notices = []
            for path, blob in files:
                if "/" not in path and path.lower().startswith(("license", "copying")):
                    raw = git(p, "cat-file", "blob", blob)
                    if len(raw) < 200000:
                        notices.append(raw.decode("utf-8", errors="replace"))
            license_text = "\n".join(notices)
            license_id = license_kind(license_text)
            if not license_id:
                self.reject("repository_license_unresolved")
                with self.db:
                    self.db.execute("INSERT OR IGNORE INTO completed VALUES(?)", (marker,))
                continue
            py = sum(path.endswith(".py") for path, blob in files)
            python_repo = (
                py >= 10
                and py
                / max(
                    1,
                    sum(
                        Path(path).suffix
                        in {".py", ".rs", ".go", ".js", ".ts", ".c", ".cpp", ".java"}
                        for path, blob in files
                    ),
                )
                >= 0.3
            )
            blobs = dict(files)
            for path, text in tree_files(p, files):
                blob = blobs[path]
                if text is None:
                    self.reject("non_utf8")
                    continue
                name, reason = category(path, text, python_repo)
                if name is None:
                    self.reject(reason)
                    continue
                spdx = re.search(r"SPDX-License-Identifier:\s*([^\n]+)", text[:3000])
                if spdx and not any(
                    x in spdx[1].lower()
                    for x in ["mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "psf-2.0"]
                ):
                    self.reject("incompatible_file_license")
                    continue
                self.add(
                    text,
                    name,
                    parent,
                    {
                        "origin": "collected_git",
                        "repository": parent,
                        "revision": sha,
                        "commit_unix": date,
                        "path": path,
                        "blob": blob,
                        "license": license_id,
                        "license_notice": license_text,
                    },
                )
            with self.db:
                self.db.execute("INSERT INTO completed VALUES(?)", (marker,))
            print(json.dumps({"repository_completed": parent, "commit_unix": date}), flush=True)
            self.report("preparing_repositories")
            commit()
        return self.report("repositories_complete")

    def supplement(self, database, commit=lambda: None, max_seconds=3600):
        """Inspect a bounded existing pinned source pool, never patch buffers."""
        start = time.monotonic()
        origin = str(database)
        old = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        old.execute("PRAGMA cache_size=-1048576")
        identity = json.loads(old.execute("SELECT value FROM settings WHERE id=1").fetchone()[0])
        if identity.get("tokenizer") != TOKENIZER_SHA256:
            raise ValueError("Supplement must preserve the existing tokenizer identity")
        names = {r[0] for r in old.execute("SELECT DISTINCT source FROM documents")}
        code_names = names & {"code", "code_python"}
        support = names & {
            "fineweb_edu",
            "finemath_4plus",
            "gutenberg",
            "pressbooks",
            "libretexts",
            "doab",
        }
        exclusion_marker = "exclusions:" + origin
        if not self.db.execute(
            "SELECT 1 FROM completed WHERE name=?", (exclusion_marker,)
        ).fetchone():
            with self.db:
                for key, parent, source in old.execute(
                    "SELECT hash,parent,source FROM documents WHERE split!='train'"
                ):
                    self.db.execute("INSERT OR IGNORE INTO excluded_hashes VALUES(?)", (key,))
                    if source in code_names:
                        self.db.execute(
                            "INSERT OR IGNORE INTO excluded_repositories VALUES(?)",
                            (canonical_repo(parent),),
                        )
                self.db.execute(
                    "UPDATE documents SET conflict=1 WHERE split='train' AND (hash IN (SELECT hash FROM excluded_hashes) OR parent IN (SELECT parent FROM excluded_repositories))"
                )
                self.db.execute("INSERT INTO completed VALUES(?)", (exclusion_marker,))
            self.report("prior_holdouts_excluded")
            commit()
        scanned = 0
        last_commit = time.monotonic()
        totals = Counter(
            dict(
                self.db.execute(
                    "SELECT source,SUM(loss_tokens) FROM documents WHERE split='train' AND conflict=0 GROUP BY source"
                )
            )
        )
        # Resolve documentation/support capacity before scanning the large code pool.
        for source in sorted(code_names | support, key=lambda name: (name in code_names, name)):
            marker = "supplement:" + origin + ":" + source
            if self.db.execute("SELECT 1 FROM completed WHERE name=?", (marker,)).fetchone():
                continue
            cursor = self.db.execute(
                "SELECT hash FROM supplement_cursors WHERE name=?", (marker,)
            ).fetchone()
            start_hash = cursor[0] if cursor else ""
            previous_hash = start_hash
            checked_counts = 0
            for key, parent, metadata, raw, previous_split, exact_count in old.execute(
                "SELECT hash,parent,provenance,text,split,loss_tokens FROM documents WHERE source=? AND conflict=0 AND hash>? ORDER BY hash",
                (source, start_hash),
            ):
                self.db.execute(
                    "INSERT INTO supplement_cursors VALUES(?,?) ON CONFLICT(name) DO UPDATE SET hash=excluded.hash",
                    (marker, previous_hash),
                )
                previous_hash = key
                scanned += 1
                # Commit cursor with completed rows only; the checkpoint below runs
                # before this row, so persist its predecessor after processing.
                now = time.monotonic()
                if scanned % 50_000 == 0 or (
                    scanned % 512 == 0 and (now - last_commit >= 60 or now - start >= max_seconds)
                ):
                    self.report("preparing_supplement")
                    commit()
                    last_commit = time.monotonic()
                    print(
                        json.dumps({"supplement_source": source, "rows_scanned": scanned}),
                        flush=True,
                    )
                    if time.monotonic() - start > max_seconds:
                        old.close()
                        return self.report("supplement_time_bound")
                if previous_split == "test":
                    self.reject("reserved_test_record")
                    continue
                try:
                    meta = json.loads(metadata)
                    if not isinstance(meta, dict):
                        raise ValueError("Expected record provenance object")
                    if source in code_names:
                        upstream = meta.get("upstream", {})
                        if not isinstance(upstream, dict):
                            raise ValueError("Expected upstream provenance object")
                        meta = {
                            **meta,
                            "path": meta.get("path", upstream.get("path")),
                            "repository": meta.get("repository", upstream.get("repo_name", parent)),
                            "license": meta.get("license", upstream.get("license")),
                        }
                        if (
                            meta.get("language", "python" if source == "code_python" else "")
                            != "python"
                        ):
                            continue
                        if (
                            meta.get("representation") == "git_patch"
                            or meta.get("task_family") == "patches"
                        ):
                            self.reject("patch_not_source")
                            continue
                        parent = canonical_repo(meta.get("repository", parent))
                        # Only the pinned recent complete snapshot for collected repos.
                        if self.db.execute(
                            "SELECT 1 FROM repositories WHERE parent=?", (parent,)
                        ).fetchone():
                            self.reject("older_collected_repository_version")
                            continue
                        if not meta.get("path") or not meta.get("revision"):
                            self.reject("missing_file_provenance")
                            continue
                    text = zlib.decompress(raw).decode("utf-8")
                    if hashlib.sha256(text.encode()).hexdigest() != key:
                        raise ValueError("Content hash mismatch")
                except (ValueError, TypeError, UnicodeDecodeError, zlib.error):
                    self.reject("malformed_source_record")
                    continue
                if source in code_names:
                    name, reason = category(meta["path"], text)
                    split = split_for(parent)
                else:
                    split = "train" if previous_split == "train" else "validation"
                    if source in {"fineweb_edu", "pressbooks", "libretexts", "doab"}:
                        lower = text.lower()
                        is_python_doc = (
                            "python" in lower
                            and any(
                                k in lower
                                for k in [
                                    "python tutorial",
                                    "python programming",
                                    "python function",
                                ]
                            )
                            and any(k in lower for k in ["def ", ">>> ", "import "])
                        )
                        name = "python_documentation" if is_python_doc else "general_education"
                        if source != "fineweb_edu" and not is_python_doc:
                            continue
                    elif source == "finemath_4plus":
                        name = "cs_math"
                    else:
                        name = "books"
                    reason = None
                    if "\x00" in text or len(text) < 64:
                        name, reason = None, "malformed_support"
                if name is None:
                    self.reject(reason)
                    continue
                if checked_counts < 16:
                    if (
                        len(self.tokenizer.encode(text, add_special_tokens=False).ids) + 1
                        != exact_count
                    ):
                        raise ValueError(
                            "Inherited exact target count disagrees with frozen tokenizer"
                        )
                    checked_counts += 1
                if split == "train" and totals[name] >= TARGETS[name] * 1.02:
                    continue
                added = self.add(
                    text,
                    name,
                    parent,
                    {"origin": "existing_pinned_pool", "upstream": meta},
                    split=split,
                    verified_count=exact_count,
                )
                if split == "train" and added:
                    totals[name] += added
                if (
                    source in {"finemath_4plus", "gutenberg"}
                    and totals[name] >= TARGETS[name] * 1.02
                ):
                    validation_count = self.db.execute(
                        "SELECT COALESCE(SUM(loss_tokens),0) FROM documents WHERE source=? AND split='validation' AND conflict=0",
                        (name,),
                    ).fetchone()[0]
                    if validation_count >= 1_000_000:
                        # The complete prior held-out hash ledger was loaded first.
                        # This bounds selection, not a claim of upstream exhaustion.
                        break
            with self.db:
                self.db.execute("INSERT INTO completed VALUES(?)", (marker,))
            self.report("supplement_source_complete")
            commit()
        old.close()
        return self.report("supplement_complete")
