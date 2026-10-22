"""Permissive-license multilingual source files from a pinned public code dataset.

`select` decides whether one dataset row becomes a training document. The partitioned
collector in `parallel_code.py` applies it to every shard; `readiness_audit.py` audits
the merged database before it is used for training.
"""

import hashlib

from .simple_llm import SPLIT_VERSION

CODE_REPO = "codeparrot/github-code-clean"
CODE_REVISION = "c48d40f9e70f0196f8236901ee35807f7d6c44c0"
LICENSES = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc"}
LANGUAGES = {
    "python",
    "javascript",
    "typescript",
    "java",
    "c",
    "c++",
    "c#",
    "go",
    "rust",
    "php",
    "ruby",
    "swift",
    "kotlin",
    "scala",
    "sql",
    "shell",
    "bash",
    "powershell",
    "lua",
    "r",
    "julia",
    "haskell",
    "clojure",
    "objective-c",
    "perl",
    "dart",
    "ocaml",
    "elixir",
    "f#",
}
VENDORED = ("node_modules", "vendor", "third_party", "dist", "build", "bower_components")
# Repositories share split buckets with the scholarly corpus's code source, so a
# repository held out there is held out here too.
PARENT_PREFIX = "code_python:"
SCHEMA = """CREATE TABLE IF NOT EXISTS documents (
 hash TEXT PRIMARY KEY,source TEXT,split TEXT,parent TEXT,provenance TEXT,
 text BLOB,loss_tokens INTEGER,conflict INTEGER DEFAULT 0);
 CREATE INDEX IF NOT EXISTS document_split ON documents(source,split,conflict,hash);"""


def task_family(path, language):
    value = "/" + path.lower()
    if any(
        x in value for x in ["/test/", "/tests/", "/spec/", "test_", "_test.", ".spec.", ".test."]
    ):
        return "tests"
    if any(
        x in value
        for x in ["/example", "/demo", "/tutorial", "/benchmark", "/algorithm", "/solution"]
    ):
        return "examples_algorithms"
    if language in {"shell", "bash", "powershell"} or any(
        x in value for x in ["/script", "/tools/", "/cli/", "/bin/"]
    ):
        return "tooling"
    return "implementation"


def split_for(parent):
    bucket = int(hashlib.sha256(f"{SPLIT_VERSION}:{parent}".encode()).hexdigest(), 16) % 10
    return "validation" if bucket == 0 else "test" if bucket == 1 else "train"


def select(row):
    """The kept fields of one dataset row, or None for a filtered row."""
    language = str(row.get("language", "")).lower()
    content, repository, path = row.get("code"), row.get("repo_name"), row.get("path")
    if (
        language not in LANGUAGES
        or str(row.get("license", "")).lower() not in LICENSES
        or not repository
        or not path
        or not isinstance(content, str)
        or not 64 <= len(content) <= 4_000_000
    ):
        return None
    if any(f"/{part}/" in "/" + path.lower() for part in VENDORED):
        return None
    header = content[:1000].lower()
    if "auto-generated" in header or "automatically generated" in header:
        return None
    return {
        "content": content,
        "language": language,
        "task_family": task_family(path, language),
        "parent": PARENT_PREFIX + repository.lower(),
        "repository": repository,
        "path": path,
        "license": row["license"],
    }
