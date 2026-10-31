"""PR -> task: turn collected commits into repository tasks with executable checks.

A task is a commit whose diff changes both source files and test files in one
language. The model sees the repository at the parent revision plus an issue
written from the commit (by a teacher model, without the code); the commit's own
test changes are the hidden check (they must fail before the source change and
pass after it, verified by running them) and the commit is the reference patch.

This module holds the pure, execution-free parts: file classification, candidate
filtering and the per-repository sample. `modal_pr_tasks.py` runs the stages.
"""

import difflib
import json
import random
import re

LANGUAGES = {
    "python": (".py",),
    "javascript": (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"),
    "go": (".go",),
}
TEST_PATH = re.compile(
    r"(^|/)(tests?|__tests__|specs?)/"
    r"|(^|/)test_[^/]*\.py$|_test\.(py|go)$|(^|/)conftest\.py$"
    r"|\.(test|spec)\.[cm]?[jt]sx?$"
)
LIMITS = {
    "source_files": (1, 3),
    "test_files": (1, 3),
    "source_lines": (3, 200),
    "test_lines": (3, 400),
    "subject_chars": (10, 300),
}


def language(path):
    for name, extensions in LANGUAGES.items():
        if path.endswith(extensions):
            return name
    return None


def is_test(path):
    return bool(TEST_PATH.search(path))


def changed_lines(before, after):
    """Added plus removed lines between two buffers."""
    return sum(
        1
        for line in difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=0)
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    )


def describe(row):
    """The commit's task-relevant shape, or the reason it is not a candidate."""
    if len(row["parents"]) != 1:
        return {"reject": "merge"}
    source, tests, languages = [], [], set()
    for file in row["files"]:
        kind = language(file["path"])
        if kind is None:
            continue
        languages.add(kind)
        (tests if is_test(file["path"]) else source).append(file)
    if len(languages) != 1:
        return {"reject": "no_code" if not languages else "mixed_languages"}
    shape = {
        "language": languages.pop(),
        "source_files": len(source),
        "test_files": len(tests),
        "source_lines": sum(changed_lines(f["before"], f["after"]) for f in source),
        "test_lines": sum(changed_lines(f["before"], f["after"]) for f in tests),
        "subject_chars": len(row["message"].split("\n", 1)[0].strip()),
    }
    for key, (low, high) in LIMITS.items():
        if not low <= shape[key] <= high:
            return {"reject": key, **shape}
    shape["source_paths"] = [f["path"] for f in source]
    shape["test_paths"] = [f["path"] for f in tests]
    return shape


def sample(candidates, count, per_repository, seed=0):
    """Up to `per_repository` candidates per repository, then `count` overall, seeded."""
    rng = random.Random(seed)
    by_repository = {}
    for candidate in sorted(candidates, key=lambda c: (c["repository"], c["sha"])):
        by_repository.setdefault(candidate["repository"], []).append(candidate)
    pool = []
    for items in by_repository.values():
        rng.shuffle(items)
        pool.extend(items[:per_repository])
    rng.shuffle(pool)
    return sorted(pool[:count], key=lambda c: (c["repository"], c["sha"]))


# Issue writing (teacher model) and the leak check.
ISSUE_SYSTEM = (
    "You write GitHub issues. You are shown a commit message and the checks that the "
    "finished work must satisfy (not the implementation). Write the issue a user or "
    "maintainer would have filed beforehand, so that a developer reading only the issue and "
    "the repository could do the work.\n"
    "Rules:\n"
    "- Describe the problem or the requested behavior: what happens now and what should "
    "happen. Include a short reproduction or usage example when it helps.\n"
    "- Name every public function, class, method, option, CLI flag, field or error message "
    "the checks use, with their exact names and signatures.\n"
    "- Describe only observable behavior, never how to implement it.\n"
    "- Never mention tests, checks, test files, assertions, the diff or the commit; write as "
    "if none of them exist yet.\n"
    "- Write as a person reporting the issue, in plain Markdown, at most 250 words.\n"
    'Answer with JSON only: {"title": "...", "body": "...", "self_contained": true|false} '
    "where self_contained says whether the issue alone specifies the work well enough."
)
MAX_DIFF_CHARS = 24000


def test_diff(task):
    """The commit's diff restricted to its test files."""
    sections = re.split(r"(?m)^(?=diff --git )", task["patch"])
    keep = [
        s
        for s in sections
        if any(f" b/{p}\n" in s.split("\n", 1)[0] + "\n" for p in task["test_paths"])
    ]
    return "".join(keep)


TRAILER = re.compile(r"(?im)^(co-authored-by|signed-off-by|claude-session|generated with|🤖).*$")


def issue_messages(task):
    """The teacher sees the commit message and the test changes, never the source changes."""
    diff = test_diff(task)
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + "\n[truncated]"
    user = (
        f"Repository: {task['repository']}\n\n"
        f"Commit message:\n{TRAILER.sub('', task['message']).strip()}\n\n"
        f"Checks the finished work must pass:\n{diff}"
    )
    return [{"role": "system", "content": ISSUE_SYSTEM}, {"role": "user", "content": user}]


def parse_issue(text):
    """The teacher's JSON answer, or None."""
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start : end + 1])
    except ValueError:
        return None
    if (
        not isinstance(data, dict)
        or not str(data.get("title", "")).strip()
        or not str(data.get("body", "")).strip()
    ):
        return None
    return {
        "title": str(data["title"]).strip(),
        "body": str(data["body"]).strip(),
        "self_contained": data.get("self_contained") is True,
    }


def added_source_lines(patch, source_paths, min_chars=25):
    """Distinctive lines the commit adds to its source files (whitespace-normalized)."""
    lines, current = set(), None
    for line in patch.splitlines():
        if line.startswith("+++ "):
            path = line[4:].removeprefix("b/")
            current = path if path in source_paths else None
        elif current and line.startswith("+") and not line.startswith("+++"):
            text = " ".join(line[1:].split())
            if len(text) >= min_chars:
                lines.add(text)
    return lines


DEFINITION = re.compile(
    r"\b(?:def|class|function|func|const|let|var|type|interface)\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w{2,})"
)
META = re.compile(
    r"\b(tests?|test files?|unit tests?|test suite|assertions?|the diff|this commit|the commit)\b",
    re.IGNORECASE,
)


def _hunks(patch, paths, added_only):
    """Lines of the patch belonging to `paths` (added lines only, or all hunk lines)."""
    out, current = [], None
    for line in patch.splitlines():
        if line.startswith("+++ "):
            path = line[4:].removeprefix("b/")
            current = path if path in paths else None
        elif current and not line.startswith(("---", "diff ", "index ", "@@")):
            if not added_only or line.startswith("+"):
                out.append(line[1:])
    return out


def hidden_names(patch, source_paths, test_paths):
    """Names the commit defines in its source files that its tests never mention."""
    defined = set()
    for line in _hunks(patch, set(source_paths), added_only=True):
        defined.update(DEFINITION.findall(line))
    tests = "\n".join(_hunks(patch, set(test_paths), added_only=False))
    distinctive = {
        n for n in defined if "_" in n.strip("_") or re.search(r"[a-z][A-Z]", n) or len(n) >= 10
    }
    return sorted(n for n in distinctive if not re.search(rf"\b{re.escape(n)}\b", tests))


RANK = ("ok", "not_self_contained", "leaked", "unparsed")


def judge(text, task):
    """(status, issue) for one teacher answer; status is one of RANK."""
    issue = parse_issue(text)
    if issue is None:
        return "unparsed", None
    issue["leaks"] = leaks(issue, task["patch"], task["source_paths"], task["test_paths"])
    if any(issue["leaks"].values()):
        return "leaked", issue
    return ("ok" if issue["self_contained"] else "not_self_contained"), issue


def best_issue(texts, task):
    """The first answer with the best status among several drafts."""
    judged = [(*judge(t, task), t) for t in texts]
    return min(judged, key=lambda j: RANK.index(j[0]))


def leaks(issue, patch, source_paths, test_paths=()):
    """What the issue gives away: quoted added source lines, names of new internal code
    the tests never use, and mentions of tests or the commit."""
    text = f"{issue['title']}\n{issue['body']}"
    body = " ".join(text.split())
    lines = sorted(line for line in added_source_lines(patch, source_paths) if line in body)
    names = [
        n
        for n in hidden_names(patch, source_paths, test_paths)
        if re.search(rf"\b{re.escape(n)}\b", text)
    ]
    return {"lines": lines, "names": names, "meta": sorted({m.lower() for m in META.findall(text)})}
