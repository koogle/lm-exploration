"""Build repository-disjoint repair SFT from an immutable CommitPackFT revision."""

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from training_data import repair_sft
from training_data.licenses import PERMISSIVE_LICENSES
from training_data.subword import digest, load_frozen

REPO = "bigcode/commitpackft"
REVISION = "fc56fe33c030c6daa414c2b112c932b8eed085e6"
LANGUAGES = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "go": "go",
    "rust": "rust",
    "c": "c",
    "c++": "cpp",
    "java": "java",
    "shell": "bash",
}
LICENSES = PERMISSIVE_LICENSES
REPAIR = re.compile(
    r"\b(fix(?:es|ed|ing)?|bug|correct(?:ly|ion)?|repair|prevent|avoid|handle|ensure|resolve|support|allow|check|return|use|remove|replace|add|implement)\b",
    re.IGNORECASE,
)
BOILERPLATE = re.compile(r"^(merge|initial commit|bump|release|update version)\b", re.IGNORECASE)


def fingerprint(text):
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def repositories(value):
    if not isinstance(value, str):
        return []
    names = []
    for raw in value.split(","):
        name = (
            raw.strip().lower().removeprefix("https://github.com/").removesuffix(".git").strip("/")
        )
        if not re.fullmatch(r"[a-z0-9_.-]+/[a-z0-9_.-]+", name):
            return []
        names.append(name)
    return sorted(set(names))


class Families:
    def __init__(self):
        self.parents = {}

    def find(self, name):
        self.parents.setdefault(name, name)
        if self.parents[name] != name:
            self.parents[name] = self.find(self.parents[name])
        return self.parents[name]

    def add(self, names):
        for name in names:
            left, right = sorted((self.find(names[0]), self.find(name)))
            self.parents[right] = left


def split_for(group):
    return (
        "validation"
        if int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 20 == 0
        else "train"
    )


def adapt(raw, lang, families):
    names = repositories(raw.get("repos"))
    if not names:
        return None, "repository"
    if raw.get("license", "").lower() not in LICENSES:
        return None, "license"
    subject = raw.get("subject", "").strip()
    if not 12 <= len(subject) <= 600 or not REPAIR.search(subject) or BOILERPLATE.search(subject):
        return None, "instruction"
    old, new = raw.get("old_contents", ""), raw.get("new_contents", "")
    if not isinstance(old, str) or not isinstance(new, str):
        return None, "contents"
    if (
        not 25 <= len(old) <= 30000
        or not 25 <= len(new) <= 30000
        or fingerprint(old) == fingerprint(new)
    ):
        return None, "size_or_unchanged"
    filename = raw.get("new_file", "")
    if re.search(
        r"(^|/)(vendor|node_modules|dist|generated|third_party)/|\.min\.", filename, re.IGNORECASE
    ):
        return None, "generated_path"
    commit = raw.get("commit", "")
    if not re.fullmatch(r"[a-fA-F0-9]{40}", commit):
        return None, "commit"
    family = families.find(names[0])
    identity = "\0".join((lang, commit, filename, fingerprint(old), fingerprint(new)))
    return {
        "id": "commitpack-" + hashlib.sha256(identity.encode()).hexdigest(),
        "group": "github:" + family,
        "split": split_for(family),
        "language": lang,
        "instruction": subject,
        "broken": old,
        "fixed": new,
        "source": f"https://github.com/{names[0]}/commit/{commit}",
        "license": raw["license"].lower(),
        "repositories": names,
        "commit": commit,
        "old_file": raw.get("old_file", ""),
        "new_file": filename,
        "dataset": REPO,
        "dataset_revision": REVISION,
        "verification": "commit-derived; repository tests not rerun",
    }, None


def build(files, output, context=2048, train_cap=6000, validation_cap=64, excluded_codes=()):
    tokenizer = load_frozen()
    families = Families()
    for path in files.values():
        with open(path) as handle:
            for line in handle:
                raw = json.loads(line)
                families.add(repositories(raw.get("repos")))
    candidates, rejected = defaultdict(list), Counter()
    seen = set()
    for upstream, path in files.items():
        lang = LANGUAGES[upstream]
        with open(path) as handle:
            for line in handle:
                row, reason = adapt(json.loads(line), lang, families)
                if row is None:
                    rejected[reason] += 1
                    continue
                # Different commits with identical instruction/input/output are not extra data.
                duplicate = fingerprint(
                    row["instruction"] + "\n" + row["broken"] + "\n" + row["fixed"]
                )
                if duplicate in seen:
                    rejected["duplicate"] += 1
                    continue
                try:
                    item = repair_sft.REPAIR.encode(row, tokenizer, context)
                except ValueError:
                    rejected["context"] += 1
                    continue
                seen.add(duplicate)
                row["input_tokens"] = len(item["x"])
                row["answer_tokens"] = sum(y != -100 for y in item["y"])
                candidates[(row["split"], lang)].append(row)
        print(
            json.dumps(
                {
                    "language": lang,
                    "eligible": sum(len(v) for (s, l), v in candidates.items() if l == lang),
                }
            ),
            flush=True,
        )
    # Block all validation-side code from training, including records beyond the sampled eval cap.
    blocked = {fingerprint(code) for code in excluded_codes}
    for (split, _), records in candidates.items():
        if split == "validation":
            blocked.update(
                fingerprint(row[field]) for row in records for field in ("broken", "fixed")
            )
    selected = []
    reserved = {fingerprint(code) for code in excluded_codes}
    for (split, lang), records in sorted(candidates.items()):
        count = 0
        for row in sorted(records, key=lambda r: r["id"]):
            hashes = {fingerprint(row[field]) for field in ("broken", "fixed")}
            if hashes & (blocked if split == "train" else reserved):
                rejected["heldout_code_overlap"] += 1
                continue
            selected.append(row)
            count += 1
            if count >= (train_cap if split == "train" else validation_cap):
                break
    counts = Counter((r["split"], r["language"]) for r in selected)
    if any(
        counts[(s, lang)] < (100 if s == "train" else 8)
        for s in ("train", "validation")
        for lang in LANGUAGES.values()
    ):
        raise ValueError(f"Insufficient language coverage: {counts}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "examples.jsonl"
    destination.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in selected))
    report = {
        "dataset": REPO,
        "revision": REVISION,
        "context": context,
        "source_sha256": digest(__file__),
        "dataset_sha256": digest(destination),
        "source_files": {lang: {"sha256": digest(path)} for lang, path in files.items()},
        "counts": {
            s: {lang: counts[(s, lang)] for lang in LANGUAGES.values()}
            for s in ("train", "validation")
        },
        "unique_repositories": len({r["group"] for r in selected}),
        "input_tokens": sum(r["input_tokens"] for r in selected if r["split"] == "train"),
        "answer_tokens": sum(r["answer_tokens"] for r in selected if r["split"] == "train"),
        "rejections": dict(rejected),
        "limitations": [
            "Real commit instructions and before/after code; tests not rerun.",
            "Includes fixes and small instruction-driven edits, not only confirmed bugs.",
            "Repository/alias separation and normalized exact dedup; no semantic clone audit.",
            "No claim of absence from base-model pretraining.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report
