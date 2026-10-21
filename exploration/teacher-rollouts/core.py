"""CPU-only validation, append-safe records, accounting and benchmark summaries."""

import ast
import hashlib
import json
import math
import os
import time
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def validate_config(c):
    for key, low, high in [
        ("max_tokens", 1, 1024),
        ("max_model_len", 1024, 8192),
        ("repeats", 1, 16),
        ("rounds", 1, 3),
        ("runtime_seconds", 60, 1200),
    ]:
        if type(c[key]) is not int or not low <= c[key] <= high:
            raise ValueError(key)
    if not c["concurrency"] or any(
        type(n) is not int or not 1 <= n <= 32 for n in c["concurrency"]
    ):
        raise ValueError("concurrency")
    if len(set(c["concurrency"])) != len(c["concurrency"]):
        raise ValueError("duplicate concurrency")
    for key in [
        "gpu_usd_per_hour",
        "resource_usd_per_hour_ceiling",
        "attempt_usd_cap",
        "campaign_usd_cap",
    ]:
        if not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(key)
    if c["resource_usd_per_hour_ceiling"] < c["gpu_usd_per_hour"]:
        raise ValueError("resource rate must cover GPU")
    # Reserve the entire function timeout plus container shutdown allowance.
    reserve = (
        math.ceil(
            (c["runtime_seconds"] + 60)
            / 3600
            * c["resource_usd_per_hour_ceiling"]
            * 100
        )
        / 100
    )
    if reserve > c["attempt_usd_cap"] or c["attempt_usd_cap"] > c["campaign_usd_cap"]:
        raise ValueError("spending cap")
    if (
        not 0 <= c["temperature"] <= 2
        or not 0 < c["top_p"] <= 1
        or not 1 <= c["top_k"] <= 100
    ):
        raise ValueError("sampling")
    return reserve


def reserve_budget(c, directory, attempt):
    """Never refund automatically: failures, retries and resumes consume reservations."""
    import fcntl

    reserve = validate_config(c)
    if c["budget_authorized"] is not True or not c["authorization_reference"].strip():
        raise ValueError("No explicit budget authorization recorded")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = directory / "ledger.json"
        ledger = json.loads(path.read_text()) if path.exists() else []
        if any(r["attempt"] == attempt for r in ledger):
            raise ValueError("attempt already reserved; use a new attempt ID")
        if sum(r["reserved_usd"] for r in ledger) + reserve > c["campaign_usd_cap"]:
            raise ValueError("campaign budget exhausted")
        ledger.append(
            {
                "attempt": attempt,
                "reserved_usd": reserve,
                "authorization_reference": c["authorization_reference"],
                "time": time.time(),
            }
        )
        atomic_json(path, ledger)
    return reserve


def validate(prompt, output, finish):
    reasons = []
    if finish != "stop":
        reasons.append("truncated" if finish == "length" else "non_stop_finish")
    if not output or not output.strip():
        reasons.append("empty_output")
    try:
        if prompt["kind"] == "code":
            # Intentionally tiny grammar, interpreted without executing generated Python.
            tree = ast.parse(output)
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
                raise ValueError("one function required")
            fn = tree.body[0]
            if (
                fn.name != "solve"
                or fn.decorator_list
                or fn.returns
                or fn.args.defaults
                or fn.args.kw_defaults
                or fn.args.kwonlyargs
                or fn.args.posonlyargs
                or fn.args.vararg
                or fn.args.kwarg
                or [a.arg for a in fn.args.args] != prompt["args"]
                or any(a.annotation for a in fn.args.args)
                or len(fn.body) != 1
                or not isinstance(fn.body[0], ast.Return)
            ):
                raise ValueError("function signature or body")
            for args, expected in prompt["cases"]:
                if (
                    expression(fn.body[0].value, dict(zip(prompt["args"], args)))
                    != expected
                ):
                    reasons.append("code_test_failed")
                    break
        else:

            def pairs(items):
                d = {}
                for key, value in items:
                    if key in d:
                        raise ValueError("duplicate JSON key")
                    d[key] = value
                return d

            obj = json.loads(output, object_pairs_hook=pairs)
            if prompt["kind"] == "structured":
                if digest(obj) != digest(prompt["expected"]):
                    reasons.append("schema_or_value_mismatch")
            else:
                if (
                    not isinstance(obj, dict)
                    or set(obj) != {"answer", "explanation"}
                    or digest(obj["answer"]) != digest(prompt["answer"])
                ):
                    reasons.append("technical_answer_mismatch")
                explanation = (
                    obj.get("explanation", "") if isinstance(obj, dict) else ""
                )
                if (
                    not isinstance(explanation, str)
                    or not 30 <= len(explanation) <= 800
                    or not all(t in explanation.lower() for t in prompt["terms"])
                ):
                    reasons.append("explanation_rubric_failed")
    except (ValueError, SyntaxError, TypeError, KeyError, RecursionError):
        reasons.append("invalid_format_or_unsupported_code")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "validator_version": 1,
        "quality_scope": "synthetic_fixture_checks",
        "human_review_required": prompt["kind"] == "technical",
    }


def expression(node, names):
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -expression(node.operand, names)
    if isinstance(node, ast.IfExp):
        return expression(
            node.body if expression(node.test, names) else node.orelse, names
        )
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        a, b = expression(node.left, names), expression(node.comparators[0], names)
        ops = {
            ast.Lt: lambda: a < b,
            ast.LtE: lambda: a <= b,
            ast.Gt: lambda: a > b,
            ast.GtE: lambda: a >= b,
            ast.Eq: lambda: a == b,
        }
        if type(node.ops[0]) in ops:
            return ops[type(node.ops[0])]()
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and not node.keywords
    ):
        funcs = {"min": min, "max": max, "abs": abs}
        if node.func.id in funcs and 1 <= len(node.args) <= 3:
            return funcs[node.func.id](*(expression(a, names) for a in node.args))
    raise ValueError("unsupported expression")


class Store:
    """One immutable file per example, explicit identity checking on resume."""

    def __init__(self, root, manifest):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("resume manifest mismatch")
        if not path.exists():
            atomic_json(path, manifest)
        self.records = self.root / "records"
        self.records.mkdir(exist_ok=True)

    def read(self, key):
        path = self.records / f"{key}.json"
        if not path.exists():
            return None
        record = json.loads(
            path.read_text()
        )  # Corruption fails closed; never silently regenerate.
        if record["example_id"] != key:
            raise ValueError("record identity mismatch")
        return record

    def put(self, record):
        key = record["example_id"]
        old = self.read(key)
        if old is not None:
            if old != record:
                raise ValueError("conflicting duplicate")
            return False
        atomic_json(self.records / f"{key}.json", record)
        return True


def summarize(records, seconds, gpu_rate):
    completed = [r for r in records if not r.get("error")]
    accepted = [r for r in completed if r["validation"]["accepted"]]
    tokens = sum(r["output_tokens"] for r in completed)

    def lengths(key):
        values = [r[key] for r in completed]
        return (
            {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}
            if values
            else None
        )

    return {
        "requests": len(records),
        "completed": len(completed),
        "accepted": len(accepted),
        "generated_tokens_per_second": tokens / seconds,
        "completed_examples_per_hour": len(completed) * 3600 / seconds,
        "accepted_examples_per_hour": len(accepted) * 3600 / seconds,
        "acceptance_rate": len(accepted) / len(completed) if completed else None,
        "truncation_rate": sum(r["finish_reason"] == "length" for r in completed)
        / len(completed)
        if completed
        else None,
        "prompt_lengths": lengths("prompt_tokens"),
        "output_lengths": lengths("output_tokens"),
        "steady_generation_seconds": seconds,
        "estimated_steady_gpu_usd_per_accepted": seconds
        * gpu_rate
        / 3600
        / len(accepted)
        if accepted
        else None,
        "actual_billed_usd": None,
        "verification_seconds": sum(r["verification_seconds"] for r in records),
    }


def select_best(trials, rounds):
    """Require every round and include zero-acceptance rounds in the denominator."""
    groups = {}
    for trial in trials:
        groups.setdefault(trial["concurrency"], []).append(trial)
    rates = {}
    for concurrency, rows in groups.items():
        if (
            len(rows) == rounds
            and all(r["comparable"] for r in rows)
            and {r["round"] for r in rows} == set(range(rounds))
        ):
            accepted = sum(r["accepted"] for r in rows)
            if accepted:
                rates[concurrency] = accepted / sum(
                    r["steady_generation_seconds"] for r in rows
                )
    return max(rates, key=rates.get) if rates else None
