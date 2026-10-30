"""Pure path, capacity, progress and resume checks for training workflows."""

import math
import re


def validate_paths(run_id, snapshot, resume):
    """Run ID, data snapshot (None for families whose run config names its data) and
    resume checkpoint."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run_id):
        raise ValueError("Invalid run ID")
    if snapshot is not None and not re.fullmatch(
        r"[a-zA-Z0-9_-]+/snapshots/[a-zA-Z0-9_-]+", snapshot
    ):
        raise ValueError("Expected version/snapshots/name")
    if resume and not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}/step-[0-9]{9}\.pt", resume):
        raise ValueError("Resume must name a trusted run-id/step-NNNNNNNNN.pt on our run volume")
    if resume and resume.split("/")[0] == run_id:
        raise ValueError("Resume uses a fresh run ID to preserve the previous job")


def check_capacity(report, target):
    if report["supported_mixture_exposures"] < target:
        raise ValueError("Not enough unique data for the target under the two-pass mixture policy")
    if any(s["validation_loss_tokens"] <= 0 for s in report["sources"].values()):
        raise ValueError("Every source needs validation data")


def _numbers(value):
    if isinstance(value, dict):
        return [n for v in value.values() for n in _numbers(v)]
    if isinstance(value, list):
        return [n for v in value for n in _numbers(v)]
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    return [value] if is_number else []


def check_progress(report, previous_tokens, target, global_positions):
    tokens = report["tokens"]
    reason = report["stop_reason"]
    # A session can spend its whole time waiting for prepared data (data/growing.py).
    if not (previous_tokens < tokens or reason == "data_late" and previous_tokens == tokens):
        raise ValueError("Training made no progress")
    if tokens > target:
        raise ValueError("Training exceeded its target")
    if not all(math.isfinite(n) for n in _numbers(report["final_validation"])):
        raise ValueError("Non-finite validation")
    if reason == "token_limit":
        if target - tokens >= global_positions:
            raise ValueError("Token stop occurred too far below target")
        return True
    if reason not in ("time_limit", "step_limit", "data_late"):
        raise ValueError(f"Training stopped unexpectedly: {reason}")
    return False


def _matches(a, b, relative):
    """Equal structures; numbers within `relative` tolerance, text ignored when relative > 0."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_matches(a[k], b[k], relative) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_matches(x, y, relative) for x, y in zip(a, b))
    numbers = (int, float)
    if isinstance(a, numbers) and isinstance(b, numbers) and not isinstance(a, bool):
        return math.isclose(a, b, rel_tol=relative, abs_tol=relative)
    return relative > 0 and isinstance(a, str) and isinstance(b, str) or a == b


def check_resume(previous, resumed, relative=0.0):
    """A resumed session's first evaluation reproduces the previous session's last one,
    exactly, or within `relative` for families whose evaluation kernels are not
    bit-reproducible across processes (generated samples are then not compared)."""
    if not _matches(previous["final_validation"], resumed["initial_validation"], relative):
        raise ValueError("Checkpoint resume changed validation")
