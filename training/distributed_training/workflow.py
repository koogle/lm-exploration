"""Pure checks used by the durable 5B workflow."""

import math
import re


def validate_paths(run_id, snapshot, resume):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run_id):
        raise ValueError("Invalid run ID")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+/snapshots/[a-zA-Z0-9_-]+", snapshot):
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


def check_progress(report, previous_tokens, target, global_positions):
    tokens = report["total_loss_tokens"]
    if not previous_tokens < tokens <= target:
        raise ValueError("Training made no progress or exceeded target")
    if not math.isfinite(report["final_validation"]["weighted_loss"]):
        raise ValueError("Non-finite validation")
    reason = report["stop_reason"]
    if reason == "token_limit":
        if target - tokens >= global_positions:
            raise ValueError("Token stop occurred too far below target")
        return True
    if reason not in ("time_limit", "step_limit"):
        raise ValueError(f"Training stopped unexpectedly: {reason}")
    return False


def check_resume(previous, resumed):
    if resumed["initial_validation"] != previous["final_validation"]:
        raise ValueError("Checkpoint resume changed validation")
