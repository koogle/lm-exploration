"""Pure checks used by the durable 5B workflow."""

import math


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
