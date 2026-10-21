"""Bounded audit-pair coordinator with durable claims and no ambiguous replay."""

from __future__ import annotations

import copy
import math
import time
import uuid

SOURCES = ("ecb", "ema")
TERMINAL = {"complete", "failed"}
STOPPED_GOALS = {"token_target_reached", "target_reached", "selected_scope_exhausted"}


def initial_state(existing_calls=None):
    state = {"schema_version": 1, "pairs": []}
    if existing_calls:
        if set(existing_calls) != set(SOURCES) or any(
            not isinstance(call_id, str) or not call_id for call_id in existing_calls.values()
        ):
            raise ValueError("Adoption requires exactly one existing ECB and EMA call")
        state["pairs"].append(
            {
                "pair_id": f"adopted-{uuid.uuid4().hex}",
                "created_at_epoch": time.time(),
                "jobs": {
                    source: {"status": "running", "call_id": call_id}
                    for source, call_id in existing_calls.items()
                },
            }
        )
    return state


def run_tick(cache, goal_status, spawn_callable, inspect_callable, max_pairs=48, clock=time.time):
    """Inspect without waiting, then durably reserve and spawn at most one pair.

    ``cache`` implements AuditCache.load/save at tokens/schedule.json. Bootstrap
    existing calls using initial_state before enabling the schedule. A missing
    call ID after a claim is intentionally unresolved forever, requiring manual
    reconciliation rather than a potentially duplicate spawn.
    """
    if type(max_pairs) is not int or not 1 <= max_pairs <= 96:
        raise ValueError("max_pairs must be an integer in [1, 96]")
    state = cache.load()
    if state is None:
        state = initial_state()
    if state.get("schema_version") != 1 or not isinstance(state.get("pairs"), list):
        raise ValueError("Incompatible audit schedule state")
    now = clock()
    for pair in state["pairs"]:
        if set(pair["jobs"]) != set(SOURCES):
            raise ValueError("Incomplete reserved audit pair")
        for job in pair["jobs"].values():
            if job.get("status") in TERMINAL:
                continue
            if not job.get("call_id"):
                job["status"] = "dispatch_gap"
                continue
            try:
                observation = inspect_callable(job["call_id"])
            except Exception as error:  # noqa: BLE001 - uncertainty must retain the claim
                observation = {"status": "unknown", "error_type": type(error).__name__}
            if not isinstance(observation, dict):
                observation = {"status": "unknown"}
            status = observation.get("status")
            job["status"] = status if status in TERMINAL | {"running"} else "unknown"
            job["last_observation"] = copy.deepcopy(observation)
            job["inspected_at_epoch"] = now
            if job["status"] == "failed":
                seconds = observation.get("retry_after_seconds", 0)
                if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
                    seconds = 0
                if observation.get("http_status") == 429:
                    seconds = max(3600, seconds)
                job["cooldown_until_epoch"] = now + seconds

    unresolved = sum(
        job.get("status") not in TERMINAL
        for pair in state["pairs"]
        for job in pair["jobs"].values()
    )
    cooldown = max(
        (
            job.get("cooldown_until_epoch", 0)
            for pair in state["pairs"]
            for job in pair["jobs"].values()
        ),
        default=0,
    )
    if goal_status in STOPPED_GOALS:
        status = "goal_stopped"
    elif unresolved:
        status = "waiting_for_calls"
    elif now < cooldown:
        status = "cooldown"
    elif len(state["pairs"]) >= max_pairs:
        status = "pair_budget_exhausted"
    else:
        status = "ready"

    def summary(status):
        return {
            "status": status,
            "pairs_reserved": len(state["pairs"]),
            "max_pairs": max_pairs,
            "unresolved_calls": sum(
                job.get("status") not in TERMINAL
                for pair in state["pairs"]
                for job in pair["jobs"].values()
            ),
            "cooldown_until_epoch": cooldown,
            "observed_at_epoch": now,
        }

    if status != "ready":
        state["summary"] = summary(status)
        cache.save(state)
        return copy.deepcopy(state["summary"])

    pair = {
        "pair_id": uuid.uuid4().hex,
        "created_at_epoch": now,
        "jobs": {source: {"status": "claimed", "call_id": None} for source in SOURCES},
    }
    state["pairs"].append(pair)
    state["summary"] = summary("dispatching")
    cache.save(state)  # Both source claims must win CAS before either RPC starts.
    for source in SOURCES:
        call = spawn_callable(source)
        call_id = call if isinstance(call, str) else getattr(call, "object_id", None)
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("Spawn must return a nonempty call ID or object_id")
        pair["jobs"][source].update(status="running", call_id=call_id)
        state["summary"] = summary("waiting_for_calls")
        cache.save(state)  # Never retry a spawn if attachment fails.
    return copy.deepcopy(state["summary"])
