"""Bounded research campaign: correctness -> two-seed screen -> confirmation/evaluation."""

import json
import time
from pathlib import Path

import modal

from .autoresearch_specs import FINAL_STEPS, PROXY, SCREEN_STEPS, VARIANTS
from .branch_study import RECIPE, prepared_volume
from .modal_study import image, registry, runs

app = modal.App("kda-autoresearch")
research_image = image.add_local_file(
    "artifacts/autoresearch/glm-tokenizer.json", "/project/glm-tokenizer.json"
)


def record(run, call):
    calls = registry[run + "/calls"]
    calls.append(call.object_id)
    registry[run + "/calls"] = calls
    return call.object_id


def save(run, name, value):
    folder = Path("/runs") / run
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    runs.commit()


@app.function(timeout=86400, retries=0)
def supervisor(run, deadline):
    assert deadline - time.time() <= 23 * 3600
    registry[run + "/ready"] = True
    while time.time() < deadline:
        if registry.get(run + "/done", False):
            return
        time.sleep(min(10, max(0, deadline - time.time())))
    for cid in registry.get(run + "/calls", []):
        try:
            modal.FunctionCall.from_id(cid).cancel()
        except Exception as error:  # noqa: BLE001
            print("Cancellation error", cid, repr(error), flush=True)
    registry[run + "/expired"] = True


@app.function(
    image=research_image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=2700,
    retries=0,
    max_containers=4,
    volumes={"/runs": runs},
)
def qualify_arm(run, variant, deadline):
    from model_hybrid.autoresearch_checks import benchmark, counts, qualify

    assert time.time() < deadline - 600
    try:
        result = {
            "variant": variant,
            "correctness": qualify(variant),
            "counts": counts(variant),
            "benchmark": benchmark(variant),
        }
        # Reject structurally expensive candidates instead of burning a full run.
        result["status"] = (
            "qualified" if result["benchmark"]["slowdown"] <= 4 else "systems_rejected"
        )
    except Exception as error:  # noqa: BLE001
        import traceback

        result = {
            "variant": variant,
            "status": "qualification_failed",
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
    save(run, f"qualification/{variant}.json", result)
    print("QUALIFICATION", variant, result["status"], flush=True)
    return result


@app.function(
    image=research_image,
    cpu=2,
    memory=4096,
    timeout=300,
    retries=0,
    volumes={"/data": prepared_volume, "/runs": runs},
)
def preflight(run, deadline):
    from model_hybrid.prepared_data import heldout_index, shard_paths, training_order

    assert time.time() < deadline - 300
    result = {"sources": {}, "config": PROXY, "recipe": RECIPE}
    for name, src in RECIPE["sources"].items():
        count = 0
        indices = []
        required = int(2_000_000_000 * src["weight"]) + 32768
        for i in training_order(RECIPE, name):
            bp, _, mp = shard_paths(Path("/data") / RECIPE["name"], name, i)
            m = json.loads(mp.read_text())
            assert not m["heldout"] and bp.stat().st_size == m["tokens"] * 4
            count += m["tokens"]
            indices.append(i)
            if count >= required:
                break
        assert count >= required, (name, count, required)
        bp, dp, mp = shard_paths(Path("/data") / RECIPE["name"], name, heldout_index(src))
        m = json.loads(mp.read_text())
        assert m["heldout"] and dp.exists() and bp.stat().st_size == m["tokens"] * 4
        result["sources"][name] = {"tokens": count, "required": required, "shards": indices}
    save(run, "data.json", result)
    return result


@app.function(
    image=research_image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=72000,
    retries=0,
    max_containers=8,
    volumes={"/data": prepared_volume, "/runs": runs},
)
def train_arm(run, variant, seed, deadline, stage="screen"):
    from model_hybrid.autoresearch_train import train

    assert time.time() < deadline - 900
    runs.reload()
    source = (
        str(Path("/runs") / run / "screen" / f"{variant}-{seed}" / "checkpoint.pt")
        if stage == "confirm"
        else None
    )
    try:
        result = train(
            run,
            variant,
            seed,
            deadline,
            RECIPE,
            runs,
            stop_step=SCREEN_STEPS if stage == "screen" else FINAL_STEPS,
            source=source,
            stage=stage,
        )
        # Keep remote return values bounded; full curves live on the volume.
        return {k: result[k] for k in ("variant", "seed", "stage", "status", "steps_completed")} | {
            "final": result["curve"][-1] if result["curve"] else None
        }
    except Exception as error:  # noqa: BLE001
        import traceback

        result = {
            "variant": variant,
            "seed": seed,
            "stage": stage,
            "status": "failed",
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
        save(run, f"{stage}/{variant}-{seed}/failure.json", result)
        return result


@app.function(
    image=research_image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=3600,
    retries=0,
    max_containers=2,
    volumes={"/data": prepared_volume, "/runs": runs},
)
def evaluate_arm(run, variant, seed, deadline, stage="confirm", max_context=32768):
    from model_hybrid.autoresearch_eval import evaluate_checkpoint

    assert time.time() < deadline - 300
    runs.reload()
    checkpoint = str(Path("/runs") / run / stage / f"{variant}-{seed}" / "checkpoint.pt")
    try:
        result = evaluate_checkpoint(
            checkpoint, RECIPE, min(deadline, time.time() + 3300), max_context
        )
    except Exception as error:  # noqa: BLE001
        import traceback

        result = {"status": "failed", "error": repr(error), "traceback": traceback.format_exc()}
    save(run, f"evaluation/{stage}-{variant}-{seed}.json", result)
    return {"variant": variant, "seed": seed, "status": result["status"]}


def select(results):
    """Require both seeds and retain at most two quality candidates, even if inconclusive."""
    complete = {(r["variant"], r["seed"]): r for r in results if r["status"] == "complete"}
    if any(("baseline", s) not in complete for s in (17, 29)):
        return {"selected": [], "reason": "baseline_incomplete"}
    ranking = []
    for variant in VARIANTS:
        if variant == "baseline" or any((variant, s) not in complete for s in (17, 29)):
            continue
        deltas = [
            complete[variant, s]["final"]["validation"]["ce"]
            - complete["baseline", s]["final"]["validation"]["ce"]
            for s in (17, 29)
        ]
        ranking.append(
            {
                "variant": variant,
                "deltas": deltas,
                "mean_delta": sum(deltas) / 2,
                "both_seeds_better": max(deltas) < 0,
            }
        )
    ranking.sort(key=lambda r: r["mean_delta"])
    # Broad shortlist avoids early-ranking overconfidence. A +/-0.01 nat band
    # is a continuation criterion, not a statistical claim of improvement.
    shortlist = [r["variant"] for r in ranking if r["mean_delta"] < 0.01][:2]
    return {
        "selected": ["baseline", *shortlist],
        "ranking": ranking,
        "reason": "two_seed_screen_not_final_architecture_selection",
    }


@app.function(image=research_image, timeout=84600, retries=0, volumes={"/runs": runs})
def campaign(run, deadline):
    while not registry.get(run + "/registered", False):
        assert time.time() < deadline - 300
        time.sleep(1)
    outcomes = []
    qualification = []
    workers = []

    def spawn(fn, *args):
        assert time.time() < deadline - 300
        call = fn.spawn(*args)
        record(run, call)
        return call

    def wait_calls(pending, stage):
        results = []
        while pending and time.time() < deadline - 300:
            remaining = []
            for info in pending:
                try:
                    value = modal.FunctionCall.from_id(info["call"]).get(timeout=0)
                    results.append(value)
                except TimeoutError:
                    remaining.append(info)
                except Exception as error:  # noqa: BLE001
                    results.append({**info, "status": "failed", "error": repr(error)})
            pending = remaining
            save(run, f"{stage}-progress.json", {"results": results, "pending": pending})
            if pending:
                time.sleep(15)
        return results, pending

    try:
        spawn(preflight, run, deadline).get()
        calls = [
            {"variant": v, "call": spawn(qualify_arm, run, v, deadline).object_id} for v in VARIANTS
        ]
        qualification, pending = wait_calls(calls, "qualification")
        assert not pending, "qualification budget exhausted"
        if not any(
            q["variant"] == "baseline" and q["status"] == "qualified" for q in qualification
        ):
            raise RuntimeError("Baseline did not qualify")
        for q in qualification:
            if q["status"] != "qualified":
                continue
            for seed in (17, 29):
                call = spawn(train_arm, run, q["variant"], seed, deadline)
                workers.append(
                    {
                        "variant": q["variant"],
                        "seed": seed,
                        "stage": "screen",
                        "call": call.object_id,
                    }
                )
        registry[run + "/workers"] = workers
        outcomes, pending = wait_calls(workers.copy(), "screen")
        selection = select(outcomes)
        save(run, "selection.json", selection)
        confirmation = []
        if not pending and time.time() < deadline - 3600:
            for variant in selection["selected"]:
                for seed in (17, 29):
                    call = spawn(train_arm, run, variant, seed, deadline, "confirm")
                    confirmation.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "stage": "confirm",
                            "call": call.object_id,
                        }
                    )
            registry[run + "/workers"] = workers + confirmation
            # Evaluate the immutable screening checkpoints while confirmation runs.
            evals = [
                {
                    "variant": v,
                    "seed": s,
                    "call": spawn(evaluate_arm, run, v, s, deadline, "screen").object_id,
                }
                for v in selection["selected"]
                for s in (17, 29)
            ]
            confirmed, pending = wait_calls(confirmation, "confirm")
            outcomes += confirmed
            _, eval_pending = wait_calls(evals, "evaluation-screen")
            pending += eval_pending
            if not pending and time.time() < deadline - 1800:
                evals = [
                    {
                        "variant": r["variant"],
                        "seed": r["seed"],
                        "call": spawn(
                            evaluate_arm, run, r["variant"], r["seed"], deadline, "confirm", 131072
                        ).object_id,
                    }
                    for r in confirmed
                    if r["status"] == "complete"
                ]
                _, pending = wait_calls(evals, "evaluation-confirm")
        result = {
            "status": "complete" if not pending else "budget_exhausted",
            "qualification": qualification,
            "outcomes": outcomes,
            "pending": pending,
            "scope": "proxy screening and confirmation; target-scale training remains a separate campaign",
        }
    except Exception as error:  # noqa: BLE001
        import traceback

        result = {
            "status": "failed",
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "qualification": qualification,
            "outcomes": outcomes,
        }
    # Cancellation is explicit even when the controller encounters an exception.
    for cid in registry.get(run + "/calls", []):
        if cid == registry.get(run + "/controller"):
            continue
        try:
            modal.FunctionCall.from_id(cid).cancel()
        except Exception as error:  # noqa: BLE001
            print("Cancellation error", cid, repr(error), flush=True)
    save(run, "completion.json", result)
    registry[run + "/result"] = result
    registry[run + "/done"] = True
    return result
