"""Execute identical held-out repairs for the matched ordinary-SFT and CIG arms."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from modal_repair_eval import app, generate, grade


@app.local_entrypoint()
def evaluate_cig(folder: str = "results/cig-sft-20261020"):
    root = Path(folder)
    training = json.loads((root / "training.json").read_text())
    rows = [json.loads(line) for line in (root / "repairs.jsonl").read_text().splitlines()]
    by_id = {r["id"]: r for r in rows}
    calibration = {}
    for row in rows:
        if row["split"] == "validation":
            calibration.setdefault(row["language"], row)

    def calibrate(row):
        good, bad = grade(row, row["fixed"]), grade(row, row["broken"])
        if not good["passed"] or bad["passed"]:
            raise ValueError(f"Grader failed calibration: {row['id']}")
        return row["language"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        calibrated = list(pool.map(calibrate, calibration.values()))
    result = {"calibrated_languages": calibrated, "models": {}}
    for arm in ("sft", "cig"):
        checkpoint = str(Path(training[arm]["checkpoint"]).relative_to("/runs"))
        samples = generate.remote(checkpoint, rows)
        with ThreadPoolExecutor(max_workers=4) as pool:
            grades = list(pool.map(lambda s: grade(by_id[s["id"]], s["code"]), samples))
        for sample, grading in zip(samples, grades):
            sample["grading"] = grading
        result["models"][arm] = {
            "checkpoint": checkpoint,
            "samples": samples,
            "passed": sum(g["passed"] for g in grades),
            "total": len(grades),
            "eos": sum(s["eos"] for s in samples),
        }
        (root / "evaluation.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({"arm": arm, "passed": result["models"][arm]["passed"]}), flush=True)
