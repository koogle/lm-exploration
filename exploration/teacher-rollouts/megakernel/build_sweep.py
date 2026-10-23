"""CPU-only, pinned CUDA builds for the launch-geometry/ThunderKittens sweep."""

import argparse
import hashlib
import io
import json
import statistics
import subprocess
import tarfile
import urllib.request
from pathlib import Path

TK_REVISION = "67845f5fa48d05343dbbc1ba1403f11061f08d2d"
TK_SHA256 = "383868318e06a0b7730b3013c0f74bb61d86b23484f7f4a3b098f309ac92ab7a"
HERE = Path(__file__).parent


def variants():
    return [
        {
            "id": f"{backend}-t{threads}-g{blocks}",
            "backend": backend,
            "threads": threads,
            "blocks": blocks,
        }
        for backend in ("manual", "thunderkittens")
        for threads in (256, 512)
        for blocks in (66, 132)
    ]


def schedule():
    plan = variants()
    return [
        (round_id, v)
        for round_id, order in enumerate((plan, plan[::-1]))
        for v in order
    ]


def rank_trials(trials):
    """Never rank incomplete, failed, or reference-divergent variants."""
    ranked = []
    for variant in variants():
        rows = [t for t in trials if t["variant"]["id"] == variant["id"]]
        if len(rows) != 2 or {t["round"] for t in rows} != {0, 1}:
            continue
        if not all(
            t["result"].get("status") == "completed"
            and t["result"].get("correctness_passed")
            and t["result"].get("reset_reproducible")
            and t["result"].get("performance_eligible")
            for t in rows
        ):
            continue
        speeds = [
            next(
                m["accepted_examples_per_hour"]
                for m in t["result"]["measurements"]
                if m["backend"] == "megakernel"
            )
            for t in rows
        ]
        ranked.append(
            {
                "variant": variant,
                "accepted_examples_per_hour_trials": speeds,
                "median_accepted_examples_per_hour": statistics.median(speeds),
            }
        )
    return sorted(
        ranked, key=lambda r: r["median_accepted_examples_per_hour"], reverse=True
    )


def build(destination, archive=None):
    destination.mkdir(parents=True, exist_ok=True)
    data = (
        archive.read_bytes()
        if archive
        else urllib.request.urlopen(
            f"https://codeload.github.com/HazyResearch/ThunderKittens/tar.gz/{TK_REVISION}",
            timeout=120,
        ).read()
    )
    if hashlib.sha256(data).hexdigest() != TK_SHA256:
        raise ValueError("ThunderKittens archive digest mismatch")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(destination / "third_party", filter="data")
    tk = destination / "third_party" / f"ThunderKittens-{TK_REVISION}"
    record = {
        "thunderkittens_revision": TK_REVISION,
        "archive_sha256": TK_SHA256,
        "variants": [],
    }
    for variant in variants():
        command = [
            "nvcc",
            "-std=c++20",
            "-O3",
            "-arch=sm_90",
            "-DTARGET_SM=90",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--shared",
            "-Xcompiler=-fPIC",
            f"-DBLOCK_SIZE={variant['threads']}",
            f"-DNUM_BLOCKS={variant['blocks']}",
        ]
        if variant["backend"] == "thunderkittens":
            command += ["-DUSE_THUNDERKITTENS", "-DKITTENS_SM90", f"-I{tk / 'include'}"]
        command += [
            str(HERE / "kernel.cu"),
            "-o",
            str(destination / f"{variant['id']}.so"),
        ]
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=300
        )
        (destination / f"{variant['id']}.build.log").write_text(
            result.stdout + result.stderr
        )
        if result.returncode:
            raise RuntimeError(
                f"Build failed: {variant['id']}: {result.stderr[-4000:]}"
            )
        record["variants"].append({**variant, "command": command})
    (destination / "build.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    build(args.destination, args.archive)
