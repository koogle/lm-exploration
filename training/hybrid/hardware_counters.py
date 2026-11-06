"""Nsight Compute access probe and rank-zero wrapper for the existing benchmark.

Counter replay timings are diagnostic, never throughput/MFU measurements.
"""

import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SECTIONS = (
    "SpeedOfLight",
    "MemoryWorkloadAnalysis",
    "SchedulerStats",
    "WarpStateStats",
    "LaunchStats",
    "Occupancy",
)
KERNELS = (
    "chunk_kda_bwd_kernel_intra",
    "chunk_kda_bwd_kernel_wy_dqkg_fused",
    "softmax_gradient",
    "triton_poi_fused__to_copy",
    "sm90_xmma_gemm",
)


def executable():
    found = shutil.which("ncu")
    if found:
        return found
    paths = sorted(Path("/opt/nvidia").glob("nsight-compute*/ncu"))
    if not paths:
        raise FileNotFoundError("Nsight Compute is not installed")
    return str(paths[-1])


def measured_rows(text):
    """Ignore diagnostic banners and require actual numeric performance counters."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if '"Metric Name"' in line), None)
    if start is None:
        return []
    return list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))


def raw_measurements(text):
    """Nsight's raw page is wide CSV, with a units row before kernel rows."""
    rows = list(csv.DictReader(io.StringIO(text)))
    return [row for row in rows if row.get("ID", "").isdigit() and row.get("Kernel Name")]


def probe(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    ncu = executable()
    command = [
        ncu,
        "--clock-control",
        "none",
        "--cache-control",
        "none",
        "--metrics",
        "sm__cycles_elapsed.avg,dram__bytes_read.sum",
        "--launch-count",
        "1",
        "--csv",
        "--log-file",
        str(folder / "probe.csv"),
        sys.executable,
        "-c",
        "import torch; x=torch.ones(1024,device='cuda'); x.add_(1); torch.cuda.synchronize()",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
    (folder / "probe.log").write_text(result.stdout + result.stderr)
    table = (folder / "probe.csv").read_text() if (folder / "probe.csv").exists() else ""
    rows = measured_rows(table)
    measured = {
        row.get("Metric Name") for row in rows if row.get("Metric Value") not in (None, "", "n/a")
    }
    evidence = {
        "exit_code": result.returncode,
        "accessible": result.returncode == 0 and "sm__cycles_elapsed.avg" in measured,
        "permission_denied": "ERR_NVGPUCTRPERM" in table + result.stdout + result.stderr,
        "command": command,
        "ncu_version": subprocess.check_output([ncu, "--version"], text=True).strip(),
        "measured_metrics": sorted(measured),
    }
    (folder / "probe.json").write_text(json.dumps(evidence, indent=2))
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--kernel", choices=KERNELS, required=True)
    args, benchmark_args = parser.parse_known_args()
    if benchmark_args[:1] == ["--"]:
        benchmark_args = benchmark_args[1:]
    command = [sys.executable, "-m", "model_hybrid.benchmark", *benchmark_args]
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        # Only one selected kernel launch is replayed. NCCL is never selected.
        # Warmup is outside cudaProfilerStart/Stop in the benchmark's step hook.
        command = [
            executable(),
            "--profile-from-start",
            "off",
            "--replay-mode",
            "kernel",
            "--kernel-name",
            "regex:" + args.kernel,
            "--launch-count",
            "1",
            "--clock-control",
            "none",
            "--cache-control",
            "none",
            "--export",
            args.output,
            "--force-overwrite",
            *[part for section in SECTIONS for part in ("--section", section)],
            *command,
            "--hardware-counters",
        ]
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
