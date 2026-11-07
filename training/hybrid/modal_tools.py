"""Hybrid tools on Modal H100s: benchmarks, held-out evaluation, export and publishing.

Training runs through the workflows like every model family (workflows/README.md).
Run from training/:
    modal run -m model_hybrid.modal_tools::bench --gpus 1 [--microbatch 4]
    modal run -m model_hybrid.modal_tools::heldout --export <run>/export/<name>
    modal run -m model_hybrid.modal_tools::export --run-id <run folder>
    modal run -m model_hybrid.modal_tools::publish --checkpoint <run>/step-NNNNNNNNN.pt \\
        --repo <owner/name> [--public] [--card <README.md>]

Each benchmark is one call with a 30-minute timeout and no retries. `publish` uploads
one checkpoint from the runs volume as bf16 weights, config, progress and the run's
logs (private unless --public), with an optional model card; one CPU call, one hour,
no retries.
"""

import json
import shutil
import subprocess
import time
from pathlib import Path

import modal
from model_hybrid.image import build, image

counter_image = build(
    extra_packages=("torchao==0.18.0",), apt_packages=("cuda-nsight-compute-13-0",)
)
app = modal.App("lm-exploration-hybrid-tools")
data_volume = modal.Volume.from_name("lm-exploration-hybrid-data", create_if_missing=True)
runs_volume = modal.Volume.from_name("lm-exploration-hybrid-runs", create_if_missing=True)
DEFAULT_CONFIG = "hybrid/configs/hybrid-720m-40b.json"
VOLUMES = {"/data": data_volume, "/runs": runs_volume}


def consolidation_variants(selection):
    allowed = ("joint-loss", "projections", "norm", "all")
    cases = tuple(selection.split(",")) if selection else allowed
    if len(set(cases)) != len(cases) or any(case not in allowed for case in cases):
        raise ValueError(f"Select distinct consolidation variants from {allowed}")
    return cases


def run_torchrun(module, args, gpus, timeout):
    """torchrun's exit code and output lines (printed as they arrive)."""
    command = ["torchrun", "--standalone", f"--nproc_per_node={gpus}", "-m", module, *args]
    lines = []
    process = subprocess.Popen(
        command, cwd="/project", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    start = time.monotonic()
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
        if time.monotonic() - start > timeout:
            process.terminate()
    return process.wait(), lines


@app.function(
    image=counter_image,
    gpu="H100:8",
    cpu=32,
    memory=262144,
    timeout=1800,
    retries=0,
    volumes=VOLUMES,
)
def benchmark(
    config,
    gpus,
    steps,
    microbatch,
    global_rows,
    document_tokens,
    deadline,
    profile=False,
    run_id="",
    real_data=False,
    compare_output_gradient=False,
    larger_microbatch=0,
    autotune=False,
    kda_save_intermediates=False,
    required_gpu_model="",
    hardware_kernel="",
    compare_kda_wy=False,
    phase_timings=False,
    compare_graphed_muon=False,
    compare_intra=False,
    compare_consolidation=False,
    consolidation_cases="",
    profile_all_ranks=False,
    profile_stacks=False,
    compare_scalar_delta=False,
    compare_fp8=False,
):
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    from model_hybrid.benchmark import summarize

    path = Path("/tmp/run-config.json")
    path.write_text(json.dumps(config))
    args = ["--config", str(path), "--steps", str(steps), "--document-tokens", str(document_tokens)]
    if microbatch:
        args += ["--microbatch", str(microbatch)]
    if global_rows:
        args += ["--global-rows", str(global_rows)]
    if phase_timings:
        args += ["--phase-timings"]
    if profile:
        args += ["--profile", "/tmp/profile.json"]
    if profile_all_ranks:
        args += ["--profile-all-ranks"]
    if profile_stacks:
        args += ["--profile-stacks"]
    if real_data:
        args += ["--real-data", "--data-root", "/data"]
    hardware = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,power.limit",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        .strip()
        .splitlines()
    )
    print(json.dumps({"hardware": hardware}), flush=True)
    if required_gpu_model and any(
        required_gpu_model not in row.split(",", 1)[0] for row in hardware
    ):
        if run_id:
            folder = Path("/runs/throughput") / run_id
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "hardware-rejected.json").write_text(
                json.dumps(
                    {
                        "required_gpu_model": required_gpu_model,
                        "hardware": hardware,
                        "reason": "Delivered hardware does not match qualification target",
                    },
                    indent=2,
                )
            )
            runs_volume.commit()
        raise RuntimeError(f"Required {required_gpu_model}; delivered hardware: {hardware}")
    if hardware_kernel:
        from model_hybrid.hardware_counters import probe

        access = probe(Path("/runs/throughput") / run_id)
        runs_volume.commit()
        if not access["accessible"]:
            raise RuntimeError(f"Hardware counters unavailable: {access}")
    cases = (
        [("reference", ["--reference-output-gradient"]), ("direct", [])]
        if compare_output_gradient
        else [("direct", [])]
    )
    if compare_kda_wy:
        cases = [
            ("original_first", ["--reference-kda-wy"]),
            ("kda_wy", []),
            ("original_repeat", ["--reference-kda-wy"]),
        ]
    if compare_graphed_muon:
        cases = [
            ("original_first", ["--reference-muon"]),
            ("graphed_muon", []),
            ("original_repeat", ["--reference-muon"]),
        ]
    if compare_intra:
        cases = [
            ("original_first", ["--reference-intra"]),
            ("adaptive_intra", []),
            ("original_repeat", ["--reference-intra"]),
        ]
    if compare_fp8:
        checked = subprocess.run(
            ["python", "/project/tests/check_float8_cuda.py"],
            cwd="/project",
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        folder = Path("/runs/throughput") / run_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "fp8-check.log").write_text(checked.stdout + checked.stderr)
        runs_volume.commit()
        if checked.returncode:
            raise RuntimeError(checked.stdout + checked.stderr)
        cases = [("bf16_first", []), ("fp8_linears", ["--fp8-linears"]), ("bf16_repeat", [])]
    if compare_scalar_delta:
        cases = [("kda_first", []), ("scalar_delta", ["--scalar-delta"]), ("kda_repeat", [])]
    if compare_consolidation:
        cases = (
            [("original_first", [])]
            + [
                (candidate, ["--consolidation", candidate])
                for candidate in consolidation_variants(consolidation_cases)
            ]
            + [("original_repeat", [])]
        )
    if larger_microbatch:
        cases.append(("larger_microbatch", ["--microbatch", str(larger_microbatch)]))
        cases.append(("original_repeat", []))
    if autotune:
        cases.append(("autotuned", ["--compile-mode", "max-autotune-no-cudagraphs"]))
    if kda_save_intermediates:
        cases.append(("kda_saved", ["--kda-save-intermediates"]))
    measurements, all_lines = {}, []
    code = 0
    for name, extra in cases:
        module, arguments = "model_hybrid.benchmark", args + extra
        if hardware_kernel:
            module = "model_hybrid.hardware_counters"
            arguments = ["--output", "/tmp/counters", "--kernel", hardware_kernel, "--", *arguments]
        code, lines = run_torchrun(module, arguments, gpus, 1500)
        measurements[name] = {
            "arguments": extra,
            "exit_code": code,
            **(
                summarize(lines, skip=10 if profile else 5)
                if code == 0 and not hardware_kernel
                else {}
            ),
        }
        all_lines.extend([json.dumps({"case": name}) + "\n", *lines])
        if run_id:
            folder = Path("/runs/throughput") / run_id
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{name}.log").write_text("".join(lines))
            (folder / f"{name}.json").write_text(json.dumps(measurements[name], indent=2))
            runs_volume.commit()
        if code and not compare_consolidation:
            break
    code = next((m["exit_code"] for m in measurements.values() if m["exit_code"]), 0)
    lines = all_lines
    import torch

    result = {
        "exit_code": code,
        "hardware_counter_kernel": hardware_kernel,
        "throughput_valid": not (hardware_kernel or profile),
        "hardware": hardware,
        "measurements": measurements,
        "config": config,
        "real_data": real_data,
        "recipe": json.loads(Path(config["recipe"]).read_text()) if real_data else None,
        "global_rows": global_rows or config["global_rows"],
        "gpus": gpus,
        "microbatch": microbatch or config["microbatch"],
        "torch": torch.__version__,
        **(measurements["direct"] if len(cases) == 1 and code == 0 else {}),
        **(
            {"profile": json.loads(Path("/tmp/profile.json").read_text())}
            if profile and Path("/tmp/profile.json").exists()
            else {}
        ),
    }
    if run_id:
        folder = Path("/runs/throughput") / run_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "torchrun.log").write_text("".join(lines))
        (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        trace = Path("/tmp/profile.json.trace.json.gz")
        if profile and trace.exists():
            shutil.copyfile(trace, folder / "trace.json.gz")
        if profile_all_ranks:
            for rank in range(1, gpus):
                for suffix in ("", ".trace.json.gz"):
                    source = Path(f"/tmp/profile.json.rank{rank}{suffix}")
                    if source.exists():
                        shutil.copyfile(source, folder / source.name)
        if hardware_kernel:
            from model_hybrid.hardware_counters import executable, raw_measurements

            report = Path("/tmp/counters.ncu-rep")
            if report.exists():
                shutil.copyfile(report, folder / report.name)
                exported = subprocess.run(
                    [executable(), "--import", str(report), "--page", "raw", "--csv"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                (folder / "counters.csv").write_text(exported.stdout)
                (folder / "counter-export.log").write_text(exported.stderr)
                result["counter_collection_valid"] = exported.returncode == 0 and bool(
                    raw_measurements(exported.stdout)
                )
            else:
                result["counter_collection_valid"] = False
            (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        runs_volume.commit()
    if code:
        raise RuntimeError("".join(lines[-60:]))
    if hardware_kernel and not result.get("counter_collection_valid"):
        raise RuntimeError("Nsight Compute produced no metric rows; inspect archived logs")
    return result


@app.function(image=image, cpu=0.25, memory=512, timeout=86400, retries=0)
def enforce_deadline(call_ids, deadline):
    """Use the shared supervisor for bounded throughput calls, including queue time."""
    from workflows.deadline import calls_finished, cancel_at_deadline

    return cancel_at_deadline(deadline, lambda: call_ids, lambda: calls_finished(call_ids))


@app.function(
    image=image,
    cpu=8,
    memory=65536,
    timeout=3600,
    retries=0,
    secrets=[modal.Secret.from_name("simple-llm-hf")],
    volumes=VOLUMES,
)
def export_checkpoint(checkpoint, repo, private, card, deadline):
    """One trusted checkpoint written by trainer.loop on our own runs volume, to HF."""
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    import torch
    from model_hybrid.train import RunConfig, family_of, upload_export

    path = Path("/runs") / checkpoint
    saved = torch.load(path, map_location="cpu", weights_only=False)
    config = RunConfig(**saved["config"])
    family = family_of(config)(config, path.parent, data=({}, None))
    lm = family.build("cpu")
    lm.load_state_dict(saved["model"])
    progress = {"step": saved["step"], "tokens": saved["tokens"]}
    upload_export(lm, family.model_config, path.parent, progress, repo, private).join()
    if card:
        from huggingface_hub import HfApi

        HfApi().upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo)
    return {"repo": repo, "private": private, **progress}


def load_config(path):
    return json.loads(Path(path).read_text())


@app.local_entrypoint()
def bench(
    config: str = DEFAULT_CONFIG,
    gpus: int = 1,
    steps: int = 20,
    microbatch: int = 0,
    global_rows: int = 0,
    document_tokens: int = 1000,
    profile: bool = False,
    mtp_loss_fraction: float = 0.0,
    run_id: str = "",
    real_data: bool = False,
    compare_output_gradient: bool = False,
    larger_microbatch: int = 0,
    gpu: str = "H100",
    autotune: bool = False,
    kda_save_intermediates: bool = False,
    require_gpu_model: bool = False,
    hardware_kernel: str = "",
    compare_kda_wy: bool = False,
    phase_timings: bool = False,
    compare_graphed_muon: bool = False,
    compare_intra: bool = False,
    compare_consolidation: bool = False,
    consolidation_cases: str = "",
    profile_all_ranks: bool = False,
    profile_stacks: bool = False,
    compare_scalar_delta: bool = False,
    compare_fp8: bool = False,
):
    """One bounded throughput benchmark; with one GPU the global batch shrinks to fit.

    `--profile` adds GPU kernel time per category for a few steps (see benchmark.py);
    `--mtp-loss-fraction` overrides the model's MTP loss fraction.
    """
    from workflows.deadline import new_deadline

    if (profile_all_ranks or profile_stacks) and not profile:
        raise ValueError("Rank and stack profiling require profile")
    if consolidation_cases and not compare_consolidation:
        raise ValueError("consolidation_cases requires compare_consolidation")
    if compare_consolidation:
        consolidation_variants(consolidation_cases)
    if gpu not in ("H100", "H200"):
        raise ValueError("Throughput qualification supports H100 and H200")
    if (
        sum(
            [
                compare_output_gradient,
                compare_kda_wy,
                compare_graphed_muon,
                compare_intra,
                compare_consolidation,
                compare_scalar_delta,
                compare_fp8,
                bool(larger_microbatch),
                autotune,
                kda_save_intermediates,
            ]
        )
        > 1
    ):
        raise ValueError("Compare one optimization at a time")
    if hardware_kernel:
        from model_hybrid.hardware_counters import KERNELS

        if gpus != 1:
            raise ValueError(
                "Kernel replay is restricted to one GPU; use timeline profiling for DDP"
            )
        if hardware_kernel not in KERNELS:
            raise ValueError(f"Select one of {KERNELS}")
        if (
            profile
            or compare_output_gradient
            or compare_kda_wy
            or compare_graphed_muon
            or compare_intra
            or compare_consolidation
            or compare_scalar_delta
            or compare_fp8
            or larger_microbatch
            or autotune
            or kda_save_intermediates
        ):
            raise ValueError("Counter profiling must run separately from timing experiments")
        if steps < 7 or not run_id:
            raise ValueError("Counter profiling requires a run ID and at least seven steps")
    settings = load_config(config)
    if mtp_loss_fraction:
        settings["model"]["mtp_loss_fraction"] = mtp_loss_fraction
    rows = global_rows or (
        settings["global_rows"] if gpus == 8 else 8 * (microbatch or settings["microbatch"]) * gpus
    )
    from model_hybrid.benchmark import validate_window

    validate_window(settings, rows, steps, profile)
    deadline = new_deadline()
    run_id = run_id or f"bench-{time.time_ns()}"
    requested_gpu = "H100!" if gpu == "H100" and require_gpu_model else gpu
    options = {"gpu": f"{requested_gpu}:{gpus}"}
    call = benchmark.with_options(**options).spawn(
        settings,
        gpus,
        steps,
        microbatch,
        rows,
        document_tokens,
        deadline,
        profile,
        run_id,
        real_data,
        compare_output_gradient,
        larger_microbatch,
        autotune,
        kda_save_intermediates,
        gpu if require_gpu_model else "",
        hardware_kernel,
        compare_kda_wy,
        phase_timings,
        compare_graphed_muon,
        compare_intra,
        compare_consolidation,
        consolidation_cases,
        profile_all_ranks,
        profile_stacks,
        compare_scalar_delta,
        compare_fp8,
    )
    try:
        supervisor = enforce_deadline.spawn([call.object_id], deadline)
    except Exception:
        call.cancel()
        raise
    print(
        json.dumps(
            {
                "run_id": run_id,
                "call": call.object_id,
                "supervisor": supervisor.object_id,
                "deadline": deadline,
            }
        ),
        flush=True,
    )
    print(json.dumps(call.get(), indent=2))


@app.function(
    image=image, gpu="H100", cpu=8, memory=65536, timeout=3600, retries=0, volumes=VOLUMES
)
def heldout_loss(export, recipe, counts, deadline):
    """Held-out loss and bits per byte of an export on a pretraining recipe's sources."""
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    import torch
    from model_hybrid import generate
    from model_hybrid.train import evaluate, heldout_windows

    lm = generate.load_export(Path("/runs") / export, device="cuda")
    root = Path("/data") / recipe["name"]
    windows = {
        name: heldout_windows(recipe, root, name, lm.config.block_size, count)
        for name, count in counts.items()
    }
    torch.backends.cuda.matmul.allow_tf32 = True
    results = evaluate(lm, windows, torch.device("cuda"), lm.config.eos_id)
    return {
        name: {"loss": r["loss"], "bits_per_byte": r["bits_per_byte"]}
        for name, r in results.items()
    }


@app.local_entrypoint()
def heldout(export: str, recipe: str = "hybrid/recipes/hybrid-40b-v1.json"):
    """Pretraining held-out loss of `<run id>/export/<export>` (the 40B run's document counts)."""
    from workflows.deadline import MAX_LAUNCH_SECONDS

    counts = {"climbmix": 1000, "code": 300, "math": 300}
    result = heldout_loss.remote(
        export, load_config(recipe), counts, time.time() + MAX_LAUNCH_SECONDS
    )
    print(json.dumps({"export": export, "heldout": result}, indent=1))


@app.function(image=image, cpu=8, memory=65536, timeout=1800, retries=0, volumes=VOLUMES)
def export_latest(run_id):
    """bf16 trunk export (config.json, model.safetensors) of a run's latest checkpoint."""
    import torch
    from model_hybrid.train import RunConfig, export_config, model_config_of
    from safetensors.torch import save_file

    from trainer.loop import latest_checkpoint

    folder = Path("/runs") / run_id
    latest = latest_checkpoint(folder)
    state = torch.load(latest, map_location="cpu", weights_only=False)
    model_config = model_config_of(RunConfig(**state["config"]))
    out = folder / "export" / latest.stem
    out.mkdir(parents=True, exist_ok=True)
    weights = {
        k: v.to(torch.bfloat16).contiguous()
        for k, v in state["model"].items()
        if not k.startswith("mtp.")
    }
    save_file(weights, out / "model.safetensors")
    (out / "config.json").write_text(json.dumps(export_config(model_config), indent=2) + "\n")
    progress = {"step": state["step"], "tokens": state["tokens"]}
    (out / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    runs_volume.commit()
    return {"export": f"{run_id}/{latest.stem}", "progress": progress}


@app.local_entrypoint()
def export(run_id: str):
    """Export a run's latest checkpoint for evaluation (`<run id>/export/step-N/`)."""
    print(json.dumps(export_latest.remote(run_id)))


@app.local_entrypoint()
def publish(checkpoint: str, repo: str, public: bool = False, card: str = ""):
    """Upload one checkpoint (RUN/step-NNNNNNNNN.pt on the runs volume) to an HF model repo."""
    from workflows.deadline import MAX_LAUNCH_SECONDS

    text = Path(card).read_text() if card else ""
    result = export_checkpoint.remote(
        checkpoint, repo, not public, text, time.time() + MAX_LAUNCH_SECONDS
    )
    print(json.dumps(result, indent=2))


@app.function(
    image=counter_image,
    gpu="H100!",
    cpu=4,
    memory=16384,
    timeout=300,
    retries=0,
    volumes=VOLUMES,
)
def counter_probe(deadline, run_id):
    """Check hardware-counter access before allocating eight training GPUs."""
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed")
    from model_hybrid.hardware_counters import probe

    folder = Path("/runs/throughput") / run_id
    try:
        result = probe(folder)
        result["hardware"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
        result["profiling_driver_parameters"] = Path("/proc/driver/nvidia/params").read_text()
        (folder / "probe.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        runs_volume.commit()


@app.local_entrypoint()
def counters_probe(run_id: str, output: str):
    """Bounded single-H100 access check; no training or optimization experiment."""
    deadline = time.time() + 900
    call = counter_probe.spawn(deadline, run_id)
    try:
        supervisor = enforce_deadline.spawn([call.object_id], deadline)
    except Exception:
        call.cancel()
        raise
    record = {
        "run_id": run_id,
        "call": call.object_id,
        "supervisor": supervisor.object_id,
        "deadline": deadline,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".launch.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    result = call.get()
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
