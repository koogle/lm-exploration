#!/usr/bin/env python3
"""Local UI for train.py and the 100M architecture. From training/: uv run lm-ui."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from model_100m.model import ModelConfig

ROOT = Path(__file__).resolve().parent
RUN_ID = re.compile(r"r-[0-9a-f]{32}\Z")
DATASET_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
READ_LIMIT = 256 * 1024
DATA_LIMIT = 64 * 1024 * 1024
STOP_TIMEOUT = 30
DEFAULTS = {
    "steps": 10,
    "batch_size": 1,
    "block_size": ModelConfig.block_size,
    "n_layer": ModelConfig.n_layer,
    "n_head": ModelConfig.n_head,
    "n_embd": ModelConfig.n_embd,
    "lr": 0.0003,
    "seed": 42,
    "device": "cpu",
    "eval_interval": 5,
    "log_interval": 1,
    "eval_batches": 2,
    "checkpoint_interval": 5,
}
BOUNDS = {
    "steps": (1, 10000),
    "batch_size": (1, 8),
    "block_size": (8, 256),
    "n_layer": (1, ModelConfig.n_layer),
    "n_head": (1, ModelConfig.n_head),
    "n_embd": (16, ModelConfig.n_embd),
    "seed": (0, 2**31 - 1),
    "eval_interval": (1, 10000),
    "log_interval": (1, 10000),
    "eval_batches": (1, 32),
    "checkpoint_interval": (1, 10000),
}


class APIError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message


def now():
    return datetime.now(UTC).isoformat()


def confined(base, relative):
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()) or path == base.resolve():
        raise ValueError("Path must remain inside its owning directory")
    return path


def reject_nonfinite(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def read_json(path, limit=READ_LIMIT):
    with path.open("rb") as handle:
        content = handle.read(limit + 1)
    if len(content) > limit:
        raise ValueError("JSON file exceeds size limit")
    return json.loads(content, parse_constant=reject_nonfinite)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_datasets(root):
    base = root / "data"
    manifest = read_json(confined(base, "datasets.json"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("datasets"), list):
        raise ValueError("datasets.json must contain a datasets array")
    datasets = {}
    for entry in manifest["datasets"]:
        if not isinstance(entry, dict):
            raise ValueError("Each dataset must be an object")
        for key in ("id", "name", "path", "description", "source", "license_notes"):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                raise ValueError(f"Dataset requires nonempty {key}")
        if not DATASET_ID.fullmatch(entry["id"]) or entry["id"] in datasets:
            raise ValueError("Dataset IDs must be unique lowercase slugs")
        if (
            not isinstance(entry.get("languages"), list)
            or not entry["languages"]
            or not all(isinstance(item, str) and item.strip() for item in entry["languages"])
        ):
            raise ValueError("Dataset requires a nonempty languages array")
        if type(entry.get("training_eligible")) is not bool:
            raise ValueError("Dataset requires boolean training_eligible")
        if Path(entry["path"]).is_absolute() or ".." in Path(entry["path"]).parts:
            raise ValueError("Dataset path must be relative to data/, without parent traversal")
        path = confined(base, entry["path"])
        if not path.is_file():
            raise ValueError(f"Dataset file missing: {entry['id']}")
        datasets[entry["id"]] = entry.copy()
    return datasets


def dataset_snapshot(root, entry):
    path = confined(root / "data", entry["path"])
    # Read and validate exactly the bytes fingerprinted in the UI metadata.
    with path.open("rb") as handle:
        content = handle.read(DATA_LIMIT + 1)
    if not content or len(content) > DATA_LIMIT:
        raise APIError(400, "Local demo datasets must be nonempty and at most 64 MiB")
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        raise APIError(400, "Dataset must be UTF-8 text") from None
    return path, dict(entry, sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content))


def validate_settings(payload, datasets):
    if not isinstance(payload, dict):
        raise APIError(400, "Request must be a JSON object")
    if set(payload) - (set(DEFAULTS) | {"dataset_id"}):
        raise APIError(400, "Unknown request fields")
    dataset_id = payload.get("dataset_id")
    if not isinstance(dataset_id, str) or dataset_id not in datasets:
        raise APIError(400, "Choose a registered dataset_id")
    if not datasets[dataset_id]["training_eligible"]:
        raise APIError(400, "Dataset is not eligible for training")
    settings = DEFAULTS | {key: value for key, value in payload.items() if key in DEFAULTS}
    for key, (low, high) in BOUNDS.items():
        if type(settings[key]) is not int or not low <= settings[key] <= high:
            raise APIError(400, f"{key} must be an integer in [{low}, {high}]")
    lr = settings["lr"]
    if type(lr) not in (int, float) or not 0.000001 <= lr <= 0.01 or not math.isfinite(lr):
        raise APIError(400, "lr must be finite and in [0.000001, 0.01]")
    if settings["device"] not in ("cpu", "mps", "cuda"):
        raise APIError(400, "device must be cpu, mps or cuda")
    if settings["n_embd"] % settings["n_head"]:
        raise APIError(400, "Model width must be divisible by heads")
    if settings["batch_size"] * settings["block_size"] > 8192:
        raise APIError(400, "batch_size × block_size must not exceed 8192")
    if settings["n_layer"] * settings["n_embd"] ** 2 > ModelConfig.n_layer * ModelConfig.n_embd**2:
        raise APIError(400, "Model exceeds the configured 100M size class")
    if settings["steps"] * settings["batch_size"] * settings["block_size"] > 10000000:
        raise APIError(400, "Local run bound: at most 10 million training token exposures")
    return dataset_id, settings


def tail(path, limit=READ_LIMIT):
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            content = handle.read(limit)
        return content.decode("utf-8", errors="replace"), size > limit
    except FileNotFoundError:
        return "", False


def metrics_tail(path):
    text, truncated = tail(path)
    lines = text.splitlines(keepends=True)
    if truncated and lines:
        lines = lines[1:]  # May begin halfway through a JSON record.
    records = []
    for line in lines:
        if not line.endswith("\n"):
            continue  # The trainer may currently be writing this final record.
        try:
            item = json.loads(line)
            if not isinstance(item, dict) or type(item.get("step")) is not int or item["step"] < 0:
                continue
            record = {"step": item["step"]}
            for key in (
                "train_loss",
                "val_loss",
                "tokens_seen",
                "tokens_per_second",
                "elapsed_seconds",
            ):
                value = item.get(key)
                if type(value) in (int, float) and math.isfinite(value):
                    record[key] = value
            records.append(record)
        except (ValueError, OverflowError):
            continue
    return records[-2000:], truncated or len(records) > 2000


class RunManager:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.datasets = load_datasets(self.root)
        self.runs = self.root / "runs"
        self.runs.mkdir(exist_ok=True)
        if self.runs.is_symlink():
            raise ValueError("runs/ must not be a symlink")
        self.instance_lock = (self.runs / ".server.lock").open("a")
        try:
            fcntl.flock(self.instance_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.instance_lock.close()
            raise ValueError("Another local UI already owns this runs/ directory") from None
        self.lock = threading.RLock()
        self.active = None
        self.closing = False
        # Never trust a persisted PID, nor signal a process we did not spawn.
        for directory in self.runs.iterdir():
            if (
                RUN_ID.fullmatch(directory.name)
                and directory.is_dir()
                and not directory.is_symlink()
            ):
                try:
                    record = self.record(directory.name)
                    if record["status"] in ("starting", "running", "stopping"):
                        record.update(
                            status="interrupted",
                            ended_at=now(),
                            note="Previous server exited; process ownership and outcome are unknown.",
                        )
                        self.save(record)
                except (APIError, ValueError, OSError, KeyError):
                    continue

    def directory(self, run_id):
        if not RUN_ID.fullmatch(run_id):
            raise APIError(404, "Run not found")
        if (self.runs / run_id).is_symlink():
            raise APIError(404, "Run not found")
        path = confined(self.runs, run_id)
        if not path.is_dir():
            raise APIError(404, "Run not found")
        return path

    def record(self, run_id):
        try:
            value = read_json(confined(self.directory(run_id), "ui.json"))
            if not isinstance(value, dict) or value.get("id") != run_id:
                raise ValueError("Invalid run metadata")
            return value
        except (ValueError, OSError):
            raise APIError(404, "Run metadata missing or invalid") from None

    def save(self, record):
        write_json(confined(self.directory(record["id"]), "ui.json"), record)

    def detail(self, run_id, full=True):
        with self.lock:
            record = self.record(run_id)
            directory = self.directory(run_id)
            metrics, truncated = metrics_tail(confined(directory, "metrics.jsonl"))
            latest = metrics[-1] if metrics else {}
            record["latest"] = latest
            for key in ("train_loss", "val_loss"):
                record[key] = next((item[key] for item in reversed(metrics) if key in item), None)
            record["checkpoint_available"] = confined(directory, "checkpoint.pt").is_file()
            try:
                config = read_json(confined(directory, "config.json"))
                count = config.get("parameter_count") if isinstance(config, dict) else None
                record["parameter_count"] = count if type(count) is int and count > 0 else None
            except (FileNotFoundError, ValueError):
                record["parameter_count"] = None
            record["metrics_truncated"] = truncated
            if full:
                record["metrics"] = metrics
                record["log"], record["log_truncated"] = tail(
                    confined(directory, "trainer.log"), 32768
                )
                record["sample"], record["sample_truncated"] = tail(
                    confined(directory, "sample.txt"), 16384
                )
            return record

    def list_runs(self):
        with self.lock:
            records = []
            for directory in self.runs.iterdir():
                if RUN_ID.fullmatch(directory.name) and not directory.is_symlink():
                    try:
                        records.append(self.detail(directory.name, full=False))
                    except (APIError, ValueError, OSError):
                        continue
            return sorted(records, key=lambda item: item.get("created_at", ""), reverse=True)

    def start(self, payload):
        dataset_id, settings = validate_settings(payload, self.datasets)
        with self.lock:
            if self.closing or self.active:
                raise APIError(409, "A local job is already active or the server is closing")
            trainer = self.root / "train.py"
            if not trainer.is_file():
                raise APIError(
                    503, "train.py is missing; restore the shared trainer before starting"
                )
            if importlib.util.find_spec("torch") is None:
                raise APIError(503, "PyTorch is missing; run uv sync --locked, then uv run lm-ui")
            try:
                data_path, snapshot = dataset_snapshot(self.root, self.datasets[dataset_id])
            except (OSError, ValueError) as error:
                raise APIError(400, f"Dataset unavailable: {error}") from None
            run_id = "r-" + uuid.uuid4().hex
            directory = self.runs / run_id
            directory.mkdir()
            argv = [
                sys.executable,
                "-u",
                str(trainer),
                "train",
                "--data",
                str(data_path),
                "--out-dir",
                str(directory),
            ]
            for key, value in settings.items():
                argv.extend(["--" + key.replace("_", "-"), str(value)])
            record = {
                "id": run_id,
                "model": "100m",
                "status": "starting",
                "created_at": now(),
                "settings": settings,
                "dataset": snapshot,
                "command": argv,
                "stop_requested": False,
            }
            self.save(record)
            try:
                with (directory / "trainer.log").open("ab") as log:
                    process = subprocess.Popen(
                        argv,
                        cwd=self.root,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
            except OSError as error:
                record.update(
                    status="failed", ended_at=now(), note=f"Unable to launch trainer: {error}"
                )
                self.save(record)
                raise APIError(503, record["note"]) from None
            record.update(status="running", pid=process.pid)
            self.save(record)
            job = {"id": run_id, "process": process, "stop_time": None, "forced": False}
            self.active = job
            thread = threading.Thread(target=self.watch, args=(job,), daemon=True)
            job["thread"] = thread
            thread.start()
            return self.detail(run_id)

    def watch(self, job):
        process = job["process"]
        while process.poll() is None:
            with self.lock:
                if (
                    job["stop_time"] is not None
                    and time.monotonic() - job["stop_time"] >= STOP_TIMEOUT
                ):
                    job["forced"] = True
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
            time.sleep(0.1)
        with self.lock:
            record = self.record(job["id"])
            code = process.wait()
            if job["forced"]:
                status, note = (
                    "stopped",
                    f"Force-stopped after {STOP_TIMEOUT} seconds; latest work may not be checkpointed.",
                )
            elif record["stop_requested"]:
                status = "stopped" if code in (0, -signal.SIGTERM, -signal.SIGINT) else "failed"
                note = "Stop requested; inspect the checkpoint and log before resuming."
            else:
                artifacts = ("config.json", "metrics.jsonl", "checkpoint.pt", "sample.txt")
                complete = code == 0 and all(
                    (self.directory(job["id"]) / name).is_file() for name in artifacts
                )
                status = "completed" if complete else "failed"
                note = (
                    "Trainer exited normally."
                    if complete
                    else "Trainer exited without the required artifacts."
                    if code == 0
                    else f"Trainer exited with code {code}."
                )
            record.update(status=status, ended_at=now(), exit_code=code, note=note)
            self.save(record)
            if self.active is job:
                self.active = None

    def stop(self, run_id):
        with self.lock:
            self.record(run_id)
            job = self.active
            if not job or job["id"] != run_id:
                raise APIError(409, "Run is not owned and active in this server")
            if job["process"].poll() is not None:
                raise APIError(409, "Trainer has exited; its final status is being recorded")
            if job["stop_time"] is None:
                record = self.record(run_id)
                record.update(status="stopping", stop_requested=True)
                self.save(record)
                job["stop_time"] = time.monotonic()
                try:
                    job["process"].send_signal(signal.SIGTERM)
                except ProcessLookupError:
                    pass
            return self.detail(run_id)

    def close(self):
        with self.lock:
            self.closing = True
            job = self.active
            if job:
                try:
                    self.stop(job["id"])
                except APIError:
                    pass
        if job:
            job["thread"].join(STOP_TIMEOUT + 2)
            if job["thread"].is_alive():
                job["process"].kill()
                job["thread"].join(2)
        self.instance_lock.close()


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalTraining/1"

    def send(self, status, content, content_type="application/json; charset=utf-8"):
        body = (
            json.dumps(content, allow_nan=False).encode()
            if content_type.startswith("application/json")
            else content
        )
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def check_local(self, changing=False):
        hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        host = self.headers.get("Host")
        if host not in hosts or len(self.headers.get_all("Host", [])) != 1:
            raise APIError(403, "Host must be this localhost server")
        origins = self.headers.get_all("Origin", [])
        if origins and (len(origins) != 1 or origins[0] != "http://" + host):
            raise APIError(403, "Origin must match this localhost server")
        if changing and self.headers.get("Sec-Fetch-Site") not in (None, "same-origin", "none"):
            raise APIError(403, "Cross-site requests are not allowed")

    def payload(self):
        if self.headers.get("Transfer-Encoding"):
            raise APIError(400, "Transfer-Encoding is unsupported")
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or not lengths[0].isdigit():
            raise APIError(400, "A single Content-Length is required")
        length = int(lengths[0])
        if not 0 < length <= 8192:
            raise APIError(413, "JSON body must be between 1 and 8192 bytes")
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            raise APIError(415, "Content-Type must be application/json")
        try:
            content = self.rfile.read(length)
            if len(content) != length:
                raise ValueError("Incomplete request body")
            return json.loads(content, parse_constant=reject_nonfinite)
        except (ValueError, UnicodeDecodeError):
            raise APIError(400, "Invalid JSON body") from None

    def dispatch(self, changing=False):
        # Error responses close the connection, so rejected bodies are never reused.
        self.close_connection = True
        self.connection.settimeout(5)
        try:
            self.check_local(changing)
            parsed = urlsplit(self.path)
            if parsed.query or parsed.fragment or parsed.scheme or parsed.netloc:
                raise APIError(404, "Route not found")
            path = parsed.path
            manager = self.server.manager
            if not changing:
                if path == "/":
                    return self.send(
                        200, (manager.root / "ui.html").read_bytes(), "text/html; charset=utf-8"
                    )
                if path == "/api/datasets":
                    return self.send(
                        200,
                        {
                            "datasets": list(manager.datasets.values()),
                            "defaults": DEFAULTS,
                            "bounds": BOUNDS,
                        },
                    )
                if path == "/api/runs":
                    return self.send(200, {"runs": manager.list_runs()})
                match = re.fullmatch(r"/api/runs/(r-[0-9a-f]{32})", path)
                if match:
                    return self.send(200, manager.detail(match[1]))
            else:
                if path == "/api/runs":
                    return self.send(201, manager.start(self.payload()))
                match = re.fullmatch(r"/api/runs/(r-[0-9a-f]{32})/stop", path)
                if match:
                    if self.payload() != {}:
                        raise APIError(400, "Stop body must be an empty JSON object")
                    return self.send(200, manager.stop(match[1]))
            raise APIError(404, "Route not found")
        except APIError as error:
            self.send(error.status, {"error": error.message})
        except (OSError, ValueError, TypeError, KeyError) as error:
            print(f"Request error: {error}", file=sys.stderr)
            self.send(500, {"error": "Unable to read local run artifacts; check the server log"})

    def do_GET(self):
        self.dispatch()

    def do_POST(self):
        self.dispatch(changing=True)

    def do_OPTIONS(self):
        self.send(405, {"error": "CORS is not enabled"})


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5174)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be in [1, 65535]")
    try:
        server = LocalServer(("127.0.0.1", args.port), Handler)
        manager = RunManager()
    except (OSError, ValueError) as error:
        parser.exit(1, f"Cannot start local UI: {error}\n")
    server.manager = manager
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Local training UI: http://127.0.0.1:{args.port}", flush=True)
    try:
        stop.wait()
    finally:
        manager.close()
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    main()
