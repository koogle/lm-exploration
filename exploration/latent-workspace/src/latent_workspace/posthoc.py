"""Read-only validation of saved study heads, including known-first-token drafting."""

import hashlib
import json
from pathlib import Path

import torch

from .backbone import FrozenBackbone
from .config import Config
from .evaluate import evaluate, generate
from .models import BlockHead
from .train import write_json


def evaluate_study(request, output):
    if request.study_eval_parents:
        from dataclasses import replace

        results = {}
        for parent in request.study_eval_parents:
            results[parent] = evaluate_study(
                replace(request, study_parent=parent, study_eval_parents=()),
                Path(output) / Path(parent).name,
            )
        report = {
            "evaluation_split": "test" if request.evaluate_test else "validation",
            "studies": results,
        }
        Path(output).mkdir(parents=True, exist_ok=True)
        write_json(Path(output) / "posthoc-suite.json", report)
        return report
    source = Path(request.study_parent)
    study = json.loads((source / "study.json").read_text())
    if not study.get("completed"):
        raise ValueError("posthoc evaluation requires a completed study")
    source_digest = hashlib.sha256((source / "study.json").read_bytes()).hexdigest()
    if request.evaluate_test and request.study_eval_hashes.get(str(source)) != source_digest:
        raise ValueError("test evaluation requires a matching frozen source report hash")
    cfg = Config(**study["config"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = FrozenBackbone(cfg, device)
    saved = torch.load(source / "corpus.pt", weights_only=True)
    split = "test" if request.evaluate_test else "validation"
    validation = saved["tensors"][split]
    report = {
        "source_run": str(source),
        "evaluation_split": split,
        "test_status": "evaluated; checkpoints selected previously on validation"
        if request.evaluate_test
        else "sealed",
        "source_report_sha256": source_digest,
        "heads": {},
    }
    for size in study["curves"]:
        checkpoint_path = source / f"size-{size}.pt"
        best_path = source / f"size-{size}-best.pt"
        if best_path.exists():
            checkpoint_path = best_path
        state = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        backbone.jacobians = state.get("jacobians")
        head = BlockHead(backbone.hidden_size, cfg, state["variant"]).to(device)
        head.load_state_dict(state["head"])
        result = {
            "step": state["step"],
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "domains": {},
            "sample_prefixes": {},
            "samples": {},
            "anchored_samples": {},
        }
        for domain, (a, b) in saved["metadata"]["domain_ranges"][split].items():
            result["domains"][domain] = evaluate(
                head, backbone, validation[a:b], cfg, device, anchor_diagnostics=True
            )
            prefix = validation[a : a + 1, : cfg.context_length].to(device)
            result["sample_prefixes"][domain] = backbone.tokenizer.decode(
                prefix[0][prefix[0] >= 0].tolist()
            )
            result["samples"][domain] = backbone.tokenizer.decode(
                generate(head, backbone, prefix, max_new_tokens=24)
            )
            if head.kind == "diffusion":
                result["anchored_samples"][domain] = backbone.tokenizer.decode(
                    generate(head, backbone, prefix, max_new_tokens=24, anchor_first=True)
                )
        with torch.no_grad():
            embeddings = backbone.embed(validation[:, cfg.context_length :].to(device)).float()
            projected = head.token_project(head.token_norm(embeddings))
            result["input_scales"] = {
                "embedding_rms": float(embeddings.square().mean().sqrt()),
                "projected_token_variation_rms": float(
                    (projected - projected.mean((0, 1), keepdim=True)).square().mean().sqrt()
                ),
                "time_embedding_rms_at_half_masked": float(
                    head.time(torch.tensor([[0.5]], device=device)).square().mean().sqrt()
                ),
                "position_embedding_rms": float(head.position.square().mean().sqrt()),
            }
        report["heads"][size] = result
    Path(output).mkdir(parents=True, exist_ok=True)
    write_json(Path(output) / "posthoc.json", report)
    return report
