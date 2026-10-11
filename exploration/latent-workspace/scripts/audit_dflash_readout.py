"""Analytical CE floor from the frozen vocabulary and current RMSNorm scale."""

import json
import math
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from latent_workspace.corpus import DOMAINS

torch.set_num_threads(4)
root = Path(__file__).resolve().parents[1]
state = torch.load(root / "artifacts/dflash-best.pt", map_location="cpu", weights_only=True)
data = torch.load(root / "artifacts/dflash-teacher.pt", map_location="cpu", weights_only=True)
path = hf_hub_download(
    state["config"]["model"],
    "model.safetensors",
    revision=state["config"]["model_revision"],
    local_files_only=True,
)
with safe_open(path, framework="pt") as f:
    weight = f.get_tensor("model.embed_tokens.weight").float()
    target_norm = f.get_tensor("model.norm.weight").float()
mean = weight.mean(0)
gamma = state["drafter"]["norm.weight"]
report = {
    "checkpoint_step": state["step"],
    "draft_norm_rms": float(gamma.square().mean().sqrt()),
    "target_norm_rms": float(target_norm.square().mean().sqrt()),
    "formula": "CE >= max(0, log(V) - sqrt(H) * ||(W_y - mean_v W_v) * gamma||_2). Jensen plus Cauchy-Schwarz; RMS-normalized pre-gamma vector has norm <= sqrt(H).",
    "scope": "A bound holding the checkpoint's norm weights fixed, not an irreducible data-entropy floor. Norm weights remain trainable.",
    "splits": {},
}
for split in data:
    by_domain = {}
    for domain in DOMAINS:
        examples = [e for e in data[split] if e["domain"] == domain]
        if split == "train":
            examples = examples[:4]
        labels = []
        for e in examples:
            first, last = e["prompt_length"], len(e["tokens"]) - 8
            anchors = torch.linspace(first, last, min(8, last - first + 1)).round().long().unique()
            labels.extend(e["tokens"][(anchors[:, None] + torch.arange(1, 8)).flatten()].tolist())
        centered = weight[labels] - mean
        bounds = {}
        for name, scale in (("trained_draft_norm", gamma), ("target_norm", target_norm)):
            lower = (
                math.log(len(weight)) - math.sqrt(weight.shape[1]) * (centered * scale).norm(dim=-1)
            ).clamp(min=0)
            bounds[name] = float(lower.mean())
        by_domain[domain] = {"labels": len(labels), "mean_ce_lower_bounds": bounds}
    report["splits"][split] = by_domain
(root / "results/dflash-readout-audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
