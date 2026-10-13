"""Training-only J-lens calibration and sparse original-residual-space projection.

The projection approximates the paper's nonnegative sparse cone using a frozen
candidate shortlist. It is not the historical top-k weighted token-vector readout.
"""

import hashlib
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from latent_workspace.backbone import FrozenBackbone
from latent_workspace.config import Config
from latent_workspace.dflash_data import load_teacher_cache, token_hash
from latent_workspace.jspace import fit_jacobians
from latent_workspace.train import atomic_save, write_json


def calibration_prefixes(examples, count=1000, width=128, seed=42):
    result = []
    domains = ("prose", "code", "translation")
    for i, domain in enumerate(domains):
        needed = count // 3 + (i < count % 3)
        eligible = sorted(
            (
                e
                for e in examples
                if e["domain"] == domain
                and e.get("split", "train") == "train"
                and len(e["tokens"]) >= width + 8
            ),
            key=lambda e: hashlib.sha256(f"{seed}:{e['group']}".encode()).hexdigest(),
        )
        seen = set()
        selected = []
        for e in eligible:
            source = e.get("source_group", e["group"])
            if source in seen:
                continue
            seen.add(source)
            selected.append(
                {
                    "domain": domain,
                    "group": e["group"],
                    "source_group": source,
                    "prefix": e["tokens"][:width].clone(),
                    "prefix_sha256": token_hash(e["tokens"][:width].tolist()),
                }
            )
            if len(selected) == needed:
                break
        if len(selected) != needed:
            raise ValueError(f"Insufficient distinct eligible {domain} training sources")
        result.extend(selected)
    return sorted(
        result, key=lambda e: hashlib.sha256(f"{seed}:order:{e['group']}".encode()).hexdigest()
    )


def j_dictionary(unembedding, final_norm_weight, jacobian):
    # RMSNorm's scalar denominator does not change vocabulary directions;
    # its learned per-channel scales must be included.
    atoms = (unembedding.float() * final_norm_weight.float()) @ jacobian.float()
    return atoms / atoms.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def sparse_component(hidden, dictionary, *, k=25, shortlist=256, refit_steps=32):
    """Nonnegative pursuit on shortlisted J-directions, followed by projected NNLS.

    Unit atoms preserve each cone. Shortlisting approximates the global support
    search; returned vector lives in the original residual basis, not final space.
    """
    if k < 1 or shortlist < k or shortlist > len(dictionary):
        raise ValueError("Invalid sparse support")
    h = hidden.float()
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if h.is_cuda else nullcontext()
    with ctx:
        scores = h @ dictionary.T
    ids = scores.float().topk(shortlist, dim=-1).indices
    candidates = dictionary[ids].float()
    gram = candidates @ candidates.transpose(-1, -2)
    dots = (candidates @ h.unsqueeze(-1)).squeeze(-1)
    coeff = torch.zeros_like(dots)
    support = torch.zeros_like(dots, dtype=torch.bool)
    for step in range(k):
        gradient = dots - (gram @ coeff.unsqueeze(-1)).squeeze(-1)
        choice = gradient.masked_fill(support, -torch.inf).argmax(-1, keepdim=True)
        support.scatter_(1, choice, True)
        for _ in range(2):
            gradient = dots - (gram @ coeff.unsqueeze(-1)).squeeze(-1)
            coeff = (coeff + gradient / (step + 1)).clamp_min(0) * support
    for _ in range(refit_steps):
        gradient = dots - (gram @ coeff.unsqueeze(-1)).squeeze(-1)
        coeff = (coeff + gradient / k).clamp_min(0) * support
    component = (candidates.transpose(-1, -2) @ coeff.unsqueeze(-1)).squeeze(-1)
    return component.to(hidden.dtype), ids, coeff


def calibrate(protocol, output, commit, resume=""):
    output = Path(output)
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    teacher = load_teacher_cache(Path(protocol["teacher_parent"]) / "teacher.pt")
    selected = calibration_prefixes(
        teacher["train"], protocol["calibration_prompts"], seed=protocol["seed"]
    )
    digest = hashlib.sha256(
        "".join(e["group"] + e["prefix_sha256"] for e in selected).encode()
    ).hexdigest()
    path = output / "jacobian-checkpoint.pt"
    if resume:
        old = Path(resume)
        previous = json.loads((old / "report.json").read_text())
        if previous["prefix_selection_sha256"] != digest:
            raise ValueError("Calibration prefix identities changed")
        path.write_bytes((old / "jacobian-checkpoint.pt").read_bytes())
    cfg = Config(
        model=protocol["model"],
        model_revision=protocol["model_revision"],
        layers=tuple(protocol["layers"]) + (24,),
        selected_layers=tuple(protocol["layers"]),
        jspace_prompts=protocol["fit_prompts"],
        jspace_dim_batch=protocol["dim_batch"],
    )
    backbone = FrozenBackbone(cfg)
    metadata = [{k: v for k, v in e.items() if k != "prefix"} for e in selected]
    write_json(
        output / "prefixes.json",
        {"selection_sha256": digest, "training_only": True, "records": metadata},
    )
    checkpoint_count = 0

    def save():
        nonlocal checkpoint_count
        checkpoint_count += 1
        if checkpoint_count % 16 == 0:
            commit()

    complete = False
    try:
        matrices = fit_jacobians(
            backbone,
            [e["prefix"] for e in selected],
            deadline=started + protocol["max_seconds"] - 90,
            checkpoint_path=path,
            on_checkpoint=save,
        )
        atomic_save(
            output / "jacobians.pt",
            {
                "matrices": matrices[:-1],
                "layers": protocol["layers"],
                "selection_sha256": digest,
                "prompts": protocol["fit_prompts"],
            },
        )
        complete = True
    except TimeoutError:
        matrices = None
    saved = torch.load(path, map_location="cpu", weights_only=True)
    report = {
        "completed": complete,
        "counts": saved["counts"],
        "goal": protocol["fit_prompts"],
        "protocol": protocol,
        "prefix_selection_sha256": digest,
        "elapsed_seconds": time.monotonic() - started,
        "target_gradients_absent": all(p.grad is None for p in backbone.model.parameters()),
        "test_status": "Previously consumed drafting test is not opened; calibration uses training prefixes only",
    }
    if complete and protocol.get("profile_projection"):
        ids = selected[0]["prefix"][None].cuda()
        with torch.no_grad():
            states = backbone.model.model(
                ids, output_hidden_states=True, use_cache=False
            ).hidden_states
        diagnostics = []
        for layer, j in zip(protocol["layers"], matrices[:-1]):
            dictionary = j_dictionary(
                backbone.model.get_output_embeddings().weight,
                backbone.model.model.norm.weight,
                j.cuda(),
            )
            h = states[layer][0, -16:]
            component, _indices, coeff = sparse_component(h, dictionary, k=25, shortlist=256)
            diagnostics.append(
                {
                    "layer": layer,
                    "finite": bool(torch.isfinite(component).all()),
                    "max_active": int((coeff > 1e-8).sum(-1).max()),
                    "negative_coefficients": bool((coeff < 0).any()),
                    "mean_relative_reconstruction_error": float(
                        (h.float() - component.float())
                        .norm(dim=-1)
                        .div(h.float().norm(dim=-1).clamp_min(1e-12))
                        .mean()
                    ),
                }
            )
            del dictionary
        report["projection_profile"] = diagnostics
        report["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()
    write_json(output / "report.json", report)
    commit()
    return report
