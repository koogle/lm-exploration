"""Write a separate safetensors checkpoint with a recorded sparse MLP edit."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.model), Path(args.output)
    if output.exists():
        raise ValueError("Use a fresh output directory; preserve existing checkpoints")
    proposals = torch.load(args.proposals, map_location="cpu", weights_only=True)
    if len(set(args.layers)) != len(args.layers):
        raise ValueError("Duplicate layers would apply the edit twice")
    output.mkdir(parents=True)
    edits = []
    with safe_open(source / "model.safetensors", framework="pt") as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}  # noqa: SIM118 - safetensors API
        for index in args.layers:
            proposal = proposals[index]
            if proposal["layer"] != index:
                raise ValueError("Proposal indices must match MLP layer indices")
            key = f"model.layers.{index}.mlp.down_proj.weight"
            tensor = tensors[key].clone()
            columns, direction = proposal["columns"], proposal["direction"].float()
            old = tensor[:, columns].float()
            updated = old - args.strength * direction[:, None] * (direction @ old)[None, :]
            if not torch.isfinite(updated).all():
                raise ValueError("Non-finite edited weights")
            tensor[:, columns] = updated.to(tensor.dtype)
            tensors[key] = tensor
            edits.append(
                {
                    "tensor": key,
                    "columns": columns.tolist(),
                    "changed_scalars": int((tensor[:, columns].float() != old).sum()),
                    "delta_l2": float((tensor[:, columns].float() - old).norm()),
                }
            )
        save_file(tensors, output / "model.safetensors", metadata=handle.metadata())
        total = sum(t.numel() for t in tensors.values())
    for path in source.iterdir():
        if path.is_file() and path.suffix in (".json", ".jinja"):
            shutil.copy2(path, output / path.name)
    hashes = {}
    for label, path in (
        ("source", source / "model.safetensors"),
        ("candidate", output / "model.safetensors"),
    ):
        with path.open("rb") as handle:
            hashes[label] = hashlib.file_digest(handle, "sha256").hexdigest()
    manifest = {
        "status": "experimental candidate; consult held-out results before use",
        "source": str(source),
        "sha256": hashes,
        "strength": args.strength,
        "formula": "W[:,J] -= strength * d[:,None] * (d @ W[:,J])[None,:]",
        "edits": edits,
        "total_parameters": total,
        "changed_parameters": sum(e["changed_scalars"] for e in edits),
    }
    (output / "weight-edit.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "edits"}, indent=2))


if __name__ == "__main__":
    main()
