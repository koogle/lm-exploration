import importlib
from pathlib import Path

import pytest
import torch


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("jspace_next")


def test_sparse_component_is_nonnegative_original_basis_and_support_bounded(module):
    atoms = torch.cat((torch.eye(4), -torch.eye(4)))
    h = torch.tensor([[1.0, -2.0, 0.0, 0.5], [0.0, 0.0, 0.0, 0.0]])
    projected, ids, coeff = module.sparse_component(h, atoms, k=2, shortlist=8)
    torch.testing.assert_close(projected[0], torch.tensor([1.0, -2.0, 0.0, 0.0]), atol=1e-4, rtol=0)
    assert (coeff >= 0).all() and ((coeff > 1e-8).sum(-1) <= 2).all()
    assert torch.equal(projected[1], torch.zeros(4))
    torch.testing.assert_close(
        projected, (atoms[ids].transpose(-1, -2) @ coeff.unsqueeze(-1)).squeeze(-1)
    )
    assert ((h - projected).norm(dim=-1) <= h.norm(dim=-1)).all()


def test_dictionary_includes_final_rmsnorm_channel_scales(module):
    u = torch.tensor([[1.0, 2.0], [2.0, -1.0]])
    j = torch.tensor([[1.0, 0.0], [0.5, 2.0]])
    gamma = torch.tensor([3.0, 0.5])
    d = module.j_dictionary(u, gamma, j)
    h = torch.tensor([2.0, -1.0])
    directions = (u * gamma) @ j
    # Equivalence to transporting h then applying the scaled vocabulary readout.
    torch.testing.assert_close(d @ h, ((u * gamma) @ (j @ h)) / directions.norm(dim=-1))


def test_calibration_uses_distinct_training_sources_and_excludes_future_tokens(module):
    rows = [
        {"domain": d, "split": "train", "group": f"{d}-{i}", "tokens": torch.arange(20) + i * 100}
        for d in ("prose", "code", "translation")
        for i in range(4)
    ]
    rows.append(
        {"domain": "prose", "split": "validation", "group": "held-out", "tokens": torch.arange(20)}
    )
    a = module.calibration_prefixes(rows, count=6, width=8)
    b = module.calibration_prefixes(list(reversed(rows)), count=6, width=8)
    assert [e["group"] for e in a] == [e["group"] for e in b]
    assert "held-out" not in {e["group"] for e in a}
    assert all(len(e["prefix"]) == 8 for e in a)
    assert all(sum(e["domain"] == d for e in a) == 2 for d in ("prose", "code", "translation"))
