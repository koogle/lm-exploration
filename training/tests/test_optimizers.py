"""Role-declared Muon/AdamW grouping, for the hybrid and the subword GPT."""

import unittest

import torch
from model_hybrid import model

from trainer import optimizers as optim
from trainer.roles import declare


def small_model():
    config = model.HybridConfig(
        vocab_size=50,
        eos_id=2,
        block_size=8,
        n_layer=4,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        head_dim=8,
        n_kda_head=2,
        mlp_hidden=48,
    )
    return model.HybridLM(config)


@torch.no_grad()
def check_muon_batching(device="cpu", production_shapes=False):
    """Compare updates, momentum, missing gradients and state resume across batches."""
    import copy

    torch.manual_seed(947)
    shapes = (
        [(1536, 1536)] * 3 + [(128, 1536)] * 3 + [(1536, 128)] * 3 + [(4096, 1536)] * 2
        if production_shapes
        else [(9, 17)] * 5 + [(17, 9)] * 3 + [(8, 8)] * 4
    )
    reference = [torch.nn.Parameter(torch.randn(s, device=device)) for s in shapes]
    candidate = [torch.nn.Parameter(p.detach().clone()) for p in reference]

    def make(params, batch_size):
        return optim.Muon(
            [{"params": params[:6], "lr": 0.02}, {"params": params[6:], "lr": 0.01}],
            lr=0.02,
            momentum=0.95,
            weight_decay=0.1,
            batch_size=batch_size,
        )

    ref, batched = make(reference, 1), make(candidate, 3)
    largest = 0.0
    for step in range(4):
        for i, (p, q) in enumerate(zip(reference, candidate, strict=True)):
            grad = torch.randn_like(p) if i != step else None
            if i == len(reference) - 1:
                grad = torch.zeros_like(p)
            p.grad = grad
            q.grad = grad.clone() if grad is not None else None
        before = [p.detach().clone() for p in reference]
        other_before = [p.detach().clone() for p in candidate]
        ref.step()
        batched.step()
        for p, q, old, other_old in zip(reference, candidate, before, other_before, strict=True):
            if p.grad is None:
                torch.testing.assert_close(p, old, rtol=0, atol=0)
                torch.testing.assert_close(q, other_old, rtol=0, atol=0)
                continue
            delta, other_delta = p - old, q - other_old
            relative = float((delta - other_delta).norm() / delta.norm().clamp_min(1e-8))
            largest = max(largest, relative)
            if relative > 0.02:
                raise AssertionError(f"Batched Muon update relative L2 error {relative}")
            torch.testing.assert_close(
                ref.state[p]["momentum_buffer"],
                batched.state[q]["momentum_buffer"],
                rtol=0,
                atol=0,
            )
        if step == 1:
            restored = make(candidate, 3)
            restored.load_state_dict(copy.deepcopy(batched.state_dict()))
            batched = restored
    return {"max_relative_update_l2_error": largest, "steps": 4, "matrices": len(shapes)}


class OptimizerGroupingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)
        self.lm = small_model()

    def test_every_parameter_lands_in_exactly_one_group(self):
        groups = optim.parameter_groups(self.lm)
        names = [name for members in groups.values() for name, _ in members]
        self.assertEqual(sorted(names), sorted(n for n, _ in self.lm.named_parameters()))
        self.assertEqual(len(names), len(set(names)))
        ids = [id(p) for members in groups.values() for _, p in members]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(sum(1 for _ in self.lm.parameters()), len(ids))

        optimizer = optim.RoleOptimizer(self.lm)
        held = [id(p) for g in optimizer.param_groups for p in g["params"]]
        self.assertEqual(sorted(held), sorted(ids))

    def test_batched_muon_updates_and_resume(self):
        check_muon_batching()

    def test_roles(self):
        roles = {n: r for r, members in optim.parameter_groups(self.lm).items() for n, _ in members}
        self.assertEqual(roles["token_embedding.weight"], "adamw_decay")
        self.assertEqual(roles["blocks.0.mixer.b_proj.weight"], "adamw_decay")
        for name in (
            "blocks.0.mixer.A_log",
            "blocks.0.mixer.dt_bias",
            "blocks.0.mixer.q_conv",
            "blocks.0.mixer.o_norm_weight",
            "blocks.3.mixer.sink",
            "blocks.3.mixer.gate_bias",
            "blocks.3.mixer.q_norm.weight",
            "final_norm.weight",
            "mtp.hidden_norm.weight",
        ):
            self.assertEqual(roles[name], "adamw_no_decay", name)
        for name in (
            "blocks.0.mixer.q_proj.weight",
            "blocks.0.mixer.f_proj.0.weight",
            "blocks.0.mixer.g_proj.1.weight",
            "blocks.3.mixer.gate_proj.weight",
            "blocks.3.mixer.o_proj.weight",
            "blocks.3.mlp.down.weight",
            "mtp.fuse.weight",
            "mtp.block.mixer.k_proj.weight",
        ):
            self.assertEqual(roles[name], "muon", name)
        muon = [p for n, p in self.lm.named_parameters() if roles[n] == "muon"]
        self.assertTrue(all(p.ndim == 2 for p in muon))

    def test_undeclared_and_double_declared_parameters_are_rejected(self):
        self.lm.extra = torch.nn.Parameter(torch.zeros(3, 3))
        with self.assertRaisesRegex(ValueError, "extra"):
            optim.parameter_groups(self.lm)
        del self.lm.extra

        block = self.lm.blocks[0]
        original = type(block).optimizer
        try:
            type(block).optimizer = lambda b: (*original(b), *declare("muon", b.mixer.q_proj))
            with self.assertRaisesRegex(ValueError, "twice"):
                optim.parameter_groups(self.lm)
            type(block).optimizer = lambda b: declare("muon", b.mlp, b.mixer_norm, b.mlp_norm)
            with self.assertRaisesRegex(ValueError, "matrix"):
                optim.parameter_groups(self.lm)
        finally:
            type(block).optimizer = original

    def test_learning_rates_and_step(self):
        optimizer = optim.RoleOptimizer(self.lm)
        optim.set_learning_rate(optimizer, 0.5)
        self.assertAlmostEqual(optimizer.muon.param_groups[0]["lr"], 0.01)
        decay, no_decay = optimizer.adamw.param_groups
        self.assertAlmostEqual(decay["lr"], 1.5e-4)
        self.assertEqual((decay["weight_decay"], no_decay["weight_decay"]), (0.1, 0.0))
        tokens = torch.randint(0, 50, (2, 8))
        before = {n: p.detach().clone() for n, p in self.lm.named_parameters()}
        losses = self.lm(tokens, tokens, mtp_labels=tokens)
        (losses["loss"] + losses["mtp_loss"]).backward()
        optimizer.step()
        changed = [n for n, p in self.lm.named_parameters() if not torch.equal(p, before[n])]
        self.assertEqual(len(changed), len(before))
        restored = optim.RoleOptimizer(small_model())
        restored.load_state_dict(optimizer.state_dict())


if __name__ == "__main__":
    unittest.main()
