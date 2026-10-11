# Latent workspace block generation

Current priority: build a **DFlash-style speculative drafter for frozen Qwen2.5-0.5B**
that predicts useful multi-token blocks. Train on Qwen-generated continuations,
condition on a known anchor, and verify proposals with Qwen. The original J-space,
workspace and layer-selection experiments are deferred until this baseline works.
No backbone pretraining or collaborative proposal module.

See the [DFlash baseline protocol](docs/dflash-baseline.md) for the paper/code
comparison, data contract and predeclared evaluation. The implementation is a compact
Qwen2 adaptation, not a full paper reproduction. The [completed 21000-update fit](results/dflash-findings.md)
memorizes the 380-response training corpus but fails to generalize: best validation
acceptance is only 0.284 new tokens / 7. FP32 verified output matches on three checked
prompts, but generation is slower than Qwen alone. Broader, better-sampled training
data is the next priority; no useful fast drafter is established.

![DFlash-style fitting and validation curves](results/dflash-fit-01.png)

A [controlled 768-versus-6144 response comparison](docs/dflash-data-scaling.md)
is now running with improved source sampling, identical initial weights and a shared
192-response validation cache. The reference-paper generalization evidence and
the differences from our small-Qwen adaptation are recorded in that protocol.

## Development

Run these commands from `exploration/latent-workspace/` in LM Exploration. This
folder retains its own uv project and lockfile. Before launching GPU work from a
new checkout, restore the existing local budget ledger as described in
[operations](docs/operations.md); it is not stored in Git.

```bash
uv sync --extra dev --extra modal
uv run pytest
uv run ruff check .
uv run modal run modal_app.py --config configs/smoke.json --run-id smoke-l40s
```

Modal uses its configured account, a dedicated app and volumes. Read
[operations](docs/operations.md) before launching; checkpoints and measured
results are persisted. A smoke test establishes plumbing, not model quality.

## Research references

- Chen et al., [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036),
  [official code](https://github.com/z-lab/dflash). Current baseline reference;
  stochastic verification and the paper's speedup results are not reproduced.
- Gurnee et al. (2026), [Verbalizable Representations Form a Global Workspace in
  Language Models](https://transformer-circuits.pub/2026/workspace/index.html),
  [arXiv:2607.15495](https://arxiv.org/abs/2607.15495).
- Anthropic, [Jacobian lens reference implementation](https://github.com/anthropics/jacobian-lens),
  reference commit `581d398613e5602a5af361e1c34d3a92ea82ba8e` (Apache-2.0).
- [Block Diffusion: Interpolating Between Autoregressive and Diffusion Language
  Models](https://arxiv.org/abs/2503.09573) (BD3-LM): a research baseline to
  reproduce later, **not** an implementation or quality claim for this code.
- [Qwen2.5-0.5B model card](https://huggingface.co/Qwen/Qwen2.5-0.5B).

A [BibTeX bibliography](references/papers.bib) and [measured run evidence](results/README.md)
are retained in the repository.

See [experiment protocol](docs/experiment.md), [knowledge index](docs/knowledge/INDEX.md),
and [open work](OPEN_WORK.md). All dependencies and files belong to this project;
it has no connection to the Age of Agents game.

## Completed studies

The initial WikiText pilot did not produce useful generation. The subsequent
[mixed-domain learning-curve study](docs/data-budget.md) separated corpus size
from optimization exposure. Its original pool had an exact-window overlap;
clean follow-up runs removed it and preserved held-out data identities.

The [first measured data-budget curves](results/data-study-01.md) are available:
more data improved validation loss, but sample generations remain degenerate.
No sufficient training budget or useful drafter quality is established yet.


The [data-size curves](results/data-scaling.md) established a 196,608-window
mixed-domain diagnostic pool. The [trained conditioning study](results/conditioning-findings.md)
is now complete: 96 layer probes, nine Qwen2.5-0.5B decoder conditions, two
Qwen3-0.6B conditions, and a frozen-checkpoint test evaluation of all 11 heads.
All-layer residual conditioning helps, but compressed J-lens conditioning and
denoising do not improve on their respective controls here. Generation remains
poor; there is no demonstrated quality parity or speedup.

![Conditioning learning curves](results/conditioning-comparison.png)

The [test report](results/conditioning-test.md) includes all domain measurements,
paired uncertainty and actual continuations. Test was withheld during training
and selection, then evaluated once; it is no longer an untouched holdout for
future adaptive work.

## Historical conditioning checkpoint

The all-layer Qwen2.5 diffusion head is saved on Modal and downloaded locally at
`artifacts/conditioning-all-best.pt`. Its checksum matches the frozen test head.
It is a research checkpoint with poor generation quality.

```bash
uv run modal volume get latent-workspace-qwen-runs \
  conditioning-all-01/size-196608-best.pt artifacts/conditioning-all-best.pt
uv run latent-workspace generate --checkpoint artifacts/conditioning-all-best.pt \
  --prompt 'Once upon a time' --tokens 24
```

Checkpoint files reference pinned pretrained Qwen weights; they do not bundle
those weights. See [current limitations and remaining work](OPEN_WORK.md).

## Reading the historical window-training curves

One **window** is a 128-token prefix and the following four target tokens. The
four-token generation block stays fixed. Curve labels such as 500/1000/2000/4000
count optimizer updates, not block length or learning-rate changes. At batch 128,
4000 updates sample 512000 windows with replacement—about 2.6 sampled passes over
196608 distinct windows. This is a fixed screening budget, not proof of convergence.

The [small-data follow-up](docs/small-data-controls.md) measures the entire training
subset alongside validation, with denser checkpoints and 96/768/3072-window sets.
It tests all-layer conditioning, whole-layer dropout, zero-content attention, and
both together. Learning rate remains 0.0003. Existing validation is reused for
exploration; the earlier test has already been evaluated and is not reused here.

[Completed small-data curves and control results](results/small-layer-controls.md):
12 fits show near-complete small-set memorization followed by validation overfitting.
Neither 25% layer dropout nor zero-content attention gives a clear validation gain
in this one-seed diagnostic. This does not establish a sufficient final data size.
