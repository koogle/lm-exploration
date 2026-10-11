# Why the current drafter remains weak

Source audit, 2026-10-11. This is a comparison and diagnostic interpretation,
not a new training experiment or an identified root cause.

The [completed 2B audit](../results/dflash-qwen35-2b-findings.md) measures
0.173 consecutive NEW matches on training versus 0.170 on validation at the
selected checkpoint. Training NLL is 6.594 versus validation 6.817. The model
is weak on both; more independent data alone is not an established remedy.
The earlier 0.5B tiny-set run memorized its training data, but that does not
establish that this 2B adaptation can fit or that either large-set fit converged.
Six fixed 2B examples repeatedly emit common tokens and punctuation. This is
consistent with learning weak token-frequency shortcuts; it does not prove
that the conditioning path is unused.

## Comparison with classic DFlash

Sources: [paper sections 4–5 and Appendix A.1](https://arxiv.org/html/2602.06036v1),
[pinned reference model code](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py),
and [our implementation](../src/latent_workspace/dflash.py).
Compare the classic classes, not DFlash2's candidate-selection additions.
The pinned tree exposes model/inference code, not a complete training runner;
training-recipe claims below come from the paper.

| Dimension | Classic DFlash standard setting | Our measured adaptation |
| --- | --- | --- |
| Training responses | Around 800k, Nemotron Post-Training V2 plus CodeAlpaca | 6144, mixed prose/code/translation |
| Draft layers / target-feature layers | Usually 5 / 5 | 2 / 2 |
| Attention | Qwen3-style, per-head query/key RMSNorm | Qwen2-style, no query/key normalization, biased Q/K/V |
| Response/context coverage | Maximum total sequence length 3072 | Responses capped at 64; source prompt budget 128; 2B chat formatting adds tokens |
| Anchors | 512 sampled per sequence | 8 per response visit, batch 3 |
| Training budget | Six epochs | 20000 updates, 3.36M NEW-label exposures |
| Block | Usually anchor plus 15 NEW tokens | Anchor plus 7 NEW tokens |
| Target | Qwen3 and LLaMA families studied in paper | Qwen2.5-0.5B; later Qwen3.5-2B hybrid |

Our target-generated labels, frozen target embeddings/readout, clean anchor plus
fully masked future positions, bidirectional isolated blocks, strictly pre-anchor
feature memory injected into every draft layer, and one-pass denoising follow
DFlash's core construction. Learning rate 6e-4, 4% warmup, cosine schedule,
gradient clipping at 1 and gamma 4 for block size 8 also follow its stated recipe.
Cached offline features are explicitly allowed by the paper. A diffusion drafter
does not require iterative denoising here: classic DFlash uses one parallel pass.

The corpus has about 130 times fewer responses. Anchors are correlated within
responses, and our short responses cannot provide 512 distinct full-block anchors;
do not present the anchor-count ratio as independent-data or compute equivalence.
The two-layer 2B draft has about 105M trainable parameters, versus 31M for 0.5B,
with the same label-exposure budget. Larger target weights remain frozen: changing
the target does not automatically give the randomly initialized draft those skills.

## What is known versus suspected

Confirmed: the current 2B adaptation has poor training fit; it is not a faithful
reproduction of DFlash. Missing query/key normalization, limited depth/features,
short teacher continuations, small batches and finite optimization exposure are
plausible contributors, not isolated causal findings. Qwen3.5's different tokenizer,
hybrid architecture and chat prompt format further confound the size comparison.
Matching a hybrid target does not inherently require a hybrid draft architecture.

Existing teacher spot checks find aligned labels, closely matching cached features,
and unchanged prefix features when future tokens change. They reduce the likelihood
of gross leakage/alignment faults but are not an exhaustive training-path audit.
Passing plumbing tests and observing a loss decrease do not establish useful block
prediction. Nor does low acceptance alone prove zero learning: it requires an
uninterrupted run of correct tokens starting at the first NEW position.

Follow-up direction: the user requests studies for the small frozen Qwen target,
so exact-2B fitting and additional size comparisons are deferred. The
[active small-target series](dflash-improvement-series.md) now measures a QK-norm
validation gain from 0.384 to 0.484 at matched exposure, but still weak full-training
fit (0.623). Zero-memory validation falls to 0.063. A separate larger-batch tiny-set
fit reaches 6.862 training matches and only 0.041 validation matches. These results
support testing substantially greater supervision on larger mixed-domain corpora;
they do not establish a useful drafter. Keep test sealed until the quality gate.
