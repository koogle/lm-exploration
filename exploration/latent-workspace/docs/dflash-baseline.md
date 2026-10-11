# Drafting baseline before representation experiments

User steering, 2026-10-11: defer hidden-state generation/J-space/workspace/layer
ablations until a useful diffusion drafter exists. Previous results remain evidence,
but do not establish a suitable drafter. The next experiment is one DFlash-style
Qwen2.5-0.5B baseline, not another architecture sweep.

## Reference and interpretation

Read Chen, Liang and Liu, [DFlash](https://arxiv.org/html/2602.06036v1), especially
§4.2–4.3, §5 and Appendix A.1; inspect the
[official implementation](https://github.com/z-lab/dflash/tree/07ebd93db9f472af339b644bb70221ad8428328a).
The repository now also includes DFlash 2; this baseline follows classic DFlash,
not its later candidate selector/convolutions. MIT attribution is retained in
[references/dflash-LICENSE](../references/dflash-LICENSE).

The paper trains on target-generated responses (~800k samples in the main mixture),
random response anchors with all subsequent block positions masked, frozen shared
embeddings/output head, fixed multi-layer target features injected as K/V at every
draft layer, and one parallel denoising pass. It uses exponentially decayed CE,
AdamW at 6e-4, cosine decay, 4% warmup and six epochs; gamma=4 for block size 8.
Its training implementation packs independent blocks with restricted attention.

DFlash uses hidden **conditioning**, so deferring hidden-state **prediction** does
not mean removing this standard part of the baseline. There is no auxiliary
hidden-state generation/loss, J-lens, workspace compression, position-specific
output head, final-state output shortcut or learned layer selection here.

## Implemented baseline

- Pinned Qwen2.5-0.5B target; original tokenizer, embedding and vocabulary head frozen.
- Two Qwen2-style draft layers at target width, with fixed post-block features from
  layers 2 and 22 (one-based, uniformly spaced following the reference rule).
- Eight block positions: **one known target anchor plus seven new draft tokens**.
  The frozen `<|fim_pad|>` embedding represents masked inputs. No anchor loss.
- Full-token target feature memory, fused pointwise and injected into each draft
  layer's K/V; RoPE uses actual positions. Packed queries attend only to context
  strictly before their own anchor and to their own bidirectional block.
- Shared vocabulary readout, gamma=4 weighted CE, 6e-4 AdamW with cosine/warmup.
  Random anchors resampled on every update; mixed-domain balanced sampling.
- Cached target verification and incremental draft context K/V. Reject speculative
  suffixes, crop target cache and commit only verified tokens. Greedy decoding only.

This is a scaled Qwen2 adaptation, not a full reproduction of the paper: two draft
layers versus its usual five, 128-token prompts/64-token pilot responses rather
than 3k-token sequences, eight anchors per sampled sequence rather than 512, and
PyTorch SDPA masks rather than Flex Attention. No pretrained 0.5B draft checkpoint
was found in the official supported-model list. Small-target latency benefits are
unproven and must be measured, not inferred from larger-model results.

## Data contract

One prompt per source group. Prose and code are natural continuations; translation
prompts contain English source only. Human continuation/German reference labels
are not used to train the drafter. The frozen target generates greedy continuations
without repetition penalties. EOS is retained; responses too short for a complete
block and exact duplicate generated responses are excluded and counted.

Prose/WMT use official splits; code splits by repository hash. All new validation
and test groups exclude all historical groups; training excludes historical held-out
groups. Exact group/source/prompt duplicates are rejected. Across new partitions,
prompts with token-5-gram Jaccard >=0.8 are rejected. This is a prompt-level near-
duplicate screen, not a comprehensive whole-document contamination audit.

The manifest pins dataset and model revisions, group/source/prompt hashes and token
IDs. Teacher caches contain whole prompt+response sequences and causal features;
strict attention masks prevent clean future features from leaking into predictions.
Unique generated response tokens, eligible anchors and repeated supervised exposures
are separate quantities. The paper's 800k **responses** must not be confused with
our old counts of 132-token windows.

Pilot: 384 requested training prompts, 48 validation and 48 reserved test prompts,
equally allocated across prose/code/translation. Usable counts after response
filtering are reported. Test responses/features/metrics are not generated. This
pilot validates construction and optimization; it cannot determine the final data
requirement. Dataset-size scaling follows only after pipeline correctness.

## Predeclared evaluation and selection

Record step zero and every 100 updates through 1000. Report unweighted CE and
accuracy for each of seven NEW positions, consecutive matches **beyond the anchor**,
and full-seven-token acceptance on fixed held-out teacher trajectories. Training
curves use four diagnostic prompts/domain and are explicitly labeled a sample.
Select maximum validation accepted-new-token count, breaking ties by lower NLL.
No test-based selection. This does not assume seven independent correct tokens form
a correct sequence; only the consecutive prefix counts.

Separately run actual verified generation against ordinary cached target AR. Require
identical greedy output (including EOS/length limits). Initial timing uses one held-
out prompt/domain, 32-token cap, three alternating-order warmed repeats, same target
and precision, CUDA synchronization and end-to-end time including prefill. That is
only a pipeline latency check, not enough evidence for a general speed claim.

Before scaling: demonstrate a stable multi-token accepted prefix on held-out target
responses, inspect domain/position failure patterns and repetition, expand independent
validation coverage, and profile end-to-end latency. Do not resume representation
ablations merely because training loss falls or a tiny set can be memorized.

## Bounded operation

`uv run modal run modal_dflash.py --config configs/dflash-pilot.json --run-id dflash-pilot-01`

The worker has an actual 900-second timeout, no retries, one L40S, and a 720-second
internal deadline. Its $1.25 conservative reservation uses the same $5/hour rate
and locked allocation ledger as historical jobs. The original user plan authorized
$200–$500; after reaching the initial conservative $200 ceiling, the current cap is
$500 for continued baseline training. Existing reservations are not refunded or reset.
The historical runner still reserves its full 3600-second timeout. Latest and best
checkpoints, optimizer/RNG state, prompt manifest and offline teacher cache persist
in the dedicated volume. Continuation restores model, AdamW and CPU/CUDA/sampler RNG from `latest.pt`,
validates unchanged baseline/data fields, and reuses the archived teacher corpus.
Each continuation explicitly starts a new warmup/cosine cycle; it is not represented
as an uninterrupted old schedule. Exact next-update state restoration is unit-tested.

### Checkpoint generation command

After a completed run, download its selected checkpoint and use the greedy verifier:

```bash
uv run modal volume get latent-workspace-qwen-runs dflash-pilot-02/best.pt artifacts/dflash-best.pt
uv run latent-workspace draft --checkpoint artifacts/dflash-best.pt --prompt 'def fibonacci(n):' --tokens 64
```

The command returns verified text and accepted-new-token statistics. Its single-call
timing is not a benchmark. Target weights are loaded from the pinned Hugging Face
revision; the checkpoint stores the draft model, not another copy of Qwen.


## Executed pilot and fitting continuation

The pilot reached 1000 updates: training-diagnostic NLL 7.747, validation NLL 7.820,
and only 0.145 accepted new tokens / 7. This is an underfitting observation on a tiny
response corpus, not evidence that the data requirement has been met. Usable data:
380 train and 47 validation responses after dropping five too-short responses;
20661 unique training response-token slots. All 48 test prompts remain ungenerated.

The BF16 verifier check failed on translation. Replaying identical fixed tokens in
single-token versus block target forwards reproduced an argmax flip: top-two gap
0.0625, maximum logit change 0.25. FP32 arithmetic on the same stored weights matched
AR for all three diagnostic prompts. FP32 is now the correctness-reference path for
the CLI and future rollout checks. This does not make FP32 performance a BF16 speed
claim, or prove bitwise parity for every possible prompt. Failures are retained in
reports; a failed output comparison suppresses the speedup ratio.

`dflash-fit-01` completed 20000 more updates on exactly the same draft/optimizer/data,
with checkpoints every 250 and a new warmup/cosine cycle. It is a fitting
study of the SAME baseline, not another model comparison. The long worker has an
actual 3600-second bound and reserves $5 under the original authorized upper cap.
No new response generation, hidden-state objective, layer selection or test access.

```bash
uv run modal run modal_dflash.py --config configs/dflash-fit.json --run-id dflash-fit-01 --long-run
```

A separate readout audit found a norm-scale difference but an essentially zero
analytical CE lower bound. It does not establish normalization as the cause; no
readout change was introduced on that evidence.

The completed post-training audit evaluated both validation-selected and final checkpoints on
every training response (up to eight fixed anchors/response, not every possible
anchor) and the same validation responses. This checks whether the 12-prompt
training diagnostic is representative; it does not alter checkpoint selection.
The read-only diagnostic has a 360-second hard bound and $0.50 reservation.

```bash
uv run modal run modal_dflash.py --verify-parent dflash-fit-01 --fit-audit --run-id dflash-fit-audit-01
```

## Data limitations to address before a larger fit

The current pilot takes the first eligible source rows and, for prose/code, the
first 128 tokens. Its one-prompt-per-group rule protects separation but does not
provide broad continuation coverage. A retained
[qualitative spot check](../results/dflash-corpus-spotcheck.json) includes a license
header continuation and consecutively selected parliamentary translation prompts.
76/127 code prompts begin with `#`; that simple heuristic does **not** classify
all their continuations as boilerplate. The spot check is not a quality estimate.
Qwen itself can produce poor translations: successful speculative verification
preserves Qwen's outputs, not necessarily human translation quality.

The next data preparation should use seeded source mixing, varied valid continuation
positions for prose/code, explicit response-length/repetition audits, and more
independent validation groups. Keep translation instructions/source intact and
keep existing test groups quarantined. Archive actual usable response counts,
unique target-token slots and repeated exposures separately. Expand response
diversity with the same drafting architecture before representation ablations.
These are follow-up requirements, **not implemented changes to the measured fit**.

## Final measured status

[Completed results and curves](../results/dflash-findings.md): all 21000 updates
finished. The all-response audit found final train NLL 0.0052 / 6.9987 new matches,
versus validation NLL 9.6334 / 0.2557 matches. Best validation acceptance was 0.2844
at update 17000. FP32 rollout matched all three checked prompts, but 0.879–0.912x
AR speed means slower generation. Full training fitting does not establish useful
held-out drafting or a sufficient data budget. The next priority is the data work
above; representation experiments remain deferred. All new test prompts stay sealed.
