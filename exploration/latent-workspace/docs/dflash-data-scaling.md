# DFlash response-data scaling

Predeclared 2026-10-11, before larger-pool results. User requested more data and an
explanation of whether DFlash generalizes. Keep the existing draft architecture.

## What the reference establishes

[Classic DFlash §5, §5.4.1 and Appendix A](https://arxiv.org/html/2602.06036v1)
uses roughly 800k samples from Nemotron Post-Training V2 and CodeAlpaca, replacing
original answers with target-generated responses. Its standard setting uses five
draft layers, block size 16, five fixed target feature layers, 3072-token sequences,
512 randomly sampled anchors per sequence and six training epochs. Targets are
Qwen3 4B/8B/30B-A3B and LLaMA3.1-8B, not our Qwen2.5-0.5B base model.

Evaluation includes GSM8K/MATH/AIME, HumanEval/MBPP/LiveCodeBench and MT-Bench/Alpaca.
The controlled LLaMA comparison trains on UltraChat/ShareGPT and evaluates other
benchmark collections. Thus the reported claim is cross-dataset drafting utility,
not lookup/memorization of a fixed list of prompts. The paper text does not establish
that every benchmark example is contamination-free or guarantee arbitrary domain
transfer. Model-target alignment, inference-matched anchor masking, rich target
features, position-weighted loss and cheap parallel drafting all contribute to the
method; the paper's ablations support several of these, not a claim that data alone
causes its performance. Verification preserves target behavior independently of
whether the drafter is good; generalization controls acceptance and speed.

Our 380-response fit is over 2000 times smaller by response count and much shorter
per response. Small-target speed is also harder to improve because the target is
already cheap. Its failure is not a reproduction-based refutation of DFlash.

| Setting | Paper's standard Qwen3 setup | This controlled comparison |
| --- | --- | --- |
| Target | 4B/8B; separate 30B-A3B Coder setting | Qwen2.5-0.5B base |
| Training responses | Around 800k; 100k for most ablations | 768 or 6144 |
| Draft layers / fixed feature layers | Usually 5 / 5 | 2 / 2 |
| Block positions including supplied anchor | 16 | 8 |
| Maximum total sequence length | 3072; 4096 for Coder | 192 (128 prompt + 64 response) |
| Sampled anchors per sequence | 512 | 8 per sampled response visit |
| Evaluation scope | Math, code, chat benchmark collections | New groups from prose, Python and English/German collections |

These differences are not separately ablated here. DFlash does not look up stored
answers at inference: the target computes features of the current prompt, then
verifies proposed blocks. An unfamiliar prompt may reduce acceptance/speed without
invalidating the speculative-decoding construction. Our observed numerical-precision
limitations are documented separately; exact output equality is checked, not assumed.

## Comparison

Generate one pool from 6600 training prompts, 192 validation and 192 test prompts.
Only train/validation receive Qwen responses; reserve test prompts without labels.
After filtering, require at least 2048 usable responses/domain. Select deterministic
balanced nested training subsets of **768 and 6144** responses. Both models use the
exact same validation cache, random initialization seed 42 and target weights.

Use the unchanged two-layer, width-896 drafter, fixed target features 2/22, anchor
plus seven fully masked proposals, gamma 4, frozen shared readout, AdamW 6e-4,
4% warmup and cosine decay. Train each from scratch for **20000 updates**, batch 3,
eight anchors/response: 480000 sampled anchor blocks and 3360000 supervised new-token
exposures per fit. Evaluate every 250 updates. This matches optimization exposure,
not epochs: mean sampled response visits are 78.125 versus 9.766. A fixed-budget
comparison does not demonstrate convergence, especially for the larger dataset.

Selection: maximum validation consecutive NEW matches, tie-breaker lower NLL.
Keep anchor excluded. Report per-position/domain metrics and 12-response training
diagnostics; conduct a post-training fitting audit before broad fitting claims.
Retain per-prompt selected-checkpoint validation scores for paired uncertainty
estimates. Run the FP32 verified-rollout/timing check on four prompts/domain
(12 total), rather than the previous three-prompt diagnostic. No test selection.

## Improved source sampling and separation

Pinned sources remain WikiText103, CodeParrot-clean Python and WMT14 English/German.
Use their official TRAIN pools only, with seeded 4096-item buffer shuffling (after
assembling prose documents). Partition source groups by deterministic 80/10/10 hash:
code by repository, prose by document hash, translation by English sentence hash.
This provides more independent prose validation groups than its small official
validation split. Translation lacks document-family identifiers: sentence-level
separation and near-duplicate screening do not guarantee family-level independence.

Register all prior DFlash prompt groups before selecting new ones; additionally,
new validation/test exclude all historical groups and new training excludes historical
held-out groups. Exact source/prompt/group duplicates and cross-split prompt token-
5-gram Jaccard >=0.8 are rejected. This is not a comprehensive contamination audit.

Select deterministic varied 128-token prose/code segments, rather than file beginnings
only; cap source tokenization at 100000 characters and record offsets. Translation
instructions/source stay intact. Responses remain at most 64 tokens to isolate data
quantity within this comparison. The previous first-prefix study is historical context,
not a matched control for the sampler changes.

Generate greedy target responses in batches of 32 using left-padded prompts; cache
causal target features using masked right-padding, trimming every example exactly.
Drop too-short and duplicate responses with validation priority. Retain EOS, source
revisions, split manifest, offsets, generated token identities and actual usable counts.
BF16 generation may differ with batch shape; both fits use the same archived cache.
CPU tests check padded generation and feature-cache alignment; a GPU preparation
smoke precedes the full pool. Existing correctness-reference rollout remains FP32.

## Execution and budget

Configs: `dflash-data-smoke.json`, `dflash-data-pool.json`, `dflash-scale-{768,6144}.json`.
Data preparation uses `modal_dflash.py --long-run --prepare-data`; training uses
`--long-run`. Each worker has an actual 3600-second bound, no retries and one L40S;
each reserves $5. Four planned workers reserve $20 total from the original $500 cap.
Current reservations before these launches: $205.50, not actual billed cost.
Two independent training apps may run concurrently; each has its own reservation.
A new job is required for any continuation. Raw curves, checkpoints and data remain
on the dedicated volume. Push configs/source/protocol before executing the comparison.
