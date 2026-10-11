# Qwen3.5-2B drafting quality — completed experiment

**This 2B setup did not improve drafting.** Both targets received fresh drafter
training on 6144 mixed-domain responses for 20000 updates. On the same 184
validation source prompts, their independently validation-selected checkpoints
match **0.387 NEW tokens for Qwen2.5-0.5B versus 0.170 for Qwen3.5-2B**, out of
seven proposed tokens after a supplied anchor. These are consecutive teacher-token
matches, not verified rollout acceptance or human answer-quality scores.

The 2B head is weak on training too: its selected checkpoint scores **0.173 on
all training responses and 0.170 on validation**. Its final checkpoint scores
0.175/0.168. This is a failure to learn useful training predictions, not evidence
that it has successfully memorized this corpus and only fails to generalize.
Neither a larger backbone nor simply adding data is established as the fix.

![Completed drafting quality curves](dflash-qwen35-2b-comparison.png)

The curves compare equal measured update counts; their validation cohorts contain
192 and 184 responses respectively. The table below removes the cohort difference
by restricting both selected checkpoints to the same source groups. Each model
still has its own responses, tokenization and valid anchors; this is not a pure
parameter-count ablation. Dashed training curves use 12 diagnostic responses;
the separate fitting audit below evaluates all 6144.

| Shared validation domain | Prompts | 0.5B selected NEW matches / 7 | 2B selected NEW matches / 7 |
| --- | ---: | ---: | ---: |
| Prose | 63 | 0.536 | 0.140 |
| Code | 57 | 0.375 | 0.239 |
| Translation | 64 | 0.249 | 0.131 |
| Domain macro average | 184 | **0.387** | **0.170** |

Selected steps are 17500 for 0.5B and 16750 for 2B, chosen within the same
20000-update budget. One seed and reused validation make this exploratory;
there is no independent test result or claim about arbitrary-domain transfer.

## Training and data

The 2B target is `Qwen/Qwen3.5-2B` at immutable revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`. Target weights, embeddings and vocabulary
readout remain frozen. The two-layer Qwen2-style bidirectional drafter has
104876032 trainable parameters, versus 31432192 before. It retains biased dense
attention projections and uses the target's partial rotary width; it does not
reproduce Qwen3.5's gated/recurrent blocks. This is a DFlash-style adaptation,
not a full reproduction of DFlash or its reported scaling results.

The targets differ in family, tokenizer, backbone architecture, draft width and
prompt format. Qwen3.5 uses the official non-thinking chat template with explicit
continuation/translation instructions. Raw-prompt smoke examples produced
repetitive prose and appended thinking text, motivating the intended instruction
interface. The raw and corrected smoke samples are not a paired formatting or
semantic-quality benchmark. No claim that the 2B language model itself is worse
follows from these drafting results.

All 6984 original source groups/splits are retained: 6600 candidate training,
192 validation and 192 reserved test prompts. Source text is unchanged except
for the official template stripping trailing whitespace in 544 records. Pinned
sources, code-repository grouping, source hashes and near-duplicate screening
come from the preceding [data study](dflash-data-scaling.md). Fresh 2B generation
yields 6390 usable training and 184 validation responses after dropping 211
short and seven duplicate responses. The balanced subset has 2048 responses per
domain and 295691 distinct response-token positions; 0.5B had 341069 positions.
The test prompts have no generated responses, features or metrics.

Both fits use seed 42, batch three, eight random anchors per sequence, seven
masked NEW positions, fixed features from post-block states 2/22, gamma four,
AdamW at 0.0006 and 4% warmup followed by cosine decay. The 2B fit completed in
2505.4 seconds on one L40S. It sampled 480000 blocks and **3360000 NEW-token label
exposures**. Checkpoint-verified sampler replay counts **72972285 conditioning
position exposures**: 63333400 prompt positions and 9638885 earlier-response
positions. These are repeated positions attended per block using cached features,
not backbone training or fresh target forward computations.

## Full fitting audit and concrete predictions

| 2B checkpoint | Step | Train NEW / 7 | Validation NEW / 7 | Train NLL | Validation NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected | 16750 | 0.173 | 0.170 | 6.594 | 6.817 |
| Final | 20000 | 0.175 | 0.168 | 6.570 | 6.801 |

Every training and validation response is evaluated at up to eight fixed anchors,
not every possible anchor. Both validation audits match their original training
curve entries exactly. NLL here diagnoses fitting within the 2B task; raw NLL
across different tokenizers and response distributions is not a fair quality ranking.

Fixed examples from the selected head illustrate the failure. These use the
first validation response per domain, with a supplied correct anchor; there is
no verifier or cherry-picked successful generation:

| Domain | Supplied anchor | Teacher's next seven tokens, decoded | Drafter's seven tokens, decoded |
| --- | --- | --- | --- |
| Prose | `The` | ` final standings were determined by the points` | ` the the the the the- @` |
| Translation | `Ich` | ` erinnere mich, dass Menschen in` | `,,,,,,,` |

All six fixed blocks in the [audit](dflash-qwen35-2b-fit-audit-01.json), including
code and later-response anchors, fail on the first NEW token. These examples are
illustrations, not the aggregate benchmark. The numerical audit is the stronger
evidence that fitting remains poor.

A separate six-response real-model [teacher audit](dflash-qwen35-2b-teacher-audit-01.json)
finds full-forward teacher-token agreement, cached-feature cosine above 0.99989,
and exactly zero change to earlier features after perturbing future tokens.
This reduces concern about a gross data/cache alignment failure, without proving
that every feature or response in the corpus is correct.

## What this supports

Increasing backbone size alone did not help this implementation. Training and
validation both remain weak; this is not the earlier tiny-corpus memorization
pattern. The next useful work is to establish reliable drafter fitting and
training fidelity, then jointly study optimization exposure and data coverage.
A Qwen2.5-1.5B control would isolate size more cleanly; a native hybrid-target
adapter would address a different architectural question. Neither was launched.
Representation/J-space/layer sweeps and inference speed optimization remain deferred.

This quality-only experiment intentionally does not exercise the existing KV-crop
verifier on Qwen3.5's recurrent cache. Verification is recorded as null, not passed.
There is no verified 2B rollout, speedup or sufficient-data claim.

## Reproducibility and artifacts

- [Training report](dflash-qwen35-2b-fit-01.json), [comparison data](dflash-qwen35-2b-comparison.json),
  [full fitting audit](dflash-qwen35-2b-fit-audit-01.json).
- [Teacher preparation](dflash-qwen35-2b-chat-data-01.json),
  [prompt identity audit](dflash-qwen35-2b-prompt-audit.json),
  [source launch audit](dflash-qwen35-2b-launch-audit.json),
  [checkpoint audit](dflash-qwen35-2b-checkpoint-audit.json).
- [Protocol](../docs/dflash-qwen35-2b.md),
  [configuration](../configs/dflash-qwen35-2b-fit.json),
  [retained preparation failures](dflash-qwen35-2b-preparation-audit.json).
- Both GPU preparation/training package hashes match committed source `4d3fa54`.
  The later fitting audit records its own source hash separately.
- Selected checkpoint: `/runs/dflash-qwen35-2b-fit-01/best.pt` on the dedicated
  Modal volume; checksum-verified local copy `artifacts/dflash-qwen35-2b-best.pt`.
  SHA-256: `82de0b090d98574e79371d4c9daa6aaedf314ce0d75a9c1acba0024628c3c748`.
- 79 local tests, Ruff/format checks, real GPU smoke, completed fitting, teacher
  consistency and full-response audits passed. All jobs in this comparison ended.
  Conservative cumulative reservations are $258.25/$500, not confirmed billing;
  failed and stopped jobs retain their reservations.

References: [DFlash paper](https://arxiv.org/abs/2602.06036),
[pinned classic implementation](https://github.com/z-lab/dflash/tree/07ebd93db9f472af339b644bb70221ad8428328a),
[Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B), and
[project bibliography](../references/papers.bib). MIT attribution remains in
[references/dflash-LICENSE](../references/dflash-LICENSE).
