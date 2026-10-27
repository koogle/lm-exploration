# 100M-class KDA mixer comparison

Follow-up to [the short mixer pilot](STUDY.md), based on PR77. The four best
pilot variants compete with the original conv-4 KDA baseline: no convolution,
four-mode rotating memory, RoPE without convolution, and RoPE replacing the
convolution in the middle KDA layer of each group. Full-attention layers are
unchanged. Rotating memory uses the custom Triton training implementation.

The planned architecture has 12 layers, width 768, SwiGLU width 1792, 12 KDA
heads of width 64, and full attention with 12 query / 4 KV heads. Its repeating
layout is KDA/KDA/KDA/full. Vocabulary size is 32,768 with tied embeddings.
Measured parameter counts range from 104,314,704 (no convolution and RoPE)
to 104,397,864 (rotating memory); baseline has 104,397,648 parameters.

Each of two seeds (17, 29) receives 12,288 updates of eight 1,024-token
sequences: 100,663,296 input token positions. The shared initial parameters,
data order, optimizer and learning-rate schedule match across variants.
AdamW uses peak learning rate 0.0003, 400-step warmup, cosine decay to 10%,
betas (0.9, 0.95), weight decay 0.1 and gradient clipping at 1. Training uses
BF16 autocast, FP32 parameters and the existing z-loss. Held-out cross-entropy
excludes z-loss and BOS labels. Evaluation uses the same 262,144 held-out
token positions every 512 steps, plus initial and first-step measurements.

Data preparation selects approximately 101M training tokens and 524K held-out
tokens from the same frozen ClimbMix/code snapshot and tokenizer as the pilot,
balanced approximately equally across its six sources. Documents are shuffled
with seed 2026 before packing. Runs consume a single prefix, without recycling
the pilot's small corpus. Preparation requires each source to supply its quota
and checks that selected document hashes do not overlap between splits.
The snapshot's existing incomplete corpus-audit status remains a limitation.

Before training, the custom rotating-memory kernel must pass its original
FP32/BF16 output/gradient tests and a BF16 check at the actual 8 × 1024 × 768
input shape, including document resets. This is a numerical qualification;
long-run training stability and quality remain experimental outcomes.

Each training worker has a three-hour execution timeout and no retries.
An independent remote supervisor cancels all tracked preparation,
qualification and training calls at the overall 3.5-hour launch deadline,
including queued work. Metrics are saved at each evaluation; model/optimizer
checkpoints are saved after the first update and every 2,048 updates. A
deadline exit saves a partial checkpoint and is reported as incomplete.
The latest checkpoint for each run lives on the training Modal volume.

Run from `training/` using the Modal environment with system CA trust:

```sh
python launch_kda_long.py
```

Executable configuration is in `hybrid/long_study.py`. The launcher archives
structured metrics and provenance to the private HF dataset
`jfrick/lm-exploration-results`, under `training/results/<run-id>`.
Compare final loss across paired seeds, learning curves at matched token
budgets, throughput and memory. Two seeds and 100M tokens can identify a
promising follow-up; they do not establish the ranking of fully pretrained
models or behavior beyond the trained context length.

The actual-shape kernel check passed: maximum relative L2 output/gradient
error was 0.0000265 (about 0.003%). Original FP32/BF16 checks also passed.
The first launch, `kda-100m-1793059455`, prepared 101,016,001 training tokens
and 527,104 validation tokens and passed qualification. Its local collector
caught the wrong timeout type when polling Modal and cancelled pending
workers before training. The failure logs and qualification evidence are
preserved in the HF results archive.

Run `kda-100m-1793059841` uses the corrected collector and hash-verified copies
of those exact prepared arrays. It retains the first launch's overall deadline.
Training results are pending. An automatic collector writes paired summaries
and plots to the same HF results folder when the worker results arrive.

After collection, run `python -m hybrid.analyze_long_study artifacts/results/<run-id>`
from `training/` to produce structured comparisons and learning-curve plots.

The speed benchmark runs all five variants on one H100 at the same batch and
context sizes, with ten warmup updates each followed by four rounds of twenty
updates in alternating forward/reverse variant order. All models stay resident;
timings cover forward/backward/clipping/AdamW and exclude data transfer,
evaluation and compilation. Launch it with
`python launch_kda_benchmark.py --run <run-id> --long` from `training/`.
