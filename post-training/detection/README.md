# Detection

Post-training a small vision-language model so that it detects requested objects in
one forward pass, with a box and a short label per detection.

**Status:** the evaluation harness, three baselines, custom decode kernels and a first
training run of the slot head are implemented and measured. The slot head is a
validation experiment, not a finished model.

Measured outputs (predictions, summaries, training curves, the trained head) are in the
Hugging Face dataset [`jfrick/lm-exploration-results`](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/pr-167/post-training/detection)
under `pr-167/post-training/detection/<run id>/`.

## Question

A vision-language model that writes detections one after another pays decoder steps
per object and one prompt per category. Can post-training alone give the same model
a set-prediction output (n slots, each "nothing" or a box with a label) so that
latency stops growing with the number of objects and phrases, and can a 2B model
trained this way compete with a larger one?

The aim is to find something reliably, not to enumerate everything: precision and the
share of phrases with a correct box matter more than exhaustive recall, and training
uses only labels worth learning from.

## Models

| Model | Size | Runtime here | Licence |
|---|---|---|---|
| Moondream 2, `vikhyatk/moondream2` `2025-06-21` | 1.9B dense | Transformers (its Hub code) | Apache-2.0 |
| Moondream 3.1, `moondream/moondream3.1-9B-A2B` (July 2026) | 9.3B mixture of experts, 2B active | Photon | Moondream Model License 1.0 |

Moondream 3.1 is the most recent release; its repository holds weights only and
[Photon](https://docs.moondream.ai/running-locally) (`pip install moondream`) is the
published way to run it. Photon also serves Moondream 2. Photon's `kestrel-kernels`
package states a proprietary licence that grants use under a separate agreement, while
Moondream's documentation describes local inference as free; this folder only calls
the engine through its public client and does not inspect it.

Moondream 2's `detect(image, phrase)` takes one phrase per call. The image is encoded
once and its key/value cache is reused. Each object then costs three decoder steps (x
centre, y centre, size), read from a region head as 1,024-bin distributions. Boxes
carry no confidence.

## Evaluation protocol

Every task has the same layout (`src/parallel_detection/tasks.py`): an image and a
list of phrases, each with its truth boxes. Only categories present in the image are
prompted.

| Dataset | Phrases per image | Source |
|---|---|---|
| COCO val2017 | Every annotated category | `detection-datasets/coco` |
| RefCOCO-M val | Every expression of every referred object | `moondream/refcoco-m`, Moondream's re-annotated RefCOCO |
| LVIS v1 minival | Every annotated category | `GLIPModel/GLIP` annotation file; COCO images |

Scores (`src/parallel_detection/metrics.py`), at IoU 0.5 and 0.75:

- Micro precision, recall and F1 with the largest one-to-one matching per phrase. For
  categories LVIS marks as not exhaustively annotated, unmatched boxes are not errors.
- Phrase hit rate: the share of phrases with at least one correct box.
- Unit-confidence AP: the average precision of
  [Moondream's COCO script](https://github.com/vikhyat/moondream/blob/main/moondream/eval/coco_map.py),
  which gives every box confidence 1 and averages over (image, category) pairs.
- For referring expressions, accuracy of the first box.
- Latency per image and per phrase, and phrase seconds against boxes emitted.

The runs use a seeded sample: 500 COCO images (1,446 phrases), 300 RefCOCO-M images
(1,409 expressions) and 300 LVIS images (953 phrases). `max_absent` in a config adds
categories LVIS verified absent, to measure false alarms.

## Baseline results

IoU 0.5. Runs `moondream2-001`, `moondream2-photon-001`, `moondream31-001`.

| Model, runtime, GPU | Dataset | Precision | Recall | F1 | Phrase hit rate | Unit-confidence AP |
|---|---|---|---|---|---|---|
| Moondream 2, Transformers, L4 | COCO | 0.500 | 0.709 | 0.587 | 0.890 | 0.780 |
| | RefCOCO-M | 0.447 | 0.837 | 0.583 | 0.837 | 0.799 |
| | LVIS minival | 0.750 | 0.564 | 0.644 | 0.902 | 0.742 |
| Moondream 2, Photon, L40S | COCO | 0.501 | 0.706 | 0.586 | 0.894 | 0.784 |
| | RefCOCO-M | 0.450 | 0.825 | 0.582 | 0.825 | 0.789 |
| | LVIS minival | 0.783 | 0.556 | 0.650 | 0.901 | 0.744 |
| Moondream 3.1, Photon, L40S | COCO | 0.682 | 0.719 | 0.700 | 0.902 | 0.790 |
| | RefCOCO-M | 0.931 | 0.941 | 0.936 | 0.941 | 0.940 |
| | LVIS minival | 0.800 | 0.464 | 0.588 | 0.785 | 0.639 |

First-box accuracy on RefCOCO-M is 0.769 for Moondream 2 (Transformers) and 0.940 for
Moondream 3.1.

What separates the two models is precision, not recall. On COCO both find about 71% of
the objects, and Moondream 2 finds more than Moondream 3.1 on LVIS. Moondream 2 emits
extra boxes: on RefCOCO-M it returns 8.8 boxes per image for 4.7 single-object
expressions. Its recall falls with object size (0.89 large, 0.75 medium, 0.45 small on
COCO) and with instances per phrase (0.86 for one, 0.62 for 6 to 20).

Latency, mean seconds:

| Model, runtime, GPU | Per COCO image | Image encoding | Phrase with no boxes | Per extra box |
|---|---|---|---|---|
| Moondream 2, Transformers, L4 | 1.083 | 0.225 | 0.035 | 0.074 |
| Moondream 2, Photon, L40S | 0.220 | in first phrase | 0.026 | 0.014 |
| Moondream 3.1, Photon, L40S | 0.208 | in first phrase | 0.037 | 0.013 |

The Transformers and Photon rows use different GPUs, so they do not isolate the
runtime. A COCO image averages 2.9 phrases and about 10 boxes.

### Against Moondream's published numbers

| Published | Value | Here | Difference in protocol |
|---|---|---|---|
| Moondream 3.1 COCO val F1@0.5 | 81.46 | 70.0 | Their label set and prompt protocol are not published |
| Moondream 3.1 LVIS val F1@0.5 | 67.4 | 58.8 | LVIS val there, a 300-image minival sample here |
| Moondream 3.1 RefCOCO-M val F1@0.5 | 87.1 | 93.6 | Same dataset; a 300-image sample here |
| Moondream 2 (`2025-03-27`) COCO mAP | 51.2 | 78.0 | Their cleaned COCO labels are not public; a later release here |

The published 3.1 figures come from the
[release notes](https://moondream.ai/blog/moondream-3-1-beyond-benchmarks) (one H100,
Photon, batch size 16, 34.2 requests per second averaged over their suite). The numbers
here do not reproduce them: the COCO and LVIS figures are lower and RefCOCO-M is
higher. Each sample is 300 to 500 images, so a few points are sampling noise; the rest
is protocol that the release notes do not specify.

An earlier run, `baseline-001`, prompted LVIS images with verified-absent categories as
well. Modal preempted it after 50 images; in those, Moondream 2 drew at least one box
for 53% of 249 absent categories.

## Custom decode kernels

`src/parallel_detection/decode_step.cu` runs one batch-one decoder step of Moondream 2
as three fused CUDA launches per layer: layer norm with the qkv and fc1 rows, attention
with rotary embedding and key/value append, and the proj and fc2 rows added to the
residual stream. It reads the model's BF16 weights in place and computes in FP32 with
its own FP32 key/value cache (`kernel.py` binds it through ctypes).

Run `decoders-003`: 100 COCO images, 265 phrases, one L4.

| Decoder | Per extra box (s) | Per image (s) | F1 | Phrases with boxes identical to eager |
|---|---|---|---|---|
| PyTorch eager (the model's `detect`) | 0.069 | 0.896 | 0.638 | 100% |
| This folder's loop, PyTorch steps | 0.067 | 0.882 | 0.638 | 100% |
| torch.compile (the model's `compile()`) | 0.046 | 0.738 | 0.614 | 32% |
| Custom kernels | 0.037 | 0.599 | 0.670 | 48% |

The kernels cut the per-box cost by 1.85x against eager and 1.2x against the model's
own compiled path, about 12 ms per decoder step. They do not reproduce eager's boxes
exactly: 95% of phrases get the same number of boxes, with a mean IoU of 0.96 between
paired boxes. The compiled path shows the same kind of divergence, which points to the
bin argmax being sensitive to BF16 rounding, but the kernels have not been checked
against an FP32 PyTorch reference step by step, and the F1 difference on 100 images is
within noise. Photon's own kernels reach 0.014 s per box on an L40S; these have not
been timed on that GPU. Image encoding and prompt prefill stay in PyTorch.

## Slot head: boxes by regression, labels by one-pass diffusion

`src/parallel_detection/slots.py`. After the 730-position image prefix, the decoder
reads 32 slots of four positions each in a single forward pass. Slot positions see the
image and each other.

- The first position of a slot is a learned query. Its final hidden state gives an
  objectness logit (the slot may be "nothing") and a box as four regressed coordinates.
- The other three positions are the slot's label. They start as a mask embedding and
  the model's own vocabulary head predicts their tokens: one step of masked discrete
  diffusion, the mechanism of the
  [DFlash-style drafter](../../exploration/latent-workspace/docs/implementing-a-diffusion-style-drafter-from-scratch-and-learnings.md).
  More passes keep the most confident tokens and predict the rest again.

The backbone is frozen. Rank-16 adapters act only at slot positions, so the image
prefix and the model's existing skills are unchanged; 16.9M parameters train. The loss
is Hungarian matching (box L1, generalized IoU, objectness, label likelihood) plus a
denoising pass that reveals a random share of each matched label. Unmatched slots are
pushed toward "nothing" with weight 0.1.

### Training run `slots-001`

- Data: four COCO train2017 shards, keeping objects that cover at least 0.2% of the
  image in images with 1 to 20 such objects: 11,105 training and 300 held-out images.
  Image features are cached once.
- 6,000 steps of batch 16 on one H100, 28 minutes (0.28 s per step).

Held-out images (labels must match exactly, detections above the objectness threshold):

| Step | Matched-box IoU | Label exact, 1 pass | Label exact, 2 passes | Best F1@0.5 |
|---|---|---|---|---|
| 500 | 0.31 | 0.772 | 0.733 | 0.10 |
| 2,000 | 0.50 | 0.893 | 0.882 | 0.34 |
| 4,000 | 0.56 | 0.911 | 0.911 | 0.41 |
| 6,000 | 0.59 | 0.919 | 0.916 | 0.45 |

COCO val2017, the same 500 images and L4 as the Moondream 2 baseline (run
`slots-001-coco`). The slot head answers every phrase from one pass.

| | Moondream 2 `detect` | Slot head |
|---|---|---|
| Seconds per image | 1.083 | 0.263 |
| of which image encoding | 0.225 | 0.225 |
| Precision / recall / F1, IoU 0.5 | 0.500 / 0.709 / 0.587 | 0.436 / 0.421 / 0.429 |
| Precision / recall / F1, IoU 0.75 | 0.380 / 0.538 / 0.445 | 0.240 / 0.231 / 0.235 |
| Phrase hit rate, IoU 0.5 | 0.890 | 0.598 |
| Recall large / medium / small, IoU 0.5 | 0.889 / 0.753 / 0.454 | 0.774 / 0.423 / 0.020 |

What this shows:

- **One-pass diffusion labels work for this label set.** 92% of matched slots get all
  three label tokens right in one pass, 28 of the 80 names being two or three tokens. A
  second pass does not help. About 5% of confident slots decode to a token block that
  is not a category name. COCO names have one wording each, so this run does not test
  labels with several valid wordings.
- **Speed.** Detection for the whole image costs about 0.04 s after image encoding,
  against 0.86 s of sequential decoding for the same images. Image encoding, at the
  baseline run's 0.225 s, is now about 85% of the time.
- **Boxes are the weak part.** The regressed boxes are loose (matched IoU 0.59; F1 at
  IoU 0.75 is half the baseline's). Held-out F1 was still rising when the schedule
  ended, and the best objectness threshold sat at the top of the searched grid (0.8).
- **Small objects are absent by construction.** The label filter removed them from
  training, and recall on small COCO objects is 0.02.

The run validates the mechanism and the latency; it does not yet match the model's own
sequential detector in accuracy.

## Run

From `post-training/detection/`:

```sh
uv venv --python 3.12
uv pip install -e ".[dev,modal]"
.venv/bin/pytest
.venv/bin/ruff check . && .venv/bin/ruff format --check .

.venv/bin/modal run --detach modal_app.py::main --run-id moondream2-001
.venv/bin/modal run --detach modal_app.py::photon --run-id moondream31-001 \
  --config configs/moondream3.1-photon.json
.venv/bin/modal run --detach modal_app.py::decoders --run-id decoders-001
.venv/bin/modal run --detach modal_app.py::slots --run-id slots-001
.venv/bin/modal run --detach modal_app.py::main --run-id slots-001-coco \
  --config configs/slots-coco.json

# Recompute scores from saved predictions.
.venv/bin/python -m parallel_detection.run score \
  --predictions artifacts/moondream2-001/predictions.jsonl
```

Results land on the Modal volume `lm-exploration-parallel-detection` (`evaluations/<run
id>/`, `slots/<run id>/`) and the launching command saves the summary under
`artifacts/<run id>/`. A local run needs the `model` extra and a GPU.

Every Modal call has `retries=0`. Evaluations stop within one hour of launch and
training within three; a separate CPU supervisor, built on
`training/workflows/deadline.py`, cancels a call still queued or running at its
deadline. A preempted evaluation continues from the images it recorded.

## Open work

- Box quality: longer training, more images, a wider threshold grid, and box
  prediction as the model's own coordinate bins instead of four regressed numbers.
- Small objects and the label filter; Objects365 and LVIS as training data.
  `jxu124/objects365` holds annotations only, so its images need a prepared subset.
- Prompted detection (a slot points at a phrase in the prompt), and labels with
  several valid wordings as the real test of one-pass diffusion.
- Fine-tuning the slot head toward Moondream 3.1's precision on RefCOCO-M and COCO.
- Kernels: an FP32 PyTorch reference check, the image encoder, and timing on the GPU
  Photon was measured on.
