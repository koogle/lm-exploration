# Detection

Post-training a small vision-language model so that it detects requested objects in
one forward pass, with a box and a short label per detection.

**Status:** the evaluation harness, three baselines, custom decode kernels and six
training runs of the slot head are implemented and measured. The best slot head beats
Moondream 2's own detector on COCO F1 at a quarter of the latency, runs 2.8 times faster
than Moondream 3.1 on the same GPU, and trails 3.1 by 6 points of F1, all in recall.

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

## Slot head: one pass, masked diffusion over boxes and labels

`src/parallel_detection/slots.py`. After the 730-position image prefix, the decoder
reads a block of six positions per slot in a single forward pass. Slot positions see
the image and each other. Every position starts as a mask and is read through the
model's own frozen heads:

- positions 0 to 2 are the box: x centre and y centre through the region head's
  coordinate decoder, width and height through its size decoder, each a distribution
  over the 1,024 bins Moondream's sequential detector uses;
- positions 3 to 5 are the label, through the vocabulary head;
- the first position also gives an objectness logit (the slot may be "nothing").

One pass with everything masked is the one-step sampler, the mechanism of the
[DFlash-style drafter](../../exploration/latent-workspace/docs/implementing-a-diffusion-style-drafter-from-scratch-and-learnings.md).
Further passes reveal the most confident positions and predict the rest again.

The backbone is frozen. Rank-16 adapters act only at slot positions, so the image
prefix and the model's existing skills are unchanged; 12.7M parameters train with 32
slots.

### Ground truth and loss

- Data: COCO train2017 boxes and category names (`detection-datasets/coco`). An object
  is kept if it covers at least `min_area` of the image; an image is used if it has 1
  to `max_objects` kept objects. 300 images of the same pool are held out.
- Targets: the category name as three tokens (padded with end-of-sequence), and the
  box as Moondream's four bins.
- Hungarian matching per image on box distance, overlap, objectness and label
  likelihood. Matched slots get a cross-entropy on the four bins, L1 and generalized
  IoU on the expected-bin box and a cross-entropy on the label. Unmatched slots are
  pushed toward "nothing" with weight 0.1, so an unlabelled object is penalised mildly.
- A second pass per step reveals a random share of each matched slot's true
  coordinates and tokens and trains the rest.
- Image features are encoded once and cached; there is no augmentation.

### Training runs

All on one H100, batch 16.

| Run | Boxes | Images | Slots | `min_area` | Steps | Minutes |
|---|---|---|---|---|---|---|
| `slots-001` | Four regressed numbers | 11k | 32 | 0.2% | 6,000 | 28 |
| `slots-bins-4` | Coordinate bins | 11k | 32 | 0.2% | 6,000 | 34 |
| `slots-bins-16` | Coordinate bins | 45k | 32 | 0.2% | 20,000 | 132 |
| `slots-sizes-none` | Coordinate bins | 23k | 48 | 0.03% | 10,000 | 78 |
| `slots-sizes-grouped` | Coordinate bins, size groups | 23k | 48 | 0.03% | 10,000 | 74 |
| `slots-small-16` | Coordinate bins | 46k | 48 | 0.03% | 20,000 | 184 |

In `slots-sizes-grouped` the first 24 slots are matched only to objects under 1% of the
image and the other 24 to larger ones.

COCO val2017, the same 500 images as the baseline, on an L4:

| | F1, IoU 0.5 | F1, IoU 0.75 | Precision | Recall | Recall on small objects | Seconds per image |
|---|---|---|---|---|---|---|
| Moondream 2 `detect` | 0.587 | 0.445 | 0.500 | 0.709 | 0.454 | 1.083 |
| `slots-001` | 0.429 | 0.235 | 0.436 | 0.421 | 0.020 | 0.263 |
| `slots-bins-4` | 0.563 | 0.457 | 0.639 | 0.503 | 0.107 | 0.259 |
| `slots-bins-16` | 0.622 | 0.518 | 0.757 | 0.528 | 0.104 | 0.255 |
| `slots-sizes-none` | 0.590 | 0.469 | 0.692 | 0.514 | 0.239 | 0.233 |
| `slots-sizes-grouped` | 0.549 | 0.431 | 0.634 | 0.484 | 0.207 | 0.217 |
| `slots-small-16` | 0.636 | 0.507 | 0.722 | 0.569 | 0.257 | 0.240 |

Findings:

- **Coordinate bins replace regression.** On the same data and schedule, reading boxes
  through Moondream's own bins raised held-out matched-box IoU from 0.59 to 0.73 and
  COCO F1 from 0.43 to 0.56; the bin head passed the regression head's final held-out
  score within six minutes.
- **More data and steps keep helping.** `slots-bins-16` beats Moondream 2's own
  detector on F1 at both thresholds at a quarter of the latency. Its precision (0.757)
  is above Moondream 3.1's (0.682); its recall is well below.
- **One-pass diffusion suffices for this label set.** Labels are exact for 92 to 95% of
  matched held-out slots in one pass, and two or three passes change neither labels nor
  boxes. COCO names have one wording each, so labels with several valid wordings are
  untested.
- **Keeping small objects in training more than doubles small-object recall.** On the
  same 45k-image, 20,000-step schedule, `slots-small-16` against `slots-bins-16` raises
  small-object recall from 0.10 to 0.26 and overall recall from 0.528 to 0.569, for 3.5
  points of precision; F1 rises from 0.622 to 0.636, the best slot head so far. Medium
  and large recall are unchanged (0.61 and 0.80), and that is most of the remaining gap
  to Moondream 3.1 (0.77 and 0.89).
- **Size-specialised slot groups did not help.** With everything else equal, the
  grouped run is behind the single group on F1 (0.549 against 0.590) and on small
  objects (0.207 against 0.239).

### Loss and capacity ablations

Each run changes one thing against `slots-small-16` (46k images, 48 slots, 20,000 steps),
plus one run with all three changes. COCO val, the same 500 images, each head at the
objectness threshold chosen on its held-out images; L40S:

| Run | Change | F1, IoU 0.5 | F1, IoU 0.75 | Precision | Recall | Recall small / medium / large | Seconds per image |
|---|---|---|---|---|---|---|---|
| `slots-small-16` | none | 0.636 | 0.506 | 0.720 | 0.570 | 0.26 / 0.61 / 0.80 | 0.089 |
| `slots-rank64` | adapter rank 64 (50M parameters) | **0.657** | 0.518 | 0.735 | **0.595** | 0.29 / 0.64 / 0.81 | 0.087 |
| `slots-iou` | objectness trained toward the matched box's IoU | 0.631 | | 0.739 | 0.550 | 0.19 / | |
| `slots-near` | "nothing here" only for unmatched slots overlapping a labelled object | 0.568 | | 0.539 | 0.600 | 0.34 / | |
| `slots-combined` | all three | 0.631 | **0.556** | 0.789 | 0.526 | 0.16 / 0.53 / 0.84 | 0.089 |

- A larger adapter rank raises precision and recall together and costs no measurable
  time, since the adapters act only at slot positions.
- Training objectness toward box quality tightens boxes (F1 at IoU 0.75 is highest with
  it) but loses small objects.
- Without a "nothing here" signal away from labelled objects, recall rises and precision
  falls by 18 points: on COCO's incomplete labels the signal suppresses junk boxes as well
  as real unlabelled objects.

These evaluations ran at threshold 0.3 with each box's confidence recorded, and are
rescored at other thresholds with `python -m parallel_detection.run score --min-score`.

### Training on Gemini 4 Carbon labels

`configs/teacher-train.json` labelled 21,187 of the prepared images (13,800 Open Images,
7,387 Objects365) with Gemini 4 Carbon for $119.54, 8.2 objects per image; the budget
ended before the planned 9,200 Objects365 images. After the label filter, 17,700 images
have usable labels. COCO val, the same 500 images, L40S, each head at its held-out
threshold:

| Run | Training images | F1, IoU 0.5 | F1, IoU 0.75 | Precision | Recall | Recall small / medium / large |
|---|---|---|---|---|---|---|
| `slots-sizes-none`: COCO labels, rank 16, 10,000 steps | 23k COCO | 0.590 | 0.469 | 0.692 | 0.514 | 0.24 / 0.52 / 0.75 |
| `slots-teacher-23k`: same settings | 18k Gemini-labelled | 0.559 | 0.445 | 0.639 | 0.496 | 0.22 / 0.52 / 0.71 |
| `slots-teacher-23k-near`: same, near-label negatives | 18k Gemini-labelled | 0.520 | 0.402 | 0.556 | 0.488 | 0.25 / 0.53 / 0.66 |
| `slots-rank64`: COCO labels, rank 64, 20,000 steps | 46k COCO | 0.657 | 0.518 | 0.735 | 0.595 | 0.29 / 0.64 / 0.81 |
| `slots-teacher-coco-rank64`: same settings | 46k COCO + 18k Gemini-labelled | **0.673** | **0.533** | **0.743** | **0.614** | 0.31 / 0.65 / 0.85 |
| Moondream 3.1 | | 0.700 | 0.560 | 0.682 | 0.719 | 0.47 / 0.77 / 0.89 |

- Adding the Gemini-labelled images to COCO's raises precision and recall together and
  halves the gap to Moondream 3.1, at the same 0.088 s per image on an L40S.
- On their own they score below COCO's labels on COCO val. Scored against Gemini 4
  Carbon's own labels of 200 COCO val images instead, the two heads tie (F1 0.662 each),
  so the gap is not COCO's labelling style. These runs change the images as well as the
  labels, and have fewer of them, so they do not isolate label quality.
- Near-label negatives lower precision on the more complete labels too.

`slots-teacher-23k` was preempted at step 7,000 and resumed from its checkpoint.

### Where the time goes

`slots-bins-16` on the same 500 images:

| GPU, setting | F1, IoU 0.5 | Seconds per image | Vision encoder | Image prefix | Slot pass |
|---|---|---|---|---|---|
| L4 | 0.623 | 0.227 | 0.138 | 0.046 | 0.042 |
| L40S | 0.622 | 0.078 | 0.046 | 0.016 | 0.016 |
| L40S, one image crop | 0.607 | 0.046 | 0.016 | 0.014 | 0.016 |
| L40S, `slots-small-16` | 0.636 | 0.074 | 0.044 | 0.016 | 0.014 |
| Moondream 3.1, Photon, L40S | 0.700 | 0.208 | | | |

On the GPU Moondream 3.1 was measured on, the slot heads take 0.074 to 0.078 s per
image against 0.208 s. The slot pass is a fifth of that time, so fewer positions per slot
would save little; the vision encoder is the largest stage, and limiting it to one crop
costs 1.5 points of F1 with a head trained on full-crop features.

`scripts/figures.py` draws the training curves, accuracy against latency per GPU,
recall by object size and side-by-side example images from downloaded run outputs; the rendered figures are in
the archive under `figures/`.

## Teacher labels from other models

COCO's human labels miss objects, and a slot that finds an unlabelled one is trained as
wrong. `src/parallel_detection/teacher.py` asks vision-language models on OpenRouter for
every instance of the 80 COCO categories (Gemini-style `box_2d`, 0 to 1000), records each
reply with its cost, stops at a dollar budget, and scores each model and multi-model
consensus against COCO's labels with the evaluation harness.

Pilot `teacher-pilot`: twelve models on 200 of the same COCO val images as the
baselines, at low reasoning effort, all with the same Gemini-style prompt; $14.19 in
total. Each model is read in whichever coordinate order scores better, since some answer
x-first regardless of the prompt. Scores are against COCO's labels, so a correct box on
an unlabelled object counts as a false positive:

| Labeller | Precision | Recall | F1, IoU 0.5 | F1, IoU 0.75 | Recall small / medium / large | Cost per image |
|---|---|---|---|---|---|---|
| `google/gemini-4-carbon` | 0.775 | 0.738 | 0.756 | 0.646 | 0.43 / 0.82 / 0.90 | $0.0051 |
| `google/gemini-4-argon` | 0.788 | 0.679 | 0.730 | 0.636 | 0.34 / 0.75 / 0.87 | $0.0049 |
| `anthropic/claude-opus-5.5` | 0.727 | 0.651 | 0.687 | 0.509 | 0.33 / 0.69 / 0.86 | $0.0073 |
| `anthropic/claude-fable-5.2` | 0.704 | 0.670 | 0.687 | 0.554 | 0.34 / 0.73 / 0.87 | $0.0147 |
| `qwen/qwen3.8-max-0902`, read x-first | 0.681 | 0.589 | 0.632 | 0.505 | 0.24 / 0.67 / 0.78 | $0.0047 |
| `google/gemini-4-hydrogen` | 0.806 | 0.431 | 0.562 | 0.491 | 0.19 / 0.41 / 0.64 | $0.0077 |
| `anthropic/claude-sonnet-5.5` | 0.565 | 0.409 | 0.475 | 0.279 | 0.10 / 0.38 / 0.67 | $0.0033 |
| `google/gemini-3.8-flash` | 0.877 | 0.308 | 0.456 | 0.411 | 0.13 / 0.27 / 0.48 | $0.0018 |
| `openai/gpt-6.1-sol` | 0.425 | 0.492 | 0.456 | 0.322 | 0.24 / 0.49 / 0.68 | $0.0033 |
| `openai/gpt-6-astra` | 0.417 | 0.494 | 0.452 | 0.326 | 0.26 / 0.48 / 0.68 | $0.0162 |
| `openai/gpt-6.1-terra` | 0.519 | 0.387 | 0.444 | 0.296 | 0.17 / 0.40 / 0.53 | $0.0015 |
| `qwen/qwen3.8-flash` | 0.372 | 0.348 | 0.360 | 0.242 | 0.15 / 0.35 / 0.50 | $0.0005 |
| Two of Carbon, Argon, Opus 5.5, Fable 5.2 agree | 0.792 | 0.737 | 0.764 | 0.652 | 0.40 / 0.82 / 0.92 | $0.0320 |
| Moondream 3.1, same images | 0.715 | 0.739 | 0.727 | | | |
| Slot head `slots-small-16`, same images | 0.720 | 0.586 | 0.646 | | | |

- Gemini 4 Carbon alone is the best labeller: as complete as Moondream 3.1 and more
  precise. Agreement among the top four adds 0.8 points of F1 for six times the cost.
- Qwen 3.8 Max answers `[x, y, x, y]` although the prompt asks for y first; read
  y-first it scores F1 0.118.
- The prompt and box convention are Gemini's own, and every model ran at low reasoning
  effort, so the GPT and Claude rows are not their best possible results.
- On 100 of the prepared Open Images training images, Carbon cost $0.0052 per image and
  found 6.8 objects per image.

The training images are 30,000 from Open Images (`vikhyatk/openimages-bbox`, 30 seeded
shards) and 20,000 from Objects365 (`guozonghao96/objects365`, every tenth image of five
patches), prepared by `src/parallel_detection/images.py` with `configs/teacher-images.json`.

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
.venv/bin/modal run --detach modal_app.py::slots --run-id slots-bins-16 \
  --config configs/slots-16-shards.json
.venv/bin/modal run --detach modal_app.py::main --run-id slots-bins-16-coco \
  --config configs/slots-coco.json --slots-run slots-bins-16
.venv/bin/python scripts/figures.py --artifacts artifacts --out artifacts/figures

# Recompute scores from saved predictions.
.venv/bin/python -m parallel_detection.run score \
  --predictions artifacts/moondream2-001/predictions.jsonl
```

Results land on the Modal volume `lm-exploration-parallel-detection` (`evaluations/<run
id>/`, `slots/<run id>/`) and the launching command saves the summary under
`artifacts/<run id>/`. A local run needs the `model` extra and a GPU.

Every Modal call has `retries=0`. Evaluations stop within one hour of launch and
training within five; a separate CPU supervisor, built on
`training/workflows/deadline.py`, cancels a call still queued or running at its
deadline. A preempted evaluation continues from the images it recorded, and preempted
training from its last held-out evaluation.

## Open work

- Recall on medium and large objects: Gemini 4 Carbon labels of COCO's own training
  images (isolating label quality), more labelled images, and adapting more of the
  backbone.
- Speed: train the head on single-crop features; fold a slot's box and label into fewer
  positions; time the vision encoder under the custom kernels.
- Prompted detection (a slot points at a phrase in the prompt), verified-absent
  categories in training, and labels with several valid wordings as the real test of
  one-pass diffusion.
- RefCOCO-M and LVIS for the slot head; it is evaluated on COCO only.
- Kernels: an FP32 PyTorch reference check and timing on the GPU Photon was measured on.
- Objects365 as training data: `jxu124/objects365` holds annotations only, so its images
  need a prepared subset.
