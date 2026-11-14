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

Precision against recall on the same images, with boxes ranked by recorded confidence
(`metrics.ranked_curve` and `ranked_ap`; all-point interpolation, pooled over phrases
and averaged per category). The evaluations recorded boxes with confidence at least 0.3,
so the curves stop there and the areas are lower bounds. Moondream 2 and 3.1 give no
confidence, so they have one operating point each and no area:

| Run | Area, IoU 0.5, pooled | Per-category mean, IoU 0.5 | IoU 0.75 | IoU 0.5 to 0.95 | Recall at precision 0.743 | Recall at precision 0.682 |
|---|---|---|---|---|---|---|
| `slots-small-16` (rank 16, 46k COCO) | 0.596 | 0.607 | 0.455 | 0.434 | | |
| `slots-rank64` (46k COCO) | 0.626 | 0.634 | 0.475 | 0.449 | 0.589 | 0.625 |
| `slots-teacher-coco-rank64` (+ 18k Gemini-labelled) | **0.639** | **0.647** | **0.482** | **0.458** | **0.614** | **0.647** |
| `slots-teacher-23k` (18k Gemini-labelled only) | 0.521 | 0.534 | 0.404 | 0.377 | | |

At Moondream 3.1's own precision (0.682), the best slot head reaches recall 0.647
against 3.1's 0.719.

### Scoring slots: duplicates and confidence

`scripts/diagnose_slots.py` reads an evaluation that recorded every slot (`--all-slots`)
and follows each COCO object. For `slots-teacher-coco-rank64` at its objectness
threshold of 0.9, of 3,603 objects on the 500 images: 62% found; 14% only loosely boxed
(IoU 0.1 to 0.5, 294 of them small); 13% boxed and labelled correctly but below the
threshold; 8% with no box near them (249 small); 3% boxed with a wrong label, mostly
close categories (cup and bowl, truck and car). No image has more objects than slots.

The slot detector now drops boxes overlapping a higher-scoring box of the same label by
IoU 0.7, and ranks slots by objectness times the sampler's confidence in the label and
box. Its threshold is chosen on 500 further COCO val images that do not overlap the 500
reported (`configs/slots-coco-tune.json`). Same heads, every box recorded, L40S:

| Head and scoring | F1, IoU 0.5 | F1, IoU 0.75 | Precision | Recall | Area, pooled | Per-category mean, IoU 0.5 / 0.75 / 0.5 to 0.95 | Recall at precision 0.682 |
|---|---|---|---|---|---|---|---|
| `slots-rank64`, objectness, duplicates kept | 0.657 | 0.521 | 0.753 | 0.583 | 0.633 | 0.640 / 0.483 / 0.455 | 0.625 |
| `slots-rank64`, combined score, duplicates removed | 0.691 | 0.558 | 0.759 | 0.635 | 0.637 | 0.659 / 0.510 / 0.480 | 0.675 |
| `slots-teacher-coco-rank64`, objectness, duplicates kept | 0.674 | 0.535 | 0.761 | 0.604 | 0.646 | 0.654 / 0.488 / 0.463 | 0.647 |
| `slots-teacher-coco-rank64`, combined score, duplicates removed | **0.694** | **0.562** | 0.757 | **0.640** | 0.645 | **0.666 / 0.514 / 0.486** | **0.684** |
| Moondream 3.1 | 0.700 | 0.560 | 0.682 | 0.719 | | | |

- Re-scoring moves the curve's middle and lower end outward and lowers precision at low
  recall, so the pooled area is unchanged while the per-category areas, recall at a fixed
  precision and F1 at the tuned threshold all rise.
- At IoU 0.75 the best slot head now matches Moondream 3.1 (0.562 against 0.560), and its
  recall on large objects equals 3.1's (0.89). The remaining gap is small and medium
  objects.

### Adapter rank 128

`slots-teacher-coco-rank128`: the data and settings of `slots-teacher-coco-rank64` with
rank-128 adapters (101M trainable parameters), 3.6 hours on one H100. Scored as above:

| Run | F1, IoU 0.5 | F1, IoU 0.75 | Precision | Recall | Area, pooled | Per-category mean, IoU 0.5 / 0.75 / 0.5 to 0.95 | Recall at precision 0.682 | Seconds per image, L40S |
|---|---|---|---|---|---|---|---|---|
| `slots-teacher-coco-rank64` | 0.694 | 0.562 | 0.757 | 0.640 | **0.645** | **0.666 / 0.514 / 0.486** | 0.684 | 0.091 |
| `slots-teacher-coco-rank128` | **0.700** | **0.566** | **0.763** | **0.647** | 0.641 | 0.665 / 0.508 / 0.482 | **0.686** | 0.091 |
| Moondream 3.1 | 0.700 | 0.560 | 0.682 | 0.719 | | | | 0.208 |

Rank 128 reaches Moondream 3.1's F1 at its tuned threshold and its IoU 0.75 F1 is higher,
but the areas are slightly lower than at rank 64: going from rank 64 to 128 does not move
the curve, unlike going from 16 to 64.

### Self-training

`self_ignore` keeps a slowly averaged copy of the head (decay 0.999). From step 3,000, an
unmatched slot where that copy sees a COCO category with objectness at least 0.8 gets no
"nothing here" gradient; 10 to 13% of unmatched slots were ignored by the end. Same
settings as the run each is paired with, scored as above:

| Run | F1, IoU 0.5 | F1, IoU 0.75 | Precision | Recall | Area, pooled | Per-category mean, IoU 0.5 to 0.95 | Recall at precision 0.682 |
|---|---|---|---|---|---|---|---|
| `slots-rank64` | 0.691 | 0.558 | 0.759 | 0.635 | 0.637 | 0.480 | 0.675 |
| `slots-rank64-self` | 0.683 | 0.545 | 0.714 | 0.654 | 0.633 | 0.477 | 0.675 |
| `slots-teacher-coco-rank64` | 0.694 | 0.562 | 0.757 | 0.640 | 0.645 | 0.486 | 0.684 |
| `slots-teacher-coco-rank64-self` | 0.685 | 0.558 | 0.749 | 0.631 | 0.630 | 0.485 | 0.673 |

Self-training does not move the curve outward on COCO val: on COCO labels alone it trades
precision for recall at the tuned threshold, and with the Gemini-labelled images it
loses slightly on every measure.

The 21,187 labelled images are in the private Hugging Face dataset
[`jfrick/coco80-teacher-labels-gemini-4-carbon`](https://huggingface.co/datasets/jfrick/coco80-teacher-labels-gemini-4-carbon),
written by `scripts/export_teacher_dataset.py`.

`slots-teacher-23k` was preempted at step 7,000 and resumed from its checkpoint.

### More teacher-labelled images: Gemini 4 Argon

`configs/teacher-train-argon.json` labelled further prepared images, none of them seen by
Carbon, with Gemini 4 Argon (the second-best labeller in the pilot): 20,841 images for
$103.14. `slots-teachers-14k-coco-rank64` adds the first 14.7k of them to the data of
`slots-teacher-coco-rank64` (76,204 training images after the label filter), and
`slots-teachers-coco-rank64` all of them (81,967); same settings, 3.1 and 3.7 hours on
one H100. Scored as above, each at the threshold chosen on the tuning split:

| Run | F1, IoU 0.5 | F1, IoU 0.75 | Precision | Recall | Area, pooled | Per-category mean, IoU 0.5 / 0.75 / 0.5 to 0.95 | Recall at precision 0.682 |
|---|---|---|---|---|---|---|---|
| `slots-teacher-coco-rank64` | **0.694** | **0.562** | **0.757** | 0.640 | **0.645** | **0.666 / 0.514 / 0.486** | 0.684 |
| `slots-teachers-14k-coco-rank64` | 0.693 | 0.559 | 0.755 | 0.640 | 0.642 | 0.661 / 0.501 / 0.479 | 0.684 |
| `slots-teachers-coco-rank64` | 0.684 | **0.574** | **0.788** | 0.604 | 0.637 | 0.652 / 0.512 / 0.475 | 0.671 |

The first 14.7k Argon-labelled images do not move the curve: every measure is within a
point, and recall by size is unchanged (0.31 / 0.68 / 0.89). With all 20.8k the head
becomes more conservative: its tuned threshold rises from 0.0133 to 0.0152, precision
rises by 3 points and F1 at IoU 0.75 is the highest of any head (0.574), but recall falls
by 3.6 points, mostly on small objects (0.31 to 0.25), and the curve's area falls.
Argon labels fewer objects than Carbon (6.1 against 8.2 per image; recall 0.679 against
0.738 in the pilot), so its images teach the head to call more small objects "nothing".
The first 18k teacher-labelled images added 1.7 points of F1 (0.657 to 0.673, before
re-scoring); a less complete labeller adds nothing or less.

The Argon-labelled images are in the private Hugging Face dataset
[`jfrick/coco80-teacher-labels-gemini-4-argon`](https://huggingface.co/datasets/jfrick/coco80-teacher-labels-gemini-4-argon),
none of them in the Carbon dataset.

#### Why the Argon labels did not help

`scripts/diagnose_slots.py --compare` follows every COCO object of the 500 test images
through both heads, each at its own tuned threshold and score
(`slots-teachers-coco-rank64-coco-v2/diagnosis.json` in the archive):

| Fate of a COCO object | COCO + Carbon head | + all 20.8k Argon images |
|---|---|---|
| Found | 2,319 | 2,189 |
| Boxed and labelled correctly, scored below the threshold | 390 | 503 |
| Only loose boxes (IoU 0.1 to 0.5) | 516 | 532 |
| No box near it | 274 | 278 |
| Wrong label | 92 | 95 |

- Localisation is unchanged. The loss is in scoring: 146 objects go from found to
  "below threshold" and 41 the other way, 56 of the 146 small.
- Two-thirds of it is the threshold. The tuning split picks 0.0152 for the Argon head
  instead of 0.0133; at 0.0133 it scores F1 0.687 with recall 0.628, against 0.694 and
  0.640. Its curve is also slightly lower (recall 0.671 against 0.684 at precision
  0.682).
- For objects both heads box correctly, the confidences match (median objectness 0.95,
  0.98 and 0.99 for small, medium and large objects in both). The combined score is
  dominated by the box confidence, a median of 0.02 to 0.08, which is why thresholds
  sit near 0.013.
- The labels differ where recall was lost. `scripts/label_stats.py` counts objects per
  image by box area: on comparable images Argon labels 30 to 47% fewer objects under
  0.3% of the image, and nearly as many large ones:

| Labeller, source | Images | Objects per image | Under 0.3% of the image | 0.3 to 1% | 1% and over |
|---|---|---|---|---|---|
| Carbon, Open Images | 13,800 | 6.64 | 1.38 | 1.57 | 3.69 |
| Argon, Open Images | 16,199 | 5.65 | 0.95 | 1.23 | 3.47 |
| Carbon, Objects365 | 7,387 | 11.22 | 2.33 | 3.06 | 5.83 |
| Argon, Objects365 | 4,642 | 7.73 | 1.23 | 1.62 | 4.87 |

  On the 200 pilot images both labelled, Argon's recall of COCO's small objects is 0.34
  against Carbon's 0.43. Unlabelled small objects become "nothing here" targets, which
  lowers the head's confidence on real ones.

### Pascal VOC 2007 test

A dataset none of these models was trained or tuned on: 1,000 seeded images of the
4,952 in VOC 2007 test (`davin11/VOC2007`, `configs/voc-*.json`), 1,548 phrases and
2,978 boxes, the 20 VOC categories prompted by their COCO names (`tasks.VOC_TO_COCO`).
Objects VOC marks difficult count as truth. Slot heads use the threshold tuned on COCO.
IoU 0.5, one L40S per model:

| Model | Precision | Recall | F1 | F1, IoU 0.75 | Recall small / medium / large | Area, pooled | Seconds per image |
|---|---|---|---|---|---|---|---|
| Moondream 2 `detect`, Photon | 0.787 | **0.927** | 0.851 | 0.703 | 0.72 / **0.90** / **0.98** | | 0.088 |
| Moondream 3.1, Photon | **0.806** | 0.915 | **0.857** | **0.727** | **0.74** / 0.88 / 0.97 | | 0.095 |
| `slots-rank64` | 0.781 | 0.892 | 0.833 | 0.714 | 0.59 / 0.85 / 0.97 | 0.878 | **0.051** |
| `slots-teacher-coco-rank64` | 0.745 | 0.905 | 0.817 | 0.696 | 0.69 / 0.86 / 0.97 | 0.874 | 0.064 |
| `slots-teacher-coco-rank128` | 0.769 | 0.906 | 0.832 | 0.710 | 0.64 / 0.88 / 0.97 | **0.884** | 0.063 |
| `slots-teachers-coco-rank64` | 0.765 | 0.897 | 0.826 | 0.714 | 0.60 / 0.87 / 0.97 | 0.869 | 0.066 |

- On VOC the slot heads trail both Moondream models by 2.4 to 4.0 points of F1; paired
  bootstrap intervals over images exclude zero for every head. With a threshold tuned on
  VOC itself (an oracle), the best head reaches 0.851. At Moondream 3.1's precision the
  best head recalls 0.895 against 0.915.
- The Gemini-labelled images, which helped on COCO, do not help here: the COCO-only head
  is as good as any. The gain on COCO may partly be COCO-specific.
- The gap is small objects (275 of the 2,978 boxes) and precision. Of the extra false
  positives, near-duplicates are the clearest: 51 for the rank-128 head against 6 for
  Moondream 3.1, because same-label duplicates are removed only above IoU 0.7. Removing
  them above IoU 0.5 instead raises F1 on the COCO tuning split (0.672 to 0.676), on
  COCO test (0.694 to 0.699; rank 128: 0.700 to 0.705) and on VOC (0.817 to 0.825;
  rank 128: 0.832 to 0.838). The detector's default is still 0.7.
- 14 of 42 wrong labels are VOC cars the head calls trucks: VOC's car includes vans and
  pickups, which COCO labels truck.
- VOC images hold 1.5 categories on average, so Moondream's per-category cost is small
  here and the slot heads are only 1.4 to 1.9 times faster; the speed advantage grows
  with the number of categories asked for (see Demo).
- Moondream 2 scores far better on VOC than on COCO (F1 0.851 against 0.587): VOC
  objects are larger and fewer per image.

### Small objects, tiles and duplicates

Recall by object size measured in cells of the 27 x 27 image grid that every model here
reads (one cell is about 24 x 18 pixels of a COCO image), on the 500 COCO test images:

| Object size | Objects | Moondream 2 | Moondream 3.1 | `slots-teacher-coco-rank128`, before | Same head, tiled |
|---|---|---|---|---|---|
| Under 0.5 cell | 175 | 0.20 | 0.18 | 0.09 | |
| 0.5 to 1 cell | 445 | 0.42 | 0.43 | 0.26 | |
| Under 1 cell, together | 620 | 0.36 | 0.36 | 0.22 | 0.37 |
| 1 to 4 cells | 1,527 | 0.70 | 0.72 | 0.61 | 0.70 |
| 4 cells and over | 1,456 | 0.87 | 0.87 | 0.88 | 0.84 |

Moondream reads the same pooled grid and finds more small objects, so the features hold
them; re-weighting the score alone does not recover them. The detector can therefore
also run the head on a 2 x 2 grid of tiles overlapping by a quarter, in one batch with
the full image (`SlotDetector(tiles=2)`), as in slicing-aided inference (Akyon et al.,
ICIP 2022). Every slot of every view becomes a candidate, and `merge.select` keeps tile
boxes that are not cut by a tile edge and are smaller than `tile_max_cells`, then
merges same-label boxes. `scripts/study_selection.py` compares the choices from one
evaluation per head that recorded every candidate (`*-tiles2`), each with its threshold
chosen on the COCO tuning split (`*-selection.json` in the archive):

| `slots-teacher-coco-rank128` | Tuning F1 | COCO F1 | COCO recall | F1, IoU 0.75 | VOC F1 | Seconds per COCO image, L40S |
|---|---|---|---|---|---|---|
| Before: no tiles, suppression at IoU 0.7, box confidence | 0.670 | 0.700 | 0.651 | 0.564 | 0.829 | 0.091 |
| No tiles, fusion at IoU 0.5, box confidence^0.25 | 0.681 | 0.707 | 0.626 | 0.569 | **0.858** | 0.09 |
| Tiles under 4 cells, fusion at IoU 0.5, box confidence^0.25 | **0.701** | **0.720** | **0.699** | **0.580** | 0.846 | 0.229 |
| Moondream 3.1 | | 0.700 | 0.719 | 0.560 | 0.857 | 0.208 |

- Tiles bring small-object recall to Moondream's level and COCO F1 above Moondream
  3.1's. On VOC, whose objects are larger, they cost precision.
- Duplicates: merging same-label boxes from IoU 0.5 instead of 0.7 is most of the gain.
  Weighted box fusion (Solovyev et al., 2021), which averages a group's boxes by score,
  ties suppression without tiles and is slightly ahead with them (tuning F1 0.701
  against 0.700), where the same object appears in several views.
- Box confidence to the power 0.25 lets the threshold move from about 0.013 to about
  0.3 and transfers better from COCO to VOC.
- These are now the detector's defaults; `tiles=1` is the fast mode. A full evaluation
  with the defaults (`slots-teacher-coco-rank128-coco-final`, `-voc-final`) gives COCO
  F1 0.718 and VOC 0.844, matching the study. All five views share one batch, yet the
  decoder pass over them takes 0.13 s, so tiling costs 2.5 times the fast mode's time.
- VOC vocabulary: VOC's car includes vans and pickups, which COCO labels truck, so on
  VOC a car query also takes the head's truck boxes (`configs/voc-slots.json`). Car
  recall rises from 0.896 to 0.938 and precision falls from 0.779 to 0.748, because
  COCO's truck also covers lorries, which VOC excludes; F1 is unchanged (0.847 and
  0.846 overall). Separating them needs a finer label in training.

The demo offers the rank-128 head in both modes.

### Training for small objects, without COCO, and without denoising

Five runs, each changing one thing against `slots-teacher-coco-rank64` (COCO 46k + 18k
Carbon-labelled images, rank 64, 20,000 steps), plus the COCO-only head for reference.
Each evaluation recorded every candidate from the full image and 2 x 2 tiles
(`*-tiles2`); `scripts/study_selection.py` scored two settings, both with fusion at IoU 0.5
and box confidence^0.25, the threshold chosen on the COCO tuning split
(`*-selection.json`). Recall is by object size in grid cells.

Without tiles:

| Head | Tuning F1 | COCO F1 | COCO P / R | F1, IoU 0.75 | Recall under 1 / 1 to 4 / 4+ cells | VOC F1 | VOC P / R |
|---|---|---|---|---|---|---|---|
| `slots-rank64`: COCO only (46k images) | 0.673 | 0.694 | 0.809 / 0.608 | 0.559 | 0.20 / 0.57 / 0.82 | **0.862** | **0.837** / 0.888 |
| `slots-teacher-coco-rank64`: COCO + Carbon (64k) | **0.678** | **0.705** | 0.779 / 0.644 | 0.555 | 0.23 / 0.61 / 0.85 | 0.840 | 0.782 / 0.908 |
| `slots-teacher-only-rank64`: Carbon only (17.7k) | 0.665 | 0.681 | 0.738 / 0.633 | 0.522 | 0.23 / 0.61 / 0.83 | 0.830 | 0.770 / 0.899 |
| `slots-prompted-teacher-coco-rank64`: category prompts | 0.676 | 0.702 | 0.770 / 0.645 | 0.555 | 0.23 / 0.60 / 0.86 | 0.849 | 0.795 / 0.912 |
| `slots-sizeweight-teacher-coco-rank64`: small objects weigh up to 4x | 0.665 | 0.700 | 0.796 / 0.624 | 0.553 | **0.26** / 0.57 / 0.83 | 0.836 | 0.790 / 0.888 |
| `slots-o365human-coco-rank64`: Objects365 human labels instead of Carbon's | 0.673 | 0.698 | 0.789 / 0.626 | **0.557** | 0.22 / 0.60 / 0.83 | 0.853 | 0.815 / 0.894 |
| `slots-nodenoise-teacher-coco-rank64`: no denoising pass | 0.677 | **0.705** | 0.779 / 0.644 | 0.556 | 0.23 / 0.62 / 0.84 | 0.845 | 0.793 / 0.905 |

With tiles (boxes under 4 cells):

| Head | Tuning F1 | COCO F1 | Recall under 1 / 1 to 4 / 4+ cells | VOC F1 |
|---|---|---|---|---|
| COCO only | 0.702 | 0.708 | 0.35 / 0.68 / 0.82 | 0.846 |
| COCO + Carbon | 0.700 | **0.714** | 0.33 / 0.66 / 0.82 | 0.841 |
| Carbon only | 0.663 | 0.676 | **0.40** / 0.68 / 0.79 | 0.806 |
| Category prompts | 0.696 | 0.708 | 0.33 / **0.69** / 0.84 | 0.832 |
| Size weighting | 0.683 | 0.705 | 0.37 / 0.65 / 0.80 | 0.841 |
| Objects365 human labels | **0.703** | 0.713 | 0.32 / 0.67 / 0.82 | **0.855** |
| No denoising pass | 0.699 | 0.713 | 0.32 / 0.65 / 0.82 | **0.855** |

On 500 COCO and 1,000 VOC images, differences of about a point and a half are within
sampling noise (see the VOC bootstrap above).

- **Denoising adds nothing.** Without the second, partly revealed training pass the head
  scores the same on every measure, on held-out images (F1 0.721 against 0.709) as on
  COCO and VOC, and trains in 2.4 instead of 3.2 hours. Several sampling passes no longer
  work without it (held-out box IoU 0.733 in one pass, 0.665 in three), but one pass is
  all inference uses. The head is a one-pass set predictor.
- **None of the three training changes fixes small objects.** Size weighting adds 3
  points of recall under one cell for a little precision; category prompts and human
  Objects365 labels leave small-object recall unchanged. Tiling at inference remains the
  lever (0.23 to 0.33 to 0.40 under one cell).
- **Category prompts** (the requested names after the image, as Moondream is prompted)
  match the unprompted head on COCO and are within noise on VOC.
- **Without COCO**, Carbon labels alone score below COCO alone on COCO (0.681 against
  0.694) and on VOC (0.830 against 0.862), with 17.7k against 46k images. Adding Carbon
  to COCO helps on COCO but costs VOC precision, and so do Carbon's Objects365 labels
  against Objects365's human ones (VOC 0.840 against 0.853). The best head on VOC is the
  COCO-only one without tiles, at F1 0.862 against Moondream 3.1's 0.857.

The runs, their head snapshots every 5,000 steps, the evaluations and the studies are in
the archive under each run id.

### Video: MOT17

Two MOT17 training sequences (`Lekim89/MOT17`, `configs/mot-*.json`): 02 (600 frames, a
crowded street) and 09 (525 frames), 1080p at 30 fps. Every third frame is a sample
(375 frames), prompted with "person"; MOT's pedestrians, people on vehicles and static
people are the truth (`tasks.mot_samples`). One L40S per model; slot heads at their COCO
thresholds:

| Model | Precision | Recall | F1 | F1, IoU 0.75 | Seconds per frame | F1 on 02 / 09 |
|---|---|---|---|---|---|---|
| Moondream 2 `detect`, Photon | 0.638 | 0.308 | 0.415 | 0.262 | 0.260 | 0.328 / **0.710** |
| Moondream 3.1, Photon | 0.816 | 0.291 | 0.429 | **0.286** | 0.206 | 0.338 / 0.708 |
| `slots-teacher-coco-rank128`, fast | 0.652 | 0.341 | 0.447 | 0.268 | 0.104 | 0.381 / 0.644 |
| `slots-teacher-coco-rank128`, tiled | 0.559 | **0.446** | **0.496** | 0.279 | 0.418 | **0.475** / 0.559 |
| `slots-rank64` (COCO only), fast | **0.851** | 0.241 | 0.376 | 0.275 | **0.087** | 0.272 / 0.681 |

- Asked for one category, the fast head is ahead of Moondream 3.1 on F1 at half the time
  per frame; at 0.10 s per detection, detecting every third frame keeps pace with
  30 fps. With all 80 categories Moondream 3.1 needs 2.79 s per frame (see Demo).
- MOT17-02 has up to 46 small, overlapping people per frame, more than most COCO images,
  and every model's recall there is low. Tiling finds the most of them; on the sparser
  MOT17-09 the Moondream models are ahead and tiling costs precision.

`scripts/figures.py` draws example sheets for COCO, VOC and MOT17 (`examples*.png`) and
the area under the precision-recall curve and per-category AP of every studied head,
fast and tiled (`selection-areas.png`); the paper's appendix shows them.

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

## Demo

`demo_app.py` deploys a web page on Modal for comparing detectors on images (uploaded,
or 16 examples: COCO val images and Objects365 images no labeller saw) and for running
one model on every n-th frame of a video. The annotated video streams back to the page
in two-second H.264 chunks while the GPU works through it. Slot-head boxes are labelled
with their score, objectness × label confidence × box confidence: a ranking score, not
a probability (kept from 0.013; half of the kept boxes on COCO val score under 0.04).
Moondream gives no confidence, so its boxes carry only the label. The models and thresholds are in
`configs/demo.json`: Moondream 2's `detect`, three slot heads and Moondream 3.1 through
Photon. Deploying with `.venv/bin/python demo_app.py` also spawns `shutdown_at`, which
stops the app just under 24 hours later; the URL is public until then.

Measured through the page on one warm L40S per model family:

| Request | Moondream 2 | Slot head, rank 128 | Moondream 3.1 |
|---|---|---|---|
| One image, 6 categories | 0.343 s | 0.081 s | 0.246 s |
| 60-frame clip, all 80 categories, every 3rd frame | | 0.047 s per frame | 2.79 s per frame |

Streaming a 125-frame, 25 fps clip (448 × 576) with the rank-128 head on all 80
categories and every 3rd frame: 0.106 s per detector call, and the stream ran at 0.86×
the video's own speed, detection included; the first chunk arrived 20 s after the request,
most of it GPU start-up.

The video gap is the cost model: Moondream answers one category per pass, the slot head
all categories in one pass.

## Paper

`paper/main.tex` is a two-page summary of the slot head: method, results against
Moondream 2 and 3.1, and what did and did not work. It reads its figure from
`artifacts/figures/`; build it with `latexmk -pdf -outdir=../artifacts/paper main.tex`
from `paper/`. The built PDF is in the archive as `paper/one-pass-detection.pdf`.

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
.venv/bin/python demo_app.py

# Recompute scores from saved predictions.
.venv/bin/python -m parallel_detection.run score \
  --predictions artifacts/moondream2-001/predictions.jsonl
```

Results land on the Modal volume `lm-exploration-parallel-detection` (`evaluations/<run
id>/`, `slots/<run id>/`) and the launching command saves the summary under
`artifacts/<run id>/`. A local run needs the `model` extra and a GPU.

Every Modal call has `retries=0`. Evaluations stop within one hour of launch and
training within eight; a separate CPU supervisor, built on
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
- RefCOCO-M and LVIS for the slot head; it is evaluated on COCO and Pascal VOC only.
- Kernels: an FP32 PyTorch reference check and timing on the GPU Photon was measured on.
- Objects365 as training data: `jxu124/objects365` holds annotations only, so its images
  need a prepared subset.
