# Parallel detection

Post-training a small vision-language model so that it detects every requested
object in one forward pass, with a box and a short label per detection.

**Status:** the evaluation harness and the Moondream 2 baseline run are implemented.
The detection head described under [Design under test](#design-under-test) is a plan;
no training code exists yet.

## Question

A vision-language model that writes detections one after another pays decoder steps
per object and one prompt per category. Can post-training alone give the same model
a set-prediction output (n slots, each "nothing" or a box with a label) so that
latency stops growing with the number of objects and phrases, without losing accuracy?

## Baseline: Moondream 2

`vikhyatk/moondream2` revision `2025-06-21` (Apache-2.0, 1.9B parameters). Read from
its Hub code:

- `detect(image, phrase)` takes one phrase per call. The image is encoded once and
  its key/value cache is reused across phrases.
- Each object costs three decoder steps: x centre, y centre, then width and height,
  read from a region head as 1,024-bin distributions, not as text.
- Decoding ends at end-of-sequence or `max_objects` (the model's default is 50; the
  baseline config raises it to 150). Boxes carry no confidence.

The cost model to beat is therefore one image encode, plus one prefill per phrase,
plus three decoder steps per box.

## Evaluation protocol

Every task has the same layout (`src/parallel_detection/tasks.py`): an image and a
list of phrases, each with its truth boxes.

| Dataset | Phrases per image | Source |
|---|---|---|
| LVIS v1 minival | Every annotated category, plus up to five categories LVIS verified absent | `GLIPModel/GLIP` annotation file; COCO images |
| RefCOCO val, RefCOCOg val | One expression per sampled referred object | `jxu124/refcoco`, `jxu124/refcocog`; COCO images |

Scores (`src/parallel_detection/metrics.py`):

- **Detection:** micro precision, recall and F1 at IoU 0.5 and 0.75, using the largest
  one-to-one matching per phrase. Boxes for an absent category are false positives.
  For categories LVIS marks as not exhaustively annotated, unmatched boxes are not
  counted as errors. Recall is broken down by box size, by instances per phrase and
  by LVIS frequency class.
- **Referring:** accuracy of the first box at IoU 0.5 and 0.75.
- **Latency:** seconds per image and per phrase, and phrase seconds against boxes
  emitted (bucket means and a linear fit). The first two images are warm-up.

Average precision is not reported: it ranks detections by confidence, which this
baseline does not produce. A model with per-slot objectness can add it.

The image subset is a seeded sample (300 LVIS images, 300 referred objects per
referring set in `configs/baseline.json`). Latency is specific to the GPU recorded in
each summary.

## Run

From `exploration/parallel-detection/`:

```sh
uv venv --python 3.12
uv pip install -e ".[dev,modal]"
.venv/bin/pytest
.venv/bin/ruff check . && .venv/bin/ruff format --check .

# One L4 for at most an hour; writes artifacts/<run id>/summary.json.
.venv/bin/modal run modal_app.py --run-id baseline-001
.venv/bin/modal volume get lm-exploration-parallel-detection \
  evaluations/baseline-001/predictions.jsonl artifacts/baseline-001/

# Recompute scores from saved predictions.
.venv/bin/python -m parallel_detection.run score \
  --predictions artifacts/baseline-001/predictions.jsonl
```

A local run needs the `model` extra and a GPU (`python -m parallel_detection.run
evaluate --config configs/baseline.json --out artifacts/local/run`).

The Modal call has `retries=0` and a one-hour timeout, bounded further by the launch
deadline. A separate CPU supervisor, built on `training/workflows/deadline.py`,
cancels the call if it is still queued or running one hour after launch.

## Design under test

Append n learned query embeddings after the image and the prompt and run the language
model once. A head shared by all slots reads each slot's hidden state and outputs:

- an objectness logit (the "nothing" option),
- a box,
- a label, within a fixed token budget.

Training matches slots to truth boxes with Hungarian matching. Prompts mix present and
verified-absent categories, as in the evaluation protocol, so the model learns to leave
slots empty for a named category that is not in the image.

Label options, in the order they will be tried:

1. **Pointer.** The slot embedding is scored against the prompt's phrase embeddings.
   No tokens are generated; it covers prompted detection.
2. **Masked label block.** Each slot owns T mask positions that are filled in the same
   forward pass. This is one step of masked discrete diffusion, the mechanism of the
   [DFlash-style drafter](../latent-workspace/docs/implementing-a-diffusion-style-drafter-from-scratch-and-learnings.md).
   One step predicts the T tokens independently, so labels with several valid wordings
   can blend. More steps repair that by re-running only the slot block against the
   cached image and prompt.
3. **Batched short decode.** Box, objectness and first token in one pass, then T-1
   steps batched across the non-empty slots.

Comparisons use the baseline's protocol and add latency at fixed object counts.

## Open work

- Objects365 validation as a dense, exhaustively annotated task. `jxu124/objects365`
  holds annotations only; the images need a prepared subset.
- Training data assembly (Objects365 and LVIS train, a referring-expression set).
- The slot head and its training, as a post-training change to Moondream 2.
