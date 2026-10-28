# Model comparison Space

A Gradio app that runs one prompt through several of our checkpoints side by side, on a
private CPU Hugging Face Space. uv installs this folder as `playground`.

| File | Role |
| --- | --- |
| `configs/base-vs-math-sft.json` | Export config: checkpoints and labels, layout, model and Space repositories |
| `modal_export.py` | Exports the configured checkpoints and publishes the Space |
| `app.py` | The Space: model loading and UI; sampling uses the shared [`trainer/rollout.py`](../../trainer/rollout.py), copied into the Space |
| `intro.md`, `space-card.md`, `requirements.txt` | Space text, card and dependencies |

## Publish

From `training/`, logged in to HF with write access:

```sh
uv run --all-extras modal run -m playground.modal_export \
  --config analysis/playground/configs/base-vs-math-sft.json --name EXPORT-NAME
```

1. One Modal CPU call loads each checkpoint with its run's own model code
   (`evaluation.checkpoints`), checks that the config's shared `model_code` reproduces
   its logits exactly, and writes float32 safetensors (the tied output head stored once),
   the tokenizer and `models.json` (provenance and headline scores from the configured
   evaluation results) to `/runs/playground/EXPORT-NAME`.
2. The local entrypoint uploads them to the private model repository `model_repo`.
3. It uploads the app with `models.json` pinned to that upload's commit to the private
   Space `space`, and sets the Space secret `HF_TOKEN` to your token so the Space can
   download the private weights. Replace it with a read-only token in the Space settings
   if you prefer.

Weights live in the model repository because Space repositories hold at most 1 GB.
