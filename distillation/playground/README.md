# Student playground

A Gradio app for trying a distillation student with your own prompts, published to a
private Hugging Face Space. The current Space,
[`jfrick/distill-student-150m-playground`](https://huggingface.co/spaces/jfrick/distill-student-150m-playground),
serves `trim-large-500m-kd` from campaign `distill-validation-20261023-01`
(149.7M backbone parameters, 500M training tokens with KD; 62.8% pass@1 on held-out
curriculum problems and 13.6% on MBPP, see `distillation/validation/README.md`).

- **Solve a problem** wraps the text in the training format (`{"problem": ..., "code":
  "`) and streams the generated `solve` code, optionally followed by the explanation.
- **Free continuation** continues any raw prefix; the default `{"problem": "` makes the
  student invent an exercise.

Temperature 0 is greedy decoding, as in the evaluation. `app.py` runs the
`training/10m/model.py` decoder on CPU with a per-layer key/value cache; on the Space's
two vCPUs it generates about 11–17 tokens/s. With greedy decoding it reproduced the
evaluated answer of a held-out problem exactly.

## Publish

From the repository root, with `HF_TOKEN` able to write to the Space and read the
private checkpoint and shard datasets:

```sh
python distillation/playground/publish.py --space jfrick/distill-student-150m-playground \
  --campaign distill-validation-20261023-01 --student trim-large-500m-kd
```

The script downloads the checkpoint, tokenizer and vocabulary into the ignored
`artifacts/` folder, assembles the Space (app, `space-card.md` as its card, the model and
prompt code copied from `training/10m/model.py` and `distillation/validation/evalsets.py`)
and uploads it. `--stage-only` assembles it locally without uploading, for testing.
