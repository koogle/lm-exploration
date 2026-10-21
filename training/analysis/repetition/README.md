# Repetition analysis

Low-temperature sampling study of the Simple LLM 124M / 8K checkpoints. The report,
plots, completions and result JSONs are published in the model repository
`jfrick/scholarly-code-modern-gpt-124m-step2` under `analysis/repetition/`.

Run from this directory with Modal credentials and access to the project output volume.
Outputs go to the ignored `training/artifacts/repetition/` by default.

```sh
modal run --detach modal_low_temperature.py
# Download previous-comparison-20261018.json from the model repository into the same folder.
python build_report.py ../../artifacts/repetition
```

The report builder needs matplotlib.
