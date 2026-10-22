# Model configurations

Active JSON files contain model dimensions, batching and optimizer/schedule settings.
Workflows supply GPU count and total token target; dataset configs supply the loader.

- `modern-10m.json`: eight layers, width 192, three heads; the existing raw 5B model.
- `modern-124m-pilot.json`: the shared one-/two-GPU pilot model and optimizer settings.
- `modern-124m.json`: the larger 20B experiment's model and optimizer settings.

Choose one with a workflow's `model_config` field or the `--model-config` selector.
See the [workflow guide](../README.md) for resolution and launch commands.

`historical/` preserves the former recipe files byte for byte, including the unused
qualification recipe and Shakespeare configs. These are configuration provenance,
not active model-only configs: they contain old execution/dataset fields and are
rejected as model selections. They are not validated against current schemas; the
character recipe schema and checkpoint loader were removed after revision `5083671`.
