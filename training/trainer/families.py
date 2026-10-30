"""Model families for the post-training stages, found from a checkpoint's architecture.

Every checkpoint records `architecture = {module, config}` (`trainer/checkpoints.py`).
The module names the family; `adapter(module)` gives its post-training adapter, defined
beside the family's pretraining adapter:

- `model_100m.model`: `trainer.subword.SubwordModel` (subword GPT, subword tokenizer).
- `model_hybrid.model`: `model_hybrid.post_train.HybridModel` (hybrid LM, GLM tokenizer).

An adapter gives the stages what depends on the family: building the model from its
config and loading a parent's weights, the tokenizer (object, SHA256, files to copy),
prompt encoding, the tokens that end an answer, sampling, the answer format the rewards
read, the optimizer, the source files a run hashes and copies, and exports.
"""

import importlib

ADAPTERS = {
    "model_100m.model": ("trainer.subword", "SubwordModel"),
    "model_hybrid.model": ("model_hybrid.post_train", "HybridModel"),
}


def adapter(module):
    """The post-training adapter of the family whose model lives in `module`."""
    if module not in ADAPTERS:
        raise ValueError(f"No post-training adapter for model module {module}")
    package, name = ADAPTERS[module]
    return getattr(importlib.import_module(package), name)()
