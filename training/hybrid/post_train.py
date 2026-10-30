"""The hybrid family for the post-training stages (`trainer/families.py`).

The hybrid LM with the pinned GLM tokenizer. A prompt is one user turn in GLM's chat
template (`chat.render`, generation prompt included) asking for a `\\boxed{}` answer or
one Python code block; an answer ends at any of `sft_data.STOP_IDS` (EOS or the next
turn's role token) and is read in the `boxed` format (`training_data.rl_tasks`).
Sampling is `generate.generate_batch` (one-pass prefill, CUDA-graph decoding) on a bf16
copy of the policy that is refreshed in place before every rollout. Parents may carry
MTP heads; post-training builds the trunk without them. Exports are bf16 trunk weights
(`train.upload_export`), optionally uploaded to an HF model repo.
"""

import dataclasses
from pathlib import Path

import torch

from . import chat, generate, sft_data
from . import tokenizer as glm
from .model import HybridConfig, HybridLM

MATH_INSTRUCTION = "Solve the problem step by step, then give the final answer as \\boxed{answer}."
CODE_INSTRUCTION = (
    "Write the Python {kind} `{declaration}`. Give the complete solution, with any imports it "
    "needs, in one ```python code block."
)


def user_message(row):
    """The pool row as the user's request in the hybrid chat format."""
    if row["task"] == "math":
        return f"{row['question'].strip()}\n\n{MATH_INSTRUCTION}"
    # `def add(a, b):` asks for the function `add(a, b)`, `class Stack:` for the class `Stack`.
    keyword, _, declaration = row["declaration"].partition(" ")
    kind = "class" if keyword == "class" else "function"
    instruction = CODE_INSTRUCTION.format(kind=kind, declaration=declaration.removesuffix(":"))
    return f"{row['question'].strip()}\n\n{instruction}"


class HybridModel:
    name = "hybrid"
    module = "model_hybrid.model"
    answer_format = "boxed"
    # generate.py returns tokens only: no sampler log-probabilities or top tokens.
    records_sampler = False
    exports = True

    def __init__(self):
        self.tokenizer_sha256 = glm.SHA256
        self.special_tokens = {"eos": glm.EOS_ID}
        self.stop_ids = tuple(sft_data.STOP_IDS)
        self.sampler = None

    def tokenizer(self):
        return glm.load()

    def copy_tokenizer(self, folder):
        import shutil

        Path(folder).mkdir(parents=True)
        shutil.copy(glm.tokenizer_file(), Path(folder) / "tokenizer.json")

    def build(self, config):
        values = {k: tuple(v) if isinstance(v, list) else v for k, v in config.items()}
        return HybridLM(HybridConfig(**{**values, "mtp": False}))

    def load(self, model, state):
        model.load_state_dict({k: v for k, v in state.items() if not k.startswith("mtp.")})

    def architecture(self, model):
        return {"module": self.module, "config": dataclasses.asdict(model.config)}

    def context(self, model):
        return model.config.block_size

    def prompt_ids(self, tokenizer, row):
        text = chat.render(
            [{"role": "user", "content": user_message(row)}], add_generation_prompt=True
        )
        return tokenizer.encode(text, add_special_tokens=False).ids

    def sample(self, model, prompts, limit, temperature, generator, record_top_k, engine):
        """(rollout.Samples without the stop token, the stop token of each finished row)."""
        from trainer.rollout import Samples

        if record_top_k:
            raise ValueError("The hybrid sampler records no top tokens")
        if self.sampler is None:
            self.sampler = self.build(dataclasses.asdict(model.config)).cuda().to(torch.bfloat16)
            self.sampler.eval()
        # In place, so the sampler's cached CUDA graphs stay valid.
        self.sampler.load_state_dict(model.state_dict())
        outputs = generate.generate_batch(
            self.sampler, prompts, limit, self.stop_ids, temperature, 1.0, generator
        )
        completions, stops = [], []
        for ids in outputs:
            stop = ids[-1] if ids and ids[-1] in self.stop_ids else None
            completions.append(ids[:-1] if stop is not None else ids)
            stops.append(stop)
        return Samples(completions, [s is not None for s in stops]), stops

    def source_files(self):
        from . import (
            cuda_extension,
            documents,
            fused_loss,
            kda,
            kda_convolution,
            model,
            softmax_kernel,
        )

        modules = (
            chat,
            cuda_extension,
            documents,
            fused_loss,
            generate,
            glm,
            kda,
            kda_convolution,
            model,
            sft_data,
            softmax_kernel,
        )
        files = {f"hybrid_{Path(m.__file__).stem}": Path(m.__file__) for m in modules}
        kernels = Path(cuda_extension.__file__).with_name("kernels")
        files.update({f"hybrid_kernel_{p.stem}": p for p in sorted(kernels.glob("*.cu"))})
        return {"hybrid_post_train": Path(__file__), **files}

    def export(self, model, output, progress, repo):
        """bf16 trunk weights under `output/export/tokens-N/`, and to `repo` if named."""
        from .train import upload_export

        return upload_export(model, model.config, output, progress, repo)
