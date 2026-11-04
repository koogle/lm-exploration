"""Hugging Face causal LMs (the pinned Qwen3 reference bases) for `trainer.loop`.

Run on one host with `torchrun -m trainer.reference --config <run config> --run-dir <dir>
--data-root <dir> --stop-unix <t>` (launched by `modal_scripts/modal_agentic.py::train_reference`).

A run fully fine-tunes `base` at `revision` on a chat SFT recipe (`training_data.chat_sft`)
whose subsets were written in the base's tokenizer and chat format (`eos_id` and
`tokenizer` in the recipe; `modal_scripts/modal_agent_mix.py`). `ChatMixture` packs whole
conversations into windows of the recipe's context; positions restart after every EOS,
and attention stays inside each conversation: `model_hybrid.model.DocumentMask` and its
compiled FlexAttention, registered as a Transformers attention implementation. Liger's
Triton kernels replace RMSNorm, SwiGLU and RoPE, and its fused linear cross-entropy
computes the loss on target tokens without [tokens, vocab] logits. Weights train in fp32
under the loop's bf16 autocast with AdamW; the learning rate warms up linearly over
`warmup_tokens` and follows a cosine to `min_learning_rate_fraction`.

Exports (`export/<name>/`) are bf16 Hugging Face models that vLLM loads as they are: the
config's positions cover the trained context, generation stops at `<|im_end|>`, and
`run.json` names the base, recipe, step and tokens. One is written every
`export_every_tokens` (0: none) and at the end, and uploaded to `hf_repo` when set.
"""

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path

import torch
from torch import nn

ATTENTION = "document_flex"
IM_END = "<|im_end|>"


@dataclasses.dataclass(frozen=True)
class RunConfig:
    base: str
    revision: str
    recipe: str
    target_tokens: int  # window tokens of one pass: `modal_agent_mix.py::target`
    global_rows: int = 8
    microbatch: int = 1
    learning_rate: float = 2e-5
    warmup_tokens: int = 50_000_000
    min_learning_rate_fraction: float = 0.1
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    gradient_checkpointing: bool = False
    seed: int = 1234
    log_every_steps: int = 10
    eval_every_steps: int = 500
    eval_conversations: int = 64
    checkpoint_seconds: int = 3600
    keep_checkpoints: int = 1
    export_every_tokens: int = 0
    hf_repo: str = ""

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def document_attention(module, query, key, value, attention_mask, document_mask=None, **_):
    """Causal attention within each document of a packed window ([B, H, T, D] in,
    [B, T, H, D] out), as the hybrid computes it."""
    from model_hybrid.model import attend

    output, _ = attend(query, key, value, document_mask)
    return output.transpose(1, 2).contiguous(), None


def register_attention():
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface

    AttentionInterface.register(ATTENTION, document_attention)
    # The document mask comes with the call; Transformers builds none.
    AttentionMaskInterface.register(ATTENTION, lambda *args, **kwargs: None)


class PackedLM(nn.Module):
    """The Hugging Face model on packed windows: summed cross-entropy of the target tokens."""

    def __init__(self, lm, eos_id):
        super().__init__()
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

        self.lm, self.eos_id = lm, eos_id
        self.cross_entropy = LigerFusedLinearCrossEntropyLoss(reduction="sum")

    def forward(self, inputs, labels):
        from model_hybrid.documents import document_ids
        from model_hybrid.model import DocumentMask

        mask = DocumentMask(document_ids(inputs, self.eos_id))
        hidden = self.lm.model(
            input_ids=inputs, position_ids=mask.positions, document_mask=mask
        ).last_hidden_state
        weight = self.lm.get_output_embeddings().weight
        return self.cross_entropy(weight, hidden.flatten(0, 1), labels.flatten())


class ReferenceFamily:
    """A reference run config for trainer.loop, training into `output`."""

    def __init__(self, config, output, data_root):
        from training_data import chat_sft

        self.config, self.output = config, Path(output)
        self.settings = dataclasses.asdict(config)
        self.recipe = json.loads(Path(config.recipe).read_text())
        if "eos_id" not in self.recipe or "tokenizer" not in self.recipe:
            raise ValueError("A reference recipe names its tokenizer and EOS id")
        self.root = Path(data_root) / self.recipe["name"]
        self.context = self.recipe["context"]
        self.global_rows, self.microbatch = config.global_rows, config.microbatch
        self.target_tokens, self.gradient_clip = config.target_tokens, config.gradient_clip
        self.seed, self.log_every = config.seed, config.log_every_steps
        self.evaluate_every = config.eval_every_steps
        self.checkpoint_every, self.checkpoint_seconds = 0, config.checkpoint_seconds
        self.keep_checkpoints = config.keep_checkpoints
        self.collective_timeout_seconds = 3600
        self.subsets = [name for name, _, _ in chat_sft.subsets(self.recipe)]
        self.lm, self.next_export = None, None

    def learning_rate_factor(self, tokens):
        config = self.config
        if tokens < config.warmup_tokens:
            return tokens / config.warmup_tokens
        progress = (tokens - config.warmup_tokens) / max(
            1, self.target_tokens - config.warmup_tokens
        )
        floor = config.min_learning_rate_fraction
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    def build(self, device):
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3
        from transformers import AutoModelForCausalLM

        torch.backends.cuda.matmul.allow_tf32 = True
        register_attention()
        apply_liger_kernel_to_qwen3(
            rope=True, rms_norm=True, swiglu=True, cross_entropy=False,
            fused_linear_cross_entropy=False,
        )  # fmt: skip
        lm = AutoModelForCausalLM.from_pretrained(
            self.config.base, revision=self.config.revision, dtype=torch.float32,
            attn_implementation=ATTENTION,
        )  # fmt: skip
        lm.config.use_cache = False
        if self.config.gradient_checkpointing:
            lm.gradient_checkpointing_enable({"use_reentrant": False})
        self.lm = lm.to(device)
        return PackedLM(self.lm, self.recipe["eos_id"]).to(device)

    def initialize(self, model):
        """`build` loads the pinned base weights."""

    def compile(self, model):
        return model

    def make_optimizer(self, model):
        config = self.config
        group = {"params": list(model.parameters()), "base_lr": config.learning_rate}
        return torch.optim.AdamW(
            [group], lr=config.learning_rate, betas=(0.9, 0.95),
            weight_decay=config.weight_decay, fused=True,
        )  # fmt: skip

    def open_data(self, deadline, state):
        from model_hybrid.train import HybridData
        from training_data import chat_sft

        identity = {"recipe": self.recipe, "base": [self.config.base, self.config.revision]}
        return HybridData(identity, chat_sft.ChatMixture(self.recipe, self.root, self.seed), state)

    def metadata(self):
        return {
            "architecture": {"base": self.config.base, "revision": self.config.revision},
            "tokenizer": self.recipe["tokenizer"],
        }

    def source_files(self):
        from model_hybrid import documents
        from model_hybrid import model as hybrid_model
        from training_data import chat_sft

        return {
            "family": Path(__file__),
            "chat_sft": Path(chat_sft.__file__),
            "hybrid.documents": Path(documents.__file__),
            "hybrid.model": Path(hybrid_model.__file__),
        }

    def branchable(self, saved):
        return False

    def check_resume(self, saved):
        """Resume needs only the loop's checks."""

    def heldout(self, data):
        from training_data import chat_sft

        windows = {
            name: chat_sft.chat_heldout_windows(
                self.recipe, self.root, name, self.context, self.config.eval_conversations
            )
            for name in self.subsets
        }
        return {name: w for name, w in windows.items() if w is not None}

    @torch.no_grad()
    def evaluate(self, model, heldout, progress, final):
        """Held-out loss and bits per byte of each subset's target tokens."""
        device = next(model.parameters()).device
        model.eval()
        results = {}
        try:
            for name, (inputs, labels, total_bytes) in heldout.items():
                total, count = 0.0, 0
                for row in range(len(inputs)):
                    x = torch.from_numpy(inputs[row : row + 1]).to(device)
                    y = torch.from_numpy(labels[row : row + 1]).to(device)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        total += float(model(x, y))
                    count += int((y != -100).sum())
                results[name] = {
                    "loss": total / count,
                    "bits_per_byte": total / math.log(2) / total_bytes,
                    "tokens": count,
                }
        finally:
            model.train()
        return {"heldout": results}

    def prepare(self, inputs, labels):
        return {}, {"ntp": (labels != -100).sum()}

    def loss(self, model, inputs, labels, extras, counts, progress):
        total = model(inputs, labels)
        return total / counts["ntp"], {"ce": total.detach()}

    def log(self, sums, counts):
        return {"loss": float(sums["ce"] / counts["ntp"])}

    def step_metrics(self, inputs, seconds, budget, world):
        """Model FLOPs utilization: 6 FLOPs per parameter and token, plus attention over
        each token's mean number of attended keys within its document."""
        from model_hybrid.documents import document_ids, document_positions
        from model_hybrid.train import H100_BF16_FLOPS, mean_attended_keys

        config = self.lm.config
        keys = mean_attended_keys(
            document_positions(document_ids(inputs[:8], self.recipe["eos_id"]))
        )
        width = config.num_attention_heads * config.head_dim
        flops = 6 * self.lm.num_parameters() + 12 * config.num_hidden_layers * width * keys
        return {"mfu": budget / seconds * flops / (world * H100_BF16_FLOPS)}

    def export(self, progress, name):
        """The model in bf16 as `export/<name>/`, uploaded to `hf_repo` when set."""
        from transformers import AutoTokenizer

        folder = self.output / "export" / name
        lm, config = self.lm, self.lm.config
        saved = (config.max_position_embeddings, config.use_cache, config._attn_implementation)
        tokenizer = AutoTokenizer.from_pretrained(self.config.base, revision=self.config.revision)
        stops = lm.generation_config.eos_token_id
        try:
            config.max_position_embeddings = max(config.max_position_embeddings, self.context)
            config.use_cache, config._attn_implementation = True, "sdpa"
            lm.generation_config.eos_token_id = [
                tokenizer.convert_tokens_to_ids(IM_END), tokenizer.eos_token_id,
            ]  # fmt: skip
            state = {key: value.to(torch.bfloat16) for key, value in lm.state_dict().items()}
            lm.save_pretrained(folder, state_dict=state)
        finally:
            config.max_position_embeddings, config.use_cache, config._attn_implementation = saved
            lm.generation_config.eos_token_id = stops
        tokenizer.save_pretrained(folder)
        run = {
            "base": self.config.base,
            "revision": self.config.revision,
            "recipe": self.recipe["name"],
            "context": self.context,
            **progress,
        }
        (folder / "run.json").write_text(json.dumps(run, indent=2) + "\n")
        if self.config.hf_repo:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(self.config.hf_repo, private=True, exist_ok=True)
            api.upload_folder(
                repo_id=self.config.hf_repo, folder_path=folder, path_in_repo=f"export/{name}",
                commit_message=f"{self.output.name}: export {name}",
            )  # fmt: skip
        return name

    def after_step(self, data, model, progress):
        every = self.config.export_every_tokens
        if not every:
            return
        if self.next_export is None:
            self.next_export = progress["tokens"] - progress["tokens"] % every + every
        if progress["tokens"] >= self.next_export:
            self.export(progress, f"tokens-{progress['tokens']:012d}")
            self.next_export += every

    def finish(self, data, model, progress, reason):
        """The final export when the run reached its target or the end of its data."""
        if reason in ("token_limit", "data_exhausted"):
            return {"export": self.export(progress, "final")}
        return {"export": None}


def main():
    from distributed.config import DistributedRuntime
    from trainer import loop

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--stop-unix", type=float, required=True)
    parser.add_argument("--resume", help="A checkpoint to resume (default: the run's latest)")
    args = parser.parse_args()
    config = RunConfig.load(args.config)
    loop.train(
        ReferenceFamily(config, args.run_dir, args.data_root),
        args.run_dir,
        runtime=DistributedRuntime.from_torchrun(os.environ),
        resume=args.resume,
        stop_unix=args.stop_unix,
    )


if __name__ == "__main__":
    main()
