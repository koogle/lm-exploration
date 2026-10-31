"""The hybrid family for trainer.loop, and the evaluation and export helpers it uses.

`HybridFamily` adapts a run config (hybrid/configs/*.json) to the common training
loop: the hybrid model with role-declared Muon/AdamW, prepared-shard or chat SFT
data with optional replay (budget: every window token), the objective mean
next-token loss plus the decaying MTP weight times mean MTP loss (each over its
global target count), held-out loss, bits per byte, gate and sink statistics and
loss by in-document position above 4K context, fixed-prompt samples, and bf16
exports to the run's HF repo every `hf_every_tokens` and at the end. `HNetFamily`
does the same for a run config with `"architecture": "hnet"`: the byte-level H-Net
(hnet.py) on the recipe's shards decoded to UTF-8 bytes (byte_data.py); budget,
context and batch rows count bytes.

    torchrun --nproc_per_node=8 -m model_hybrid.train --config <run.json> \\
        --run-dir <dir> --data-root <dir> --stop-unix <time> [--max-tokens N] \\
        [--max-steps N] [--resume <checkpoint>] [--stop-file <path>]

A run directory resumes from `--resume` or else its latest checkpoint;
`workflows.modal_train` runs sessions this way for every family.
"""

import argparse
import dataclasses
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from tokenizer import glm

from . import documents as doc_utils
from . import hnet
from . import model as hybrid

H100_BF16_FLOPS = 989e12
SAMPLE_PROMPTS = (
    "The history of the printing press",
    "def fibonacci(n):",
    "Question: A train travels 120 km in 2 hours. What is its average speed?\nAnswer:",
    "In mathematics, a prime number is",
)


@dataclasses.dataclass(frozen=True)
class RunConfig:
    recipe: str
    model: dict
    target_tokens: int
    architecture: str = "hybrid"
    global_rows: int = 256
    microbatch: int = 4
    warmup_tokens: int = 500_000_000
    decay_fraction: float = 0.15
    optimizer: dict = dataclasses.field(default_factory=dict)
    gradient_clip: float = 1.0
    compile: bool = True
    checkpoint_seconds: int = 5400
    keep_checkpoints: int = 2
    eval_every_steps: int = 500
    eval_documents: dict = dataclasses.field(
        default_factory=lambda: {"climbmix": 1000, "code": 300, "math": 300}
    )
    sample_every_steps: int = 2000
    log_every_steps: int = 10
    hf_repo: str = ""
    hf_every_tokens: int = 5_000_000_000
    seed: int = 1234
    # An export folder (config.json, model.safetensors) whose trunk weights start the
    # run when it has no checkpoint yet: a pretrained model for mid-training or SFT.
    init_from: str = ""

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def __post_init__(self):
        if self.architecture not in ("hybrid", "hnet"):
            raise ValueError("architecture must be 'hybrid' or 'hnet'")


def learning_rate_factor(config, tokens):
    """Linear warmup, constant, then linear decay to 0 over the last decay_fraction."""
    if tokens < config.warmup_tokens:
        return tokens / config.warmup_tokens
    decay_start = (1 - config.decay_fraction) * config.target_tokens
    if tokens < decay_start:
        return 1.0
    return max(0.0, (config.target_tokens - tokens) / (config.target_tokens - decay_start))


def flops_per_token(model_config, lm, attended_keys):
    """Training FLOPs per token: 6 x matmul weights (incl. both output-layer uses) + attention.

    The MTP module reuses the output layer for its sampled fraction of positions.
    """
    matmul = sum(p.numel() for n, p in lm.named_parameters() if p.ndim == 2)
    if model_config.mtp:
        matmul += model_config.mtp_loss_fraction * lm.output_weight.numel()
    full_layers = model_config.layer_kinds.count("full") + (
        1 if model_config.mtp and model_config.mtp_layer_kind == "full" else 0
    )
    attention = 12 * full_layers * model_config.n_head * model_config.head_dim * attended_keys
    return 6 * matmul + attention


def mean_attended_keys(positions):
    return float((positions.float() + 1).mean())


def model_config_of(config):
    if config.architecture == "hnet":
        return hnet.HNetConfig(**config.model)
    model_config = hybrid.HybridConfig(
        **{k: tuple(v) if isinstance(v, list) else v for k, v in config.model.items()}
    )
    if model_config.eos_id != glm.EOS_ID:
        raise ValueError("model eos_id must be the GLM tokenizer's EOS")
    return model_config


# The launcher creates this file when a graceful stop is requested.
STOP_FILE = os.environ.get("HYBRID_STOP_FILE", "/tmp/hybrid-stop")


def gate_sink_statistics(lm):
    """Parameter statistics of every attention sink, gate bias and KDA decay."""
    stats = {}
    for name, module in lm.named_modules():
        if isinstance(module, hybrid.Attention):
            sink = module.sink.detach().float()
            bias = module.gate_bias.detach().float()
            stats[name] = {
                "sink_mean": sink.mean().item(),
                "sink_max": sink.max().item(),
                "gate_bias_open": torch.sigmoid(bias).mean().item(),
            }
        elif name.endswith("mixer") and hasattr(module, "A_log"):
            stats[name] = {"decay_rate_mean": module.A_log.detach().float().exp().mean().item()}
    return stats


# In-document position buckets for long-context evaluation (used above 4K context).
POSITION_EDGES = (0, 1024, 4096, 16384, 32768)


def document_positions(inputs, eos_id):
    """Each input token's position within its document; EOS ends a document."""
    positions = np.zeros(inputs.shape, dtype=np.int64)
    for row, tokens in enumerate(inputs):
        position = 0
        for column, token in enumerate(tokens):
            positions[row, column] = position
            position = 0 if token == eos_id else position + 1
    return positions


def position_losses(lm, inputs, labels, device, eos_id, edges):
    """Held-out loss of the targets whose input sits in each in-document position bucket."""
    positions = document_positions(inputs, eos_id)
    out = {}
    for low, high in itertools.pairwise(edges):
        keep = (positions >= low) & (positions < high) & (labels != -100)
        if not keep.any():
            continue
        ce, count = 0.0, 0
        for start in range(0, len(inputs), 4):
            x = torch.from_numpy(inputs[start : start + 4]).to(device)
            y = torch.from_numpy(np.where(keep, labels, -100)[start : start + 4]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = lm(x, y)
            ce, count = ce + losses["ce"].item(), count + int(losses["count"])
        if count:
            out[f"{low}-{high}"] = ce / count
    return out


@torch.no_grad()
def evaluate(lm, heldout, device, eos_id):
    """Held-out loss, bits per byte and MTP loss per source; gate/sink activations; above
    4K context also the loss per in-document position bucket."""
    lm.eval()
    results = {}
    hybrid.Attention.record_statistics = True
    try:
        for name, (inputs, labels, total_bytes) in heldout.items():
            sums = {"ce": 0.0, "count": 0, "mtp_ce": 0.0, "mtp_count": 0}
            chunking = {"chunks": 0, "positions": 0}
            activations = []
            for start in range(0, len(inputs), 4):
                x = torch.from_numpy(inputs[start : start + 4]).to(device)
                y = torch.from_numpy(labels[start : start + 4]).to(device)
                targets = doc_utils.mtp_labels(x, y, eos_id) if lm.config.mtp else None
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    losses = lm(x, y, targets)
                # Models without MTP (the H-Net) return no mtp_ sums; chunk counts are H-Net only.
                for key in sums:
                    if key in losses:
                        sums[key] += losses[key].item() if key.endswith("ce") else int(losses[key])
                for key in chunking:
                    chunking[key] += int(losses.get(key, 0))
                activations.append(
                    [m.statistics for m in lm.modules() if isinstance(m, hybrid.Attention)]
                )
            layers = list(zip(*activations, strict=True))
            results[name] = {
                "loss": sums["ce"] / sums["count"],
                "bits_per_byte": sums["ce"] / math.log(2) / total_bytes,
                "mtp_loss": sums["mtp_ce"] / max(sums["mtp_count"], 1),
                "tokens": sums["count"],
                "gate_open": [float(np.mean([s["gate_open"] for s in layer])) for layer in layers],
                "sink_weight": [
                    float(np.mean([s["sink_weight"] for s in layer])) for layer in layers
                ],
            }
            if lm.config.block_size > POSITION_EDGES[2]:
                edges = [e for e in POSITION_EDGES if e < lm.config.block_size]
                edges.append(lm.config.block_size)
                results[name]["loss_by_position"] = position_losses(
                    lm, inputs, labels, device, eos_id, edges
                )
            if chunking["chunks"]:
                results[name]["bytes_per_chunk"] = chunking["positions"] / chunking["chunks"]
    finally:
        hybrid.Attention.record_statistics = False
        lm.train()
    return results


@torch.no_grad()
def samples(lm, tokenizer, device, prompts=SAMPLE_PROMPTS, stop_ids=(), new_tokens=48):
    """Greedy continuations of fixed prompts (full recompute per token), up to a stop id."""
    lm.eval()
    outputs = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        start = len(ids)
        for _ in range(new_tokens):
            x = torch.tensor([ids], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ids.append(int(lm(x)[0, -1].argmax()))
            if ids[-1] in stop_ids:
                break
        completion = tokenizer.decode(ids[start:], skip_special_tokens=False)
        outputs.append({"prompt": prompt, "completion": completion})
    lm.train()
    return outputs


def export_config(model_config):
    """An export's config.json: the model config with its architecture and tokenizer."""
    if isinstance(model_config, hnet.HNetConfig):
        return dataclasses.asdict(model_config) | {
            "architecture": "hnet",
            "tokenizer": "utf-8 bytes, EOS 256",
        }
    return dataclasses.asdict(model_config) | {
        "architecture": "hybrid",
        "tokenizer": glm.SOURCE,
        "tokenizer_sha256": glm.SHA256,
    }


def upload_export(lm, model_config, run_dir, progress, repo, private=True):
    """bf16 trunk weights and config under `<run_dir>/export/`, and to the HF model repo
    with the run's metrics in a background thread (returned; None without a repo)."""
    import threading

    from safetensors.torch import save_file

    folder = Path(run_dir, "export", f"tokens-{progress['tokens']:012d}")
    folder.mkdir(parents=True, exist_ok=True)
    weights = {
        k: v.detach().to("cpu", torch.bfloat16).contiguous()
        for k, v in lm.state_dict().items()
        if not k.startswith("mtp.")
    }
    save_file(weights, folder / "model.safetensors")
    (folder / "config.json").write_text(json.dumps(export_config(model_config), indent=2) + "\n")
    (folder / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    logs = [Path(run_dir, name) for name in ("metrics.jsonl", "validation.jsonl")]
    if not repo:
        return None

    def push():
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo, private=private, exist_ok=True)
        api.upload_folder(repo_id=repo, folder_path=folder, path_in_repo=folder.name)
        for log in logs:
            if log.exists():
                api.upload_file(path_or_fileobj=log, path_in_repo=log.name, repo_id=repo)

    thread = threading.Thread(target=push, daemon=False)
    thread.start()
    return thread


def heldout_windows(recipe, root, name, context, max_documents):
    """Held-out windows of a pretraining source or a chat subset: (inputs, labels, bytes).

    Pretraining sources concatenate whole documents into one stream
    (`packed_shards.heldout_windows`); chat subsets pack whole conversations and train
    only assistant tokens (`chat_sft.chat_heldout_windows`, None when empty).
    """
    if recipe.get("format") == "chat-sft":
        from training_data.chat_sft import chat_heldout_windows

        return chat_heldout_windows(recipe, root, name, context, max_documents)
    from training_data.packed_shards import heldout_windows as source_windows

    return source_windows(recipe, root, name, context, max_documents)


def pretraining_heldout(config):
    def heldout(recipe, mixture, context):
        from training_data.packed_shards import wait_for_heldout

        wait_for_heldout(recipe, mixture)
        return {
            name: heldout_windows(recipe, mixture.root, name, context, count)
            for name, count in config.eval_documents.items()
            if name in recipe["sources"]
        }

    return heldout


def recipe_data(config, data_root):
    """(identity, mixture factory, held-out factory(mixture, context), chat) for a run
    config's recipe.

    Replay reads an earlier recipe's prepared shards at this recipe's context; chat
    SFT recipes hold out conversations per subset, pretraining recipes per source,
    plus the replayed sources' held-out shards.
    """
    from training_data.packed_shards import PreparedMixture, ReplayMixture

    recipe = json.loads(Path(config.recipe).read_text())
    root = Path(data_root) / recipe["name"]
    replay = recipe.get("replay")
    if replay:
        replay_recipe = json.loads(
            (Path(config.recipe).parent / f"{replay['recipe']}.json").read_text()
        )
        replay_recipe["context"] = recipe["context"]
        replay_root = Path(data_root) / replay_recipe["name"]
        replay_rows = round(config.global_rows * replay["fraction"])

    def with_replay(primary):
        if not replay:
            return primary
        return ReplayMixture(primary, PreparedMixture(replay_recipe, replay_root), replay_rows)

    def replay_heldout(context):
        return {
            name: heldout_windows(replay_recipe, replay_root, name, context, count)
            for name, count in config.eval_documents.items()
            if replay and name in replay_recipe["sources"]
        }

    if recipe.get("format") == "chat-sft":
        from training_data import chat_sft

        def mixture():
            return with_replay(chat_sft.ChatMixture(recipe, root, config.seed))

        def heldout(mixture, context):
            count = config.eval_documents["chat"]
            own = {
                name: windows
                for name, _, _ in chat_sft.subsets(recipe)
                if (windows := heldout_windows(recipe, root, name, context, count)) is not None
            }
            return own | replay_heldout(context)
    else:
        own = pretraining_heldout(config)

        def mixture():
            return with_replay(PreparedMixture(recipe, root))

        def heldout(mixture, context):
            primary = getattr(mixture, "primary", mixture)
            return own(recipe, primary, context) | replay_heldout(context)

    identity = {
        "recipe": recipe,
        "replay": replay_recipe["name"] if replay else None,
        "tokenizer_sha256": glm.SHA256,
    }
    return identity, mixture, heldout, recipe.get("format") == "chat-sft"


class HybridData:
    """A recipe mixture for the loop: every window token counts toward the budget."""

    def __init__(self, identity, mixture, state):
        self.identity, self.mixture = identity, mixture
        if state is not None:
            mixture.load_state_dict(state)

    def next_batch(self, rows):
        inputs, labels, names = self.mixture.next_batch(rows)
        return inputs, labels, names, int(inputs.size)

    def state_dict(self):
        return self.mixture.state_dict()

    def load_state_dict(self, state):
        self.mixture.load_state_dict(state)

    def close(self):
        pass


class HybridFamily:
    """A hybrid run config for trainer.loop, training into `output` (its exports go to
    `output/export/`); `data` = (identity, mixture factory) overrides the recipe
    (synthetic benchmarks), and then nothing is held out."""

    def __init__(self, config, output, data_root=None, data=None):
        self.config, self.output = config, Path(output)
        self.settings = dataclasses.asdict(config)
        self.model_config = model_config_of(config)
        self.global_rows, self.microbatch = config.global_rows, config.microbatch
        self.target_tokens, self.gradient_clip = config.target_tokens, config.gradient_clip
        self.seed, self.log_every = config.seed, config.log_every_steps
        self.evaluate_every = config.eval_every_steps
        self.checkpoint_every, self.checkpoint_seconds = 0, config.checkpoint_seconds
        self.keep_checkpoints = config.keep_checkpoints
        self.collective_timeout_seconds = 3600
        if data is None:
            identity, factory, self.heldout_factory, chat = self.recipe_data(data_root)
        else:
            (identity, factory), self.heldout_factory, chat = data, None, False
        self.identity, self.factory = identity, factory
        if chat:
            from training_data import chat_sft

            self.sampling = (chat_sft.sample_prompts(), chat_sft.STOP_IDS, 160)
        else:
            self.sampling = (SAMPLE_PROMPTS, (), self.sample_length)
        self.lm, self.uploads = None, []
        self.next_upload = None

    sample_length = 48

    def recipe_data(self, data_root):
        return recipe_data(self.config, data_root)

    def learning_rate_factor(self, tokens):
        return learning_rate_factor(self.config, tokens)

    def build(self, device):
        torch.backends.cuda.matmul.allow_tf32 = True
        self.lm = hybrid.HybridLM(self.model_config).to(device)
        return self.lm

    def initialize(self, model):
        """An `init_from` export's trunk weights start a fresh run (mid-training, SFT);
        otherwise the model keeps its own initialization."""
        if self.config.init_from:
            from safetensors.torch import load_file

            path = Path(self.config.init_from) / "model.safetensors"
            model.load_state_dict(load_file(path, device="cpu"))

    def compile(self, model):
        return torch.compile(model) if self.config.compile else model

    def make_optimizer(self, model):
        from trainer.optimizers import OptimizerConfig, RoleOptimizer

        settings = {
            k: tuple(v) if isinstance(v, list) else v for k, v in self.config.optimizer.items()
        }
        return RoleOptimizer(model, OptimizerConfig(**settings), fused=True)

    def open_data(self, deadline, state):
        return HybridData(self.identity, self.factory(), state)

    def metadata(self):
        return {
            "architecture": {
                "module": hybrid.HybridConfig.__module__,
                "config": dataclasses.asdict(self.model_config),
            },
            "tokenizer": glm.SOURCE,
        }

    def source_files(self):
        from training_data import chat, chat_sft, mixture, packed_shards

        from tokenizer import glm

        from . import (
            cuda_extension,
            documents,
            fused_loss,
            kda,
            kda_convolution,
            softmax_kernel,
        )
        from . import model as hybrid_model

        modules = (
            cuda_extension,
            documents,
            fused_loss,
            hybrid_model,
            kda,
            kda_convolution,
            softmax_kernel,
        )
        files = {f"hybrid.{Path(m.__file__).stem}": Path(m.__file__) for m in modules}
        return {
            "family": Path(__file__),
            **files,
            "chat": Path(chat.__file__),
            "chat_sft": Path(chat_sft.__file__),
            "tokenizer_glm": Path(glm.__file__),
            "packed_shards": Path(packed_shards.__file__),
            "mixture": Path(mixture.__file__),
        }

    def branchable(self, saved):
        return False

    def check_resume(self, saved):
        """Resume needs only the loop's checks."""

    def heldout(self, data):
        if self.heldout_factory is None:
            return None
        windows = self.heldout_factory(data.mixture, self.model_config.block_size)
        return {"windows": windows, "tokenizer": glm.load()}

    def evaluate(self, model, heldout, progress, final):
        if heldout is None:
            return {}
        device = next(model.parameters()).device
        result = {
            "heldout": evaluate(model, heldout["windows"], device, self.model_config.eos_id),
            "parameters": gate_sink_statistics(model),
        }
        if final or progress["step"] % self.config.sample_every_steps == 0:
            prompts, stop_ids, new_tokens = self.sampling
            result["samples"] = samples(
                model, heldout["tokenizer"], device, prompts, stop_ids, new_tokens
            )
        return result

    def prepare(self, inputs, labels):
        counts = {"ntp": (labels != -100).sum()}
        if not self.model_config.mtp:
            return {}, counts
        targets = doc_utils.mtp_labels(inputs, labels, self.model_config.eos_id)
        return {"mtp_labels": targets}, counts | {"mtp": (targets != -100).sum()}

    def loss(self, model, inputs, labels, extras, counts, progress):
        losses = model(inputs, labels, extras.get("mtp_labels"))
        objective = losses["loss"] / counts["ntp"]
        sums = {"ce": losses["ce"], "z": losses["z"]}
        if self.model_config.mtp:
            # Expected number of sampled valid MTP targets (see HybridLM.forward).
            mtp_count = counts["mtp"].clamp_min(1) * self.model_config.mtp_loss_fraction
            weight = hybrid.mtp_weight(self.model_config, progress["tokens"] / self.target_tokens)
            objective = objective + weight * losses["mtp_loss"] / mtp_count
            sums["mtp_ce"] = losses["mtp_ce"]
        return objective, sums

    def log(self, sums, counts):
        row = {
            "loss": float(sums["ce"] / counts["ntp"]),
            "z": float(sums["z"] / counts["ntp"]),
        }
        if self.model_config.mtp:
            mtp_count = counts["mtp"].clamp_min(1) * self.model_config.mtp_loss_fraction
            row["mtp_loss"] = float(sums["mtp_ce"] / mtp_count)
        return row

    def step_metrics(self, inputs, seconds, budget, world):
        documents = doc_utils.document_ids(inputs[:8], self.model_config.eos_id)
        keys = mean_attended_keys(doc_utils.document_positions(documents))
        flops = flops_per_token(self.model_config, self.lm, keys)
        return {"mfu": budget / seconds * flops / (world * H100_BF16_FLOPS)}

    def after_step(self, data, model, progress):
        """A bf16 export to the HF repo every `hf_every_tokens`."""
        config = self.config
        if not config.hf_repo:
            return
        if self.next_upload is None:
            done = progress["tokens"] - progress["tokens"] % config.hf_every_tokens
            self.next_upload = done + config.hf_every_tokens
        if progress["tokens"] >= self.next_upload:
            self.uploads.append(
                upload_export(model, self.model_config, self.output, progress, config.hf_repo)
            )
            self.next_upload += config.hf_every_tokens

    def finish(self, data, model, progress, reason):
        """The final export when the run reached its target; waits for every upload."""
        exported = None
        if self.config.hf_repo and reason == "token_limit":
            self.uploads.append(
                upload_export(model, self.model_config, self.output, progress, self.config.hf_repo)
            )
            exported = f"tokens-{progress['tokens']:012d}"
        for thread in self.uploads:
            thread.join()
        return {"export": exported}


class HNetFamily(HybridFamily):
    """The byte-level H-Net for trainer.loop.

    Data are the recipe's prepared shards decoded to UTF-8 bytes in the same order and
    document mix (`byte_data.ByteMixture`); every window byte counts toward the budget.
    The objective is the mean next-byte loss plus the weighted ratio loss, which the
    model folds into its summed loss. Logs add the ratio loss and bytes per chunk.
    """

    sample_length = 192

    def recipe_data(self, data_root):
        from .byte_data import recipe_data as byte_recipe_data

        return (*byte_recipe_data(self.config, data_root), False)

    def build(self, device):
        torch.backends.cuda.matmul.allow_tf32 = True
        hnet.static_compilation(self.model_config)
        self.lm = hnet.HNetLM(self.model_config).to(device)
        return self.lm

    def metadata(self):
        return {
            "architecture": {
                "module": hnet.HNetConfig.__module__,
                "config": dataclasses.asdict(self.model_config),
            },
            "tokenizer": "utf-8 bytes, EOS 256",
        }

    def source_files(self):
        from . import byte_data

        files = {"hybrid.hnet": Path(hnet.__file__), "hybrid.byte_data": Path(byte_data.__file__)}
        return super().source_files() | files

    def heldout(self, data):
        if self.heldout_factory is None:
            return None
        windows = self.heldout_factory(data.mixture, self.model_config.block_size)
        return {"windows": windows, "tokenizer": hnet.ByteTokenizer()}

    def prepare(self, inputs, labels):
        return {}, {"ntp": (labels != -100).sum()}

    def loss(self, model, inputs, labels, extras, counts, progress):
        losses = model(inputs, labels)
        sums = {key: losses[key].detach().float() for key in ("ce", "z", "ratio", "chunks")}
        return losses["loss"] / counts["ntp"], sums

    def log(self, sums, counts):
        microbatches = self.global_rows // self.microbatch
        self.bytes_per_chunk = (
            self.global_rows * self.model_config.block_size / float(sums["chunks"])
        )
        return {
            "loss": float(sums["ce"] / counts["ntp"]),
            "z": float(sums["z"] / counts["ntp"]),
            "ratio_loss": float(sums["ratio"]) / microbatches,
            "bytes_per_chunk": self.bytes_per_chunk,
        }

    def step_metrics(self, inputs, seconds, budget, world):
        documents = doc_utils.document_ids(inputs[:8], self.model_config.eos_id)
        keys = mean_attended_keys(doc_utils.document_positions(documents))
        ratio = self.bytes_per_chunk
        flops = hnet.flops_per_byte(self.lm, ratio, keys / ratio)
        return {"mfu": budget / seconds * flops / (world * H100_BF16_FLOPS)}


def family_of(config):
    return HNetFamily if config.architecture == "hnet" else HybridFamily


def main():
    from distributed.config import DistributedRuntime
    from trainer import loop

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--stop-unix", type=float, required=True)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", help="A checkpoint to resume (default: the run's latest)")
    parser.add_argument("--stop-file", default=STOP_FILE)
    args = parser.parse_args()
    config = RunConfig.load(args.config)
    if config.architecture == "hnet":
        # The main network's length changes with the chunk count; expandable segments
        # keep the varying activation sizes from fragmenting the allocator. Set before
        # the first CUDA allocation.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    loop.train(
        family_of(config)(config, args.run_dir, args.data_root),
        args.run_dir,
        runtime=DistributedRuntime.from_torchrun(os.environ),
        resume=args.resume,
        stop_unix=args.stop_unix,
        max_steps=args.max_steps,
        stop_tokens=args.max_tokens,
        stop_file=args.stop_file,
    )


if __name__ == "__main__":
    main()
