"""Load a trained draft checkpoint and generate through the frozen Qwen verifier."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .dflash import DFlashDrafter, DraftConfig, verified_generate


def generate_checkpoint(checkpoint, prompt, tokens):
    if not prompt.strip() or tokens <= 0:
        raise ValueError("prompt must be nonempty and tokens positive")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = DraftConfig(**state["config"]).validate()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target = (
        AutoModelForCausalLM.from_pretrained(
            cfg.model,
            revision=cfg.model_revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(device)
        .float()
        .eval()
        .requires_grad_(False)
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.model_revision)
    drafter = DFlashDrafter(target.config, cfg).to(device).eval()
    drafter.load_state_dict(state["drafter"])
    prefix = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    result = verified_generate(
        drafter, target, prefix, tokens, state["mask_token_id"], tokenizer.eos_token_id
    )
    result["target_arithmetic"] = "float32; stored BF16 target weights upcast"
    result["text"] = tokenizer.decode(result["tokens"], skip_special_tokens=True)
    result["checkpoint_step"] = state["step"]
    result["note"] = (
        "FP32 greedy verification matched checked GPU cases. BF16 AR/block parity is not established. Anchor excluded; this call is not a speed benchmark."
    )
    return result
