"""Exact modern-student conversion to Llama weights for standard HF/vLLM serving."""


def convert(student, bos_token_id=1, eos_token_id=2, pad_token_id=0):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    c = student.config
    if c.architecture_version != 2:
        raise ValueError("Only modern RMSNorm/RoPE/SwiGLU checkpoints are supported")
    config = LlamaConfig(
        vocab_size=c.vocab_size,
        hidden_size=c.n_embd,
        intermediate_size=(8 * c.n_embd + 2) // 3,
        num_hidden_layers=c.n_layer,
        num_attention_heads=c.n_head,
        num_key_value_heads=c.n_head,
        max_position_embeddings=c.block_size,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        hidden_act="silu",
        attention_bias=True,
        mlp_bias=True,
        tie_word_embeddings=True,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).to(
        device=next(student.parameters()).device, dtype=next(student.parameters()).dtype
    )
    original = student.state_dict()
    mapped = {
        "model.embed_tokens.weight": original["token_embedding.weight"],
        "lm_head.weight": original["lm_head.weight"],
        "model.norm.weight": original["final_norm.weight"],
    }

    def permute(tensor):
        # Native RoPE pairs adjacent coordinates; Llama pairs the two half-heads.
        shape = tensor.shape
        return (
            tensor.reshape(c.n_head, c.n_embd // c.n_head // 2, 2, *shape[1:])
            .transpose(1, 2)
            .reshape(shape)
            .contiguous()
        )

    for i in range(c.n_layer):
        src = f"blocks.{i}."
        dst = f"model.layers.{i}."
        for suffix in ["weight", "bias"]:
            q, k, v = original[src + "attention.qkv." + suffix].chunk(3, 0)
            for name, tensor in [
                ("q_proj", permute(q)),
                ("k_proj", permute(k)),
                ("v_proj", v),
            ]:
                mapped[dst + "self_attn." + name + "." + suffix] = tensor
            mapped[dst + "self_attn.o_proj." + suffix] = original[
                src + "attention.projection." + suffix
            ]
            for name in ["gate", "up", "down"]:
                mapped[dst + "mlp." + name + "_proj." + suffix] = original[
                    src + "mlp." + name + "." + suffix
                ]
        mapped[dst + "input_layernorm.weight"] = original[src + "attention_norm.weight"]
        mapped[dst + "post_attention_layernorm.weight"] = original[
            src + "mlp_norm.weight"
        ]
    model.load_state_dict(mapped, strict=True)
    model.eval()
    student.eval()
    # Probe exact logits before permitting export. This also catches RoPE permutation errors.
    device = next(student.parameters()).device
    ids = torch.arange(min(17, c.block_size), device=device).remainder(c.vocab_size)[
        None
    ]
    with torch.no_grad():
        native = student(ids).float()
        exported = model(ids, use_cache=False).logits.float()
    torch.testing.assert_close(native, exported, rtol=2e-4, atol=2e-5)
    return model, {"max_abs_logit_error": (native - exported).abs().max().item()}


def export_checkpoint(checkpoint, output):
    import json
    from pathlib import Path

    import torch
    from model_100m.model import GPT, ModelConfig
    from training_data.subword import load_frozen
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("format") not in {
        "byte-distillation-v1",
        "shared-token-distillation-v1",
    }:
        raise ValueError("Expected trusted distillation checkpoint")
    student = GPT(ModelConfig(**saved["model_config"]))
    student.load_state_dict(saved["model"], strict=True)
    if saved["format"] == "shared-token-distillation-v1":
        import hashlib

        tokenizer = AutoTokenizer.from_pretrained(Path(checkpoint).parent / "tokenizer")
        digest = hashlib.sha256(
            tokenizer.backend_tokenizer.to_str().encode()
        ).hexdigest()
        if digest != saved["settings"]["tokenizer_sha256"]:
            raise ValueError("Archived tokenizer mismatch")
        eos = tokenizer.eos_token_id
        model, report = convert(student, eos, eos, eos)
    else:
        model, report = convert(student)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=load_frozen(),
            bos_token="<|bos|>",
            eos_token="<|eos|>",
            pad_token="<|pad|>",
            unk_token="<|unk|>",
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    (output / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
