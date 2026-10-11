"""The encoder only sees committed prefix tokens, never prediction targets."""

import torch
from torch import nn


class FrozenBackbone(nn.Module):
    def __init__(self, cfg, device="cuda"):
        super().__init__()
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForCausalLM

        self.cfg = cfg
        model_config = AutoConfig.from_pretrained(cfg.model, revision=cfg.model_revision)
        loader = (
            Qwen3_5ForCausalLM if model_config.model_type == "qwen3_5" else AutoModelForCausalLM
        )
        self.model = (
            loader.from_pretrained(
                cfg.model,
                revision=model_config._commit_hash,
                dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                attn_implementation="sdpa",
            )
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model, revision=model_config._commit_hash
        )
        self.hidden_size = self.model.config.hidden_size
        if max(cfg.layers) != self.model.config.num_hidden_layers:
            raise ValueError("last selected layer must be the model's final hidden state")
        self.jacobians = None
        self.revision = self.model.config._commit_hash

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def encode(self, prefix, *, past_key_values=None):
        residuals, attentions, hooks = {}, {}, []

        def capture(destination, index):
            def hook(module, args, output):
                value = output[0] if isinstance(output, tuple) else output
                destination[index] = value[:, -self.cfg.memory_tokens :].detach()

            return hook

        try:
            for index in self.cfg.layers:
                layer = self.model.model.layers[index - 1]
                hooks.append(layer.register_forward_hook(capture(residuals, index)))
                attention = getattr(layer, "self_attn", getattr(layer, "linear_attn", None))
                hooks.append(attention.register_forward_hook(capture(attentions, index)))
            mask = prefix.ge(0).long()
            if past_key_values is not None:
                previous = getattr(past_key_values, "latent_attention_mask", None)
                if previous is not None:
                    mask = torch.cat([previous, mask], dim=-1)
            positions = (mask.cumsum(-1) - 1).clamp(min=0)[:, -prefix.shape[1] :]
            out = self.model.model(
                input_ids=prefix.clamp(min=0),
                attention_mask=mask,
                position_ids=positions,
                use_cache=True,
                past_key_values=past_key_values,
            )
            out.past_key_values.latent_attention_mask = mask
        finally:
            for hook in hooks:
                hook.remove()
        residual = torch.stack([residuals[i] for i in self.cfg.layers], dim=1)
        attention = torch.stack([attentions[i] for i in self.cfg.layers], dim=1)
        parts = [residual, attention]
        if self.jacobians is not None and getattr(self, "conditioning", None) in (None, "jspace"):
            from .jspace import readout

            parts.append(readout(self, residual))
        return torch.cat(parts, dim=1), out.past_key_values

    @torch.no_grad()
    def embed(self, ids):
        return self.model.get_input_embeddings()(ids)

    def project(self, features):
        # Frozen weights, but preserve gradients to the trainable head.
        weight = self.model.get_output_embeddings().weight
        return torch.nn.functional.linear(self.model.model.norm(features.to(weight.dtype)), weight)

    @torch.no_grad()
    def teacher_logits(self, prefix, target):
        joined = torch.cat([prefix, target[:, :-1]], dim=1)
        return self.model(
            joined.clamp(min=0), attention_mask=joined.ge(0).long(), use_cache=False
        ).logits[:, prefix.shape[1] - 1 :]


def compatible_tokenizers(source, target):
    """Permit a frozen token stream only when every source token ID is preserved."""
    import hashlib
    import json

    source_vocab, target_vocab = source.get_vocab(), target.get_vocab()
    if any(target_vocab.get(token) != index for token, index in source_vocab.items()):
        raise ValueError("parent tokenizer token IDs are incompatible")
    source_spec = json.loads(source.backend_tokenizer.to_str())
    target_spec = json.loads(target.backend_tokenizer.to_str())
    for key in ("model", "normalizer", "pre_tokenizer", "post_processor"):
        if source_spec.get(key) != target_spec.get(key):
            raise ValueError(f"parent tokenizer segmentation differs: {key}")
    return {
        "source_vocabulary_sha256": hashlib.sha256(
            json.dumps(source_vocab, sort_keys=True).encode()
        ).hexdigest(),
        "source_token_count": len(source_vocab),
        "target_token_count": len(target_vocab),
        "all_source_token_ids_preserved": True,
        "note": "Reuse the exact frozen token stream. Additional target special tokens are not introduced into it.",
    }
