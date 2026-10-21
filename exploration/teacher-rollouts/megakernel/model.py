"""Checked batch-one Qwen3.5-9B transformer megakernel, with a cuBLAS LM head."""

import ctypes
import json
import struct
import subprocess
from pathlib import Path

import torch

MODEL = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
LAYERS = [0, 0, 0, 1] * 8
HERE = Path(__file__).parent


def compile_kernel(destination):
    destination = Path(destination)
    cmd = [
        "nvcc",
        "-std=c++17",
        "-O3",
        "-arch=sm_90",
        "-DTARGET_SM=90",
        "--shared",
        "-Xcompiler=-fPIC",
        str(HERE / "kernel.cu"),
        "-o",
        str(destination),
    ]
    result = subprocess.run(
        cmd, check=False, capture_output=True, text=True, timeout=180
    )
    destination.with_suffix(".build.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(result.stderr[-10000:])
    return destination


def load_teacher(cache_dir):
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoTokenizer, Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5TextModel,
        Qwen3_5TextRotaryEmbedding,
    )

    path = Path(
        snapshot_download(
            MODEL,
            revision=REVISION,
            cache_dir=cache_dir,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "tokenizer*",
                "*.jinja",
                "merges.txt",
                "vocab.json",
            ],
        )
    )
    config = Qwen3_5TextConfig(
        **json.loads((path / "config.json").read_text())["text_config"]
    )
    expected = {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
    }
    for key, value in expected.items():
        if getattr(config, key) != value:
            raise ValueError(f"unsupported {key}")
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen3_5TextModel(config)
    weights = {}
    head = None
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cuda") as f:
            for key in f.keys():  # noqa: SIM118 - safetensors safe_open exposes keys, not iteration
                if key.startswith("model.language_model."):
                    name = key.removeprefix("model.language_model.")
                    dtype = (
                        torch.float32
                        if name.endswith(("A_log", "dt_bias"))
                        else torch.bfloat16
                    )
                    weights[name] = f.get_tensor(key).to(dtype).contiguous()
                elif key == "lm_head.weight":
                    head = f.get_tensor(key).to(torch.bfloat16).contiguous()
    model.load_state_dict(weights, assign=True, strict=True)
    model.rotary_emb = Qwen3_5TextRotaryEmbedding(config, device="cuda")
    model.eval().requires_grad_(False)
    if head is None:
        raise ValueError("missing untied LM head")
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, head, tokenizer


class Decoder:
    def __init__(self, model, head, library, max_seq_len=1024):
        if torch.cuda.get_device_capability() != (9, 0):
            raise ValueError("This build is qualified only for SM90/H100")
        if not 1 <= max_seq_len <= 4096:
            raise ValueError("max_seq_len must be 1..4096")
        self.model, self.head = model, head
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.library.decode.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.library.decode.restype = ctypes.c_int
        self.library.decode_device.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.library.decode_device.restype = ctypes.c_int
        self.library.error_string.argtypes = [ctypes.c_int]
        self.library.error_string.restype = ctypes.c_char_p
        self.resident_blocks = self.library.maximum_blocks()
        if self.resident_blocks < 32:
            raise ValueError("insufficient cooperative grid occupancy")
        self.max_seq_len, self.position = max_seq_len, 0
        weights = model.state_dict()
        blob = bytearray(32 * 128)
        for i, full in enumerate(LAYERS):
            common = ["input_layernorm.weight"]
            if full:
                names = common + [
                    "self_attn." + n
                    for n in [
                        "q_proj.weight",
                        "k_proj.weight",
                        "v_proj.weight",
                        "q_norm.weight",
                        "k_norm.weight",
                        "o_proj.weight",
                    ]
                ]
            else:
                names = common + [
                    "linear_attn." + n
                    for n in [
                        "in_proj_qkv.weight",
                        "in_proj_z.weight",
                        "in_proj_b.weight",
                        "in_proj_a.weight",
                        "conv1d.weight",
                        "A_log",
                        "dt_bias",
                        "norm.weight",
                        "out_proj.weight",
                    ]
                ]
            names += [
                "post_attention_layernorm.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ]
            struct.pack_into("iiii", blob, i * 128, full, 0, 0, 0)
            for j, name in enumerate(names):
                t = weights[f"layers.{i}.{name}"]
                if not t.is_cuda or not t.is_contiguous():
                    raise ValueError("weights must be contiguous CUDA tensors")
                expected = (
                    torch.float32
                    if name.endswith(("A_log", "dt_bias"))
                    else torch.bfloat16
                )
                if t.dtype != expected:
                    raise ValueError(f"wrong dtype for {name}")
                struct.pack_into("Q", blob, i * 128 + 16 + j * 8, t.data_ptr())
        self.packed = torch.frombuffer(blob, dtype=torch.uint8).cuda()

        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(shape, dtype=dtype, device="cuda")

        self.k = zeros(8, 4, max_seq_len, 256, dtype=torch.bfloat16)
        self.v = torch.zeros_like(self.k)
        # CUDA layout [value, key], HF cache layout [key, value].
        self.state = zeros(24, 32, 128, 128)
        self.conv = zeros(24, 8192, 4)
        self.hidden = zeros(4096, dtype=torch.bfloat16)
        self.debug = zeros(33, 4096)
        self.residual = zeros(4096, dtype=torch.bfloat16)
        self.qkv, self.kv = zeros(8192), zeros(2048)
        self.attn, self.mlp = zeros(4096), zeros(12288)
        self.z, self.beta, self.alpha = zeros(4096), zeros(32), zeros(32)
        self.normalized = zeros(4096)
        self.barriers = [zeros(1, dtype=torch.int32), zeros(1, dtype=torch.int32)]
        self.seen = zeros(1)  # unused when repetition penalty is one
        self.tensors = [
            model.embed_tokens.weight,
            model.norm.weight,
            head,
            self.packed,
            self.k,
            self.v,
            self.state,
            self.conv,
            self.hidden,
            self.debug,
            self.residual,
            self.qkv,
            self.kv,
            self.attn,
            self.mlp,
            self.z,
            self.beta,
            self.alpha,
            self.normalized,
            *self.barriers,
            self.seen,
        ]
        self.pointers = (ctypes.c_void_p * len(self.tensors))(
            *(t.data_ptr() for t in self.tensors)
        )
        assert len(self.tensors) == 22

    @torch.inference_mode()
    def step(self, token, debug=False):
        if not 0 <= token < 248320 or self.position >= self.max_seq_len:
            raise ValueError("token or cache position out of bounds")
        code = self.library.decode(
            self.pointers,
            token,
            self.position,
            self.max_seq_len,
            int(debug),
            torch.cuda.current_stream().cuda_stream,
        )
        if code:
            raise RuntimeError(self.library.error_string(code).decode())
        self.position += 1
        return torch.nn.functional.linear(
            self.normalized.to(torch.bfloat16), self.head
        ).float()

    def reset(self):
        self.position = 0
        for t in (self.k, self.v, self.state, self.conv):
            t.zero_()

    @torch.inference_mode()
    def generate(self, ids, max_tokens=16, eos=None):
        if not ids or len(ids) + max_tokens > self.max_seq_len:
            raise ValueError("empty prompt or output exceeds cache capacity")
        self.reset()
        for token in ids:
            logits = self.step(token)
        out = []
        for i in range(max_tokens):
            token = int(logits.argmax())
            if token == eos:
                break
            out.append(token)
            if i + 1 < max_tokens:
                logits = self.step(token)
        torch.cuda.synchronize()
        return out

    @torch.inference_mode()
    def step_device(self, token):
        """Device token must come from a checked vocabulary argmax, on this stream."""
        if (
            token.device != self.head.device
            or token.dtype != torch.int64
            or token.numel() != 1
            or not token.is_contiguous()
            or self.position >= self.max_seq_len
        ):
            raise ValueError("Invalid device token or cache position")
        code = self.library.decode_device(
            self.pointers,
            token.data_ptr(),
            self.position,
            self.max_seq_len,
            0,
            torch.cuda.current_stream().cuda_stream,
        )
        if code:
            raise RuntimeError(self.library.error_string(code).decode())
        self.position += 1
        return torch.nn.functional.linear(
            self.normalized.to(torch.bfloat16), self.head
        ).float()

    @torch.inference_mode()
    def generate_buffered(self, ids, max_tokens=16, eos=None, capture="logits"):
        """Greedy decode with no per-token D2H read; one final CPU transfer per tensor.

        Row i is p(output_token[i] | prompt, output_tokens[:i]). EOS is excluded,
        like generate(). Fixed-size extra work after EOS is discarded. 'hidden'
        stores normalized BF16 hidden states and projects them together afterward.
        """
        if (
            not ids
            or max_tokens < 1
            or len(ids) + max_tokens > self.max_seq_len
            or capture not in ("logits", "hidden")
        ):
            raise ValueError("Invalid prompt, budget or capture mode")
        self.reset()
        for token in ids:
            logits = self.step(token)
        tokens = torch.empty(max_tokens, dtype=torch.int64, device=self.head.device)
        width = self.head.shape[0] if capture == "logits" else self.head.shape[1]
        buffer = torch.empty(
            (max_tokens, width),
            device=self.head.device,
            dtype=torch.float32 if capture == "logits" else torch.bfloat16,
        )
        for i in range(max_tokens):
            buffer[i].copy_(logits if capture == "logits" else self.normalized)
            torch.argmax(logits, dim=0, out=tokens[i])
            if i + 1 < max_tokens:
                logits = self.step_device(tokens[i])
        if capture == "hidden":
            buffer = torch.nn.functional.linear(buffer, self.head).float()
        cpu_tokens = tokens.cpu().tolist()
        stop = cpu_tokens.index(eos) if eos in cpu_tokens else max_tokens
        return cpu_tokens[:stop], buffer[:stop].cpu()
