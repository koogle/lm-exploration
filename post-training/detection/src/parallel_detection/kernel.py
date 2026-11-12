"""Custom CUDA decode step for Moondream 2's text decoder, bound through ctypes.

`decode_step.cu` runs one batch-one decoder step in 72 kernel launches queued by a
single host call. It reads the model's BF16 weight tensors in place and keeps its own
FP32 key/value cache, which `load_cache` fills from PyTorch's cache after the image and
prompt prefill.
"""

import ctypes
import subprocess
from pathlib import Path

import torch

SOURCE = Path(__file__).parent / "decode_step.cu"
LAYERS, DIM, FF, HEADS, HEAD_DIM, MAX_CONTEXT = 24, 2048, 8192, 32, 64, 2048
# Shapes of one block's tensors, in the kernel's Layer field order.
LAYER_SHAPES = [
    (DIM,),
    (DIM,),
    (3 * DIM, DIM),
    (3 * DIM,),
    (DIM, DIM),
    (DIM,),
    (FF, DIM),
    (FF,),
    (DIM, FF),
    (DIM,),
]


def build(destination):
    """Compile the kernels into a shared library with nvcc."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ["nvcc", "-std=c++17", "-O3", "--shared", "-Xcompiler=-fPIC"]
    result = subprocess.run(
        [*command, str(SOURCE), "-o", str(destination)],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-10000:])
    return destination


def layer_tensors(block):
    return [
        block.ln.weight,
        block.ln.bias,
        block.attn.qkv.weight,
        block.attn.qkv.bias,
        block.attn.proj.weight,
        block.attn.proj.bias,
        block.mlp.fc1.weight,
        block.mlp.fc1.bias,
        block.mlp.fc2.weight,
        block.mlp.fc2.bias,
    ]


class KernelDecoder:
    def __init__(self, text, library):
        if len(text.blocks) != LAYERS:
            raise ValueError(f"Expected {LAYERS} decoder blocks, found {len(text.blocks)}")
        self.weights = [t for block in text.blocks for t in layer_tensors(block)]
        for tensor, shape in zip(self.weights, LAYER_SHAPES * LAYERS, strict=True):
            if tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16:
                raise ValueError(f"Unexpected weight {tuple(tensor.shape)} {tensor.dtype}")
            if not tensor.is_contiguous() or not tensor.is_cuda:
                raise ValueError("Weights must be contiguous CUDA tensors")
        cache = text.blocks[0].kv_cache.k_cache
        if tuple(cache.shape) != (1, HEADS, MAX_CONTEXT, HEAD_DIM):
            raise ValueError(f"Unexpected key cache shape {tuple(cache.shape)}")
        device = cache.device

        def zeros(*shape):
            return torch.zeros(*shape, device=device, dtype=torch.float32)

        self.x, self.qkv, self.h, self.attn = zeros(DIM), zeros(3 * DIM), zeros(FF), zeros(DIM)
        self.keys = zeros(LAYERS, HEADS, MAX_CONTEXT, HEAD_DIM)
        self.values = zeros(LAYERS, HEADS, MAX_CONTEXT, HEAD_DIM)
        # Cosine and sine per position and rotated pair, at the precision the model holds.
        self.rope = text.freqs_cis.to(device=device, dtype=torch.float32).contiguous()
        if tuple(self.rope.shape) != (MAX_CONTEXT, 16, 2):
            raise ValueError(f"Unexpected rotary table shape {tuple(self.rope.shape)}")
        state = [self.x, self.qkv, self.h, self.attn, self.keys, self.values, self.rope]

        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.library.md_init.argtypes = [ctypes.POINTER(ctypes.c_void_p)] * 2
        self.library.md_step.argtypes = [ctypes.c_int]
        self.library.md_error_string.argtypes = [ctypes.c_int]
        self.library.md_error_string.restype = ctypes.c_char_p
        layers = (ctypes.c_void_p * len(self.weights))(*(t.data_ptr() for t in self.weights))
        pointers = (ctypes.c_void_p * len(state))(*(t.data_ptr() for t in state))
        self._check(self.library.md_init(layers, pointers))

    def _check(self, code):
        if code:
            raise RuntimeError(self.library.md_error_string(code).decode())

    def load_cache(self, caches, start, end):
        """Copy positions [start, end) of per-layer (key, value) tensors shaped [1, H, T, D]."""
        for layer, (key, value) in enumerate(caches):
            self.keys[layer, :, start:end] = key[0, :, start:end]
            self.values[layer, :, start:end] = value[0, :, start:end]

    def step(self, embedding, position):
        """Run all layers for one input embedding; returns the hidden state as [1, 1, DIM]."""
        self.x.copy_(embedding.reshape(DIM))
        self._check(self.library.md_step(position))
        return self.x.to(embedding.dtype).view(1, 1, DIM)
