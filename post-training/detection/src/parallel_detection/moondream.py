"""Moondream 2 as the baseline detector.

Its `detect` skill takes one phrase per call. After the image is encoded once, each
phrase is a prompt prefill followed by three decoder steps per object (x, y, size)
until the model emits end-of-sequence. Boxes come back without a confidence.

Decoders:

- "eager": the model's own `detect`.
- "compiled": the same after the model's `compile()` (torch.compile on its vision
  encoder, prefill and decode step).
- "loop": this module's copy of the detection loop, calling the PyTorch decoder
  directly. It runs the vocabulary head once per object, where the model's loop runs
  it on all three steps.
- "kernel": the same loop with the decoder step in custom CUDA kernels (`kernel.py`).
"""

import sys
import time

import torch
from transformers import AutoModelForCausalLM

DECODERS = ("eager", "compiled", "loop", "kernel")


class MoondreamDetector:
    def __init__(self, repo, revision, max_objects, device, decoder="eager", kernel_library=None):
        if decoder not in DECODERS:
            raise ValueError(f"Unknown decoder {decoder!r}")
        self.device = torch.device(device)
        self.max_objects = max_objects
        self.decoder = decoder
        self.model = AutoModelForCausalLM.from_pretrained(
            repo,
            revision=revision,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            device_map={"": device},
        ).eval()
        self.inner = self.model.model
        # The Hub module that defines the model also holds its region and text functions.
        self.functions = sys.modules[type(self.inner).__module__]
        if decoder == "compiled":
            self.inner.compile()
        if decoder == "kernel":
            from parallel_detection.kernel import KernelDecoder, build

            self.kernel = KernelDecoder(self.inner.text, build(kernel_library))

    def _clock(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def _step(self, embedding, position):
        """One decoder step for one input embedding; returns the hidden state [1, 1, dim]."""
        if self.decoder == "kernel":
            return self.kernel.step(embedding, position)
        mask = torch.zeros(1, 1, 2048, device=self.device, dtype=torch.bool)
        mask[:, :, : position + 1] = 1
        positions = torch.tensor([position], device=self.device, dtype=torch.long)
        return self.functions.text_decoder(
            embedding, self.inner.text, mask, positions, self.inner.config.text, None
        )

    @torch.inference_mode()
    def _detect_stepwise(self, encoded, phrase):
        """The model's detection loop (its `detect` and `_generate_points`), step by step."""
        m, f = self.inner, self.functions
        tokens = m.config.tokenizer
        m.load_encoded_image(encoded)
        template = tokens.templates["detect"]
        prompt = template["prefix"] + m.tokenizer.encode(" " + phrase).ids + template["suffix"]
        prompt = torch.tensor([prompt], device=self.device)
        _, hidden, next_token, pos = m._prefill_prompt(prompt, encoded.pos, temperature=0, top_p=0)
        hidden = hidden[:, -1:, :]
        if self.decoder == "kernel":
            caches = [(b.kv_cache.k_cache, b.kv_cache.v_cache) for b in m.text.blocks]
            self.kernel.load_cache(caches, encoded.pos, pos)
        objects = []
        while next_token.item() != tokens.eos_id and len(objects) < self.max_objects:
            x_logits = f.decode_coordinate(hidden, m.region)
            x = torch.argmax(x_logits, dim=-1) / x_logits.size(-1)
            embedding = f.encode_coordinate(x.to(dtype=x_logits.dtype), m.region).unsqueeze(0)
            hidden = self._step(embedding, pos)
            pos += 1
            y_logits = f.decode_coordinate(hidden, m.region)
            y = torch.argmax(y_logits, dim=-1) / y_logits.size(-1)
            embedding = f.encode_coordinate(y.to(dtype=y_logits.dtype), m.region).unsqueeze(0)
            hidden = self._step(embedding, pos)
            pos += 1
            size_logits = f.decode_size(hidden, m.region)
            # Sizes are 1,024 log-scale bins: size = 2^((bin / 1023) * 10 - 10).
            w = torch.pow(
                2.0, (torch.argmax(size_logits[0], dim=-1).float() / 1023.0) * 10.0 - 10.0
            )
            h = torch.pow(
                2.0, (torch.argmax(size_logits[1], dim=-1).float() / 1023.0) * 10.0 - 10.0
            )
            size = torch.tensor([w, h], device=self.device, dtype=size_logits.dtype)
            embedding = f.encode_size(size, m.region).unsqueeze(0).unsqueeze(0)
            objects.append(
                {
                    "x_min": x.item() - w.item() / 2,
                    "y_min": y.item() - h.item() / 2,
                    "x_max": x.item() + w.item() / 2,
                    "y_max": y.item() + h.item() / 2,
                }
            )
            hidden = self._step(embedding, pos)
            pos += 1
            next_token = torch.argmax(f.lm_head(hidden, m.text), dim=-1)
        return objects

    def detect(self, image, phrases):
        """Return {"boxes": per-phrase pixel boxes, "shared_seconds", "phrase_seconds"}."""
        width, height = image.size
        start = self._clock()
        encoded = self.model.encode_image(image)
        if self.decoder == "kernel":
            self.kernel.load_cache(encoded.caches, 0, encoded.pos)
        shared = self._clock() - start
        boxes, seconds = [], []
        for phrase in phrases:
            start = self._clock()
            if self.decoder in ("loop", "kernel"):
                objects = self._detect_stepwise(encoded, phrase)
            else:
                settings = {"max_objects": self.max_objects}
                objects = self.model.detect(encoded, phrase, settings=settings)["objects"]
            seconds.append(self._clock() - start)
            boxes.append(
                [
                    [
                        min(max(o["x_min"], 0.0), 1.0) * width,
                        min(max(o["y_min"], 0.0), 1.0) * height,
                        min(max(o["x_max"], 0.0), 1.0) * width,
                        min(max(o["y_max"], 0.0), 1.0) * height,
                    ]
                    for o in objects
                ]
            )
        return {"boxes": boxes, "shared_seconds": shared, "phrase_seconds": seconds}
