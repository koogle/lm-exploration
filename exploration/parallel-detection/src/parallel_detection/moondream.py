"""Moondream 2 as the baseline detector.

Its `detect` skill takes one phrase per call. After the image is encoded once, each
phrase is a prompt prefill followed by three decoder steps per object (x, y, size)
until the model emits end-of-sequence. Boxes come back without a confidence.
"""

import time

import torch
from transformers import AutoModelForCausalLM


class MoondreamDetector:
    def __init__(self, repo, revision, max_objects, device):
        self.device = torch.device(device)
        self.max_objects = max_objects
        self.model = AutoModelForCausalLM.from_pretrained(
            repo,
            revision=revision,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            device_map={"": device},
        ).eval()

    def _clock(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def detect(self, image, phrases):
        """Return {"boxes": per-phrase pixel boxes, "shared_seconds", "phrase_seconds"}."""
        width, height = image.size
        start = self._clock()
        encoded = self.model.encode_image(image)
        shared = self._clock() - start
        boxes, seconds = [], []
        for phrase in phrases:
            start = self._clock()
            objects = self.model.detect(encoded, phrase, settings={"max_objects": self.max_objects})
            seconds.append(self._clock() - start)
            boxes.append(
                [
                    [
                        min(max(o["x_min"], 0.0), 1.0) * width,
                        min(max(o["y_min"], 0.0), 1.0) * height,
                        min(max(o["x_max"], 0.0), 1.0) * width,
                        min(max(o["y_max"], 0.0), 1.0) * height,
                    ]
                    for o in objects["objects"]
                ]
            )
        return {"boxes": boxes, "shared_seconds": shared, "phrase_seconds": seconds}
