"""Moondream models served by Photon, Moondream's own inference engine.

`pip install moondream` brings the engine and its precompiled kernels. It is the only
published way to run Moondream 3.1, and it also serves Moondream 2. Photon keeps a
prefix cache, so the first phrase for an image pays for the image encoding.
"""

import time


class PhotonDetector:
    def __init__(self, name, max_objects, device):
        import moondream

        if device != "cuda":
            raise ValueError("The Photon adapter is set up for CUDA only")
        self.model = moondream.photon(name)
        self.max_objects = max_objects

    def detect(self, image, phrases):
        """Return {"boxes": per-phrase pixel boxes, "shared_seconds", "phrase_seconds"}."""
        width, height = image.size
        start = time.perf_counter()
        encoded = self.model.encode_image(image)  # JPEG bytes; the engine encodes on first use
        shared = time.perf_counter() - start
        boxes, seconds = [], []
        for phrase in phrases:
            start = time.perf_counter()
            objects = self.model.detect(encoded, phrase, settings={"max_objects": self.max_objects})
            seconds.append(time.perf_counter() - start)
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
