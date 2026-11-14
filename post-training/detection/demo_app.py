"""Detection demo on Modal: compare detectors on images and run one on a video.

Run from post-training/detection/:

    .venv/bin/python demo_app.py        # deploy, then arm the shutdown 24 hours later

Models and example images are listed in `configs/demo.json`. One L40S container loads
Moondream 2 once and shares it between its own `detect` skill and the slot heads;
Moondream 3.1 runs in a second L40S container through Photon. Both scale to zero after
five idle minutes, and each call is limited to 30 minutes. The web page runs on a small
CPU container. `shutdown_at` stops the whole app at the deadline set when it is deployed,
so the demo stops on its own within 24 hours.
"""

import io
import json
import os
import tempfile
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent if modal.is_local() else Path("/root")
APP = "lm-exploration-detection-demo"
CONFIG = json.loads((HERE / "configs" / "demo.json").read_text())
LIFETIME_SECONDS = 24 * 60 * 60 - 900
CALL_SECONDS = 30 * 60

app = modal.App(APP)
volume = modal.Volume.from_name("lm-exploration-parallel-detection")
PACKAGE = HERE / "src" / "parallel_detection" if modal.is_local() else HERE / "parallel_detection"


def with_source(image):
    return (
        image.env({"HF_HOME": "/runs/hf-cache"})
        .add_local_file(HERE / "configs" / "demo.json", "/root/configs/demo.json")
        .add_local_dir(PACKAGE, "/root/parallel_detection")
    )


gpu_image = with_source(
    modal.Image.debian_slim(python_version="3.12").pip_install(
        "torch==2.6.0",
        "transformers==4.52.4",
        "accelerate==1.7.0",
        "einops==0.8.1",
        "pyvips-binary==8.16.0",
        "pyvips==2.2.3",
        "numpy==2.2.6",
        "scipy==1.15.3",
        "pillow==11.2.1",
        "pyarrow==20.0.0",
        "huggingface-hub==0.32.4",
        "imageio-ffmpeg==0.8.0",
    )
)
photon_image = with_source(
    modal.Image.debian_slim(python_version="3.12").pip_install(
        "moondream==2.6.1",
        "numpy==2.2.6",
        "scipy==1.15.3",
        "pillow==11.2.1",
        "imageio-ffmpeg==0.8.0",
    )
)
# Gradio's streaming video output probes each chunk with ffmpeg and ffprobe.
web_image = with_source(
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("gradio==6.31.0", "fastapi==0.162.5", "pillow==11.2.1")
)


def found_in(result, phrases):
    from parallel_detection.demo import detections

    return detections(result, phrases)


@app.cls(
    image=gpu_image,
    gpu="L40S",
    volumes={"/runs": volume},
    timeout=CALL_SECONDS,
    retries=0,
    max_containers=1,
    scaledown_window=300,
)
class Detectors:
    @modal.enter()
    def load(self):
        from parallel_detection.moondream import MoondreamDetector
        from parallel_detection.train_slots import SlotDetector, load_backbone

        moondream = CONFIG["moondream"]
        backbone = load_backbone(moondream["repo"], moondream["revision"], device="cuda")
        self.models = {
            "Moondream 2 detect": MoondreamDetector(
                moondream["repo"], moondream["revision"], 150, "cuda", model=backbone[0]
            )
        }
        for name, head in CONFIG["slot_heads"].items():
            self.models[name] = SlotDetector(
                f"/runs/slots/{head['run']}/head.pt",
                "cuda",
                threshold=head["threshold"],
                backbone=backbone,
                **head.get("options", {}),
            )

    def _detect(self, name, image, phrases):
        from parallel_detection.tasks import COCO_NAMES

        if name != "Moondream 2 detect":
            phrases = [p for p in phrases if p in COCO_NAMES]  # the heads know these only
        start = time.perf_counter()
        result = self.models[name].detect(image, phrases)
        return found_in(result, phrases), time.perf_counter() - start

    @modal.method()
    def detect(self, data, phrases, names):
        """Per model: detections and seconds on one image."""
        from PIL import Image

        image = Image.open(io.BytesIO(data)).convert("RGB")
        self._detect(names[0], image, phrases)  # warm the first model's kernels
        out = {}
        for name in names:
            found, seconds = self._detect(name, image, phrases)
            out[name] = {"found": found, "seconds": seconds}
        return out

    @modal.method()
    def video(self, data, name, phrases, every, max_frames):
        """Annotated video chunks and statistics, as they are produced."""
        from parallel_detection.demo import stream_video

        yield from stream_video(
            data, lambda image: self._detect(name, image, phrases)[0], every, max_frames
        )


@app.cls(
    image=photon_image,
    gpu="L40S",
    volumes={"/runs": volume},
    timeout=CALL_SECONDS,
    retries=0,
    max_containers=1,
    scaledown_window=300,
)
class Photon:
    @modal.enter()
    def load(self):
        from parallel_detection.photon import PhotonDetector

        self.detector = PhotonDetector(CONFIG["photon_model"], 150, "cuda")

    def _detect(self, image, phrases):
        start = time.perf_counter()
        result = self.detector.detect(image, phrases)
        return found_in(result, phrases), time.perf_counter() - start

    @modal.method()
    def detect(self, data, phrases):
        from PIL import Image

        image = Image.open(io.BytesIO(data)).convert("RGB")
        self._detect(image, phrases[:1])  # warm up
        found, seconds = self._detect(image, phrases)
        return {"found": found, "seconds": seconds}

    @modal.method()
    def video(self, data, phrases, every, max_frames):
        """Annotated video chunks and statistics, as they are produced."""
        from parallel_detection.demo import stream_video

        yield from stream_video(
            data, lambda image: self._detect(image, phrases)[0], every, max_frames
        )


MOONDREAM_31 = "Moondream 3.1 (Photon)"


@app.function(image=web_image, volumes={"/runs": volume}, timeout=CALL_SECONDS, max_containers=1)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web():
    import gradio as gr
    from fastapi import FastAPI

    from parallel_detection.demo import draw, phrases_of

    names = ["Moondream 2 detect", *CONFIG["slot_heads"], MOONDREAM_31]
    examples = [str(Path("/runs") / path) for path in CONFIG["examples"]]

    def compare(image, text, chosen):
        if image is None or not chosen:
            return [], "Choose an image and at least one model."
        phrases = phrases_of(text)
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=95)
        data = buffer.getvalue()
        local = [n for n in chosen if n != MOONDREAM_31]
        results = Detectors().detect.remote(data, phrases, local) if local else {}
        if MOONDREAM_31 in chosen:
            results[MOONDREAM_31] = Photon().detect.remote(data, phrases)
        gallery, rows = [], []
        for name in chosen:
            r = results[name]
            caption = f"{name}: {len(r['found'])} boxes, {r['seconds']:.3f} s"
            gallery.append((draw(image, r["found"], caption), caption))
            rows.append(f"| {name} | {len(r['found'])} | {r['seconds']:.3f} |")
        table = "| Model | Boxes | Seconds |\n|---|---|---|\n" + "\n".join(rows)
        note = (
            "Moondream 2 and 3.1 answer one category per pass, so their time grows with the "
            "number of categories; the slot heads answer all categories in one pass."
        )
        return gallery, table + "\n\n" + note

    def video(path, text, name, every, max_frames):
        """Stream the annotated video chunk by chunk while the GPU works through it."""
        if not path:
            yield None, "Upload a video."
            return
        data = Path(path).read_bytes()
        phrases = phrases_of(text)
        if name == MOONDREAM_31:
            chunks = Photon().video.remote_gen(data, phrases, int(every), int(max_frames))
        else:
            chunks = Detectors().video.remote_gen(data, name, phrases, int(every), int(max_frames))
        folder = Path(tempfile.mkdtemp(prefix="stream-"))
        for index, (chunk, stats) in enumerate(chunks):
            target = folder / f"chunk-{index:05d}.ts"
            target.write_bytes(chunk)
            yield str(target), "```json\n" + json.dumps(stats, indent=2) + "\n```"

    with gr.Blocks(title="Detection demo") as page:
        gr.Markdown(
            "## Detection demo\nCompare Moondream 2, Moondream 3.1 and our one-pass slot "
            "heads. Categories: comma-separated COCO names; empty means all 80. The GPUs "
            "start on the first request (about a minute) and stop after five idle minutes.\n\n"
            "Slot-head boxes show their score: objectness × label confidence × "
            "(box confidence)^0.25, a ranking score rather than a probability; boxes are "
            'kept from about 0.3. "Fast" heads read the full image once; "tiled" also '
            "reads a 2 × 2 grid of tiles for small objects, at about 2.5 times the time. Moondream "
            "gives no confidence, so its boxes show only the label. Videos stream in "
            "two-second chunks as they are processed."
        )
        with gr.Tab("Images"):
            with gr.Row():
                picture = gr.Image(type="pil", label="Image")
                with gr.Column():
                    text = gr.Textbox(label="Categories", value="person, car, dog, cup, chair")
                    chosen = gr.CheckboxGroup(
                        names, value=[names[0], names[-2], names[-1]], label="Models"
                    )
                    run = gr.Button("Detect", variant="primary")
            gallery = gr.Gallery(label="Results", columns=2, height="auto")
            summary = gr.Markdown()
            run.click(compare, [picture, text, chosen], [gallery, summary])
            gr.Examples([[e] for e in examples if os.path.exists(e)], [picture])
        with gr.Tab("Video"):
            with gr.Row():
                clip = gr.Video(label="Video")
                with gr.Column():
                    vtext = gr.Textbox(label="Categories", value="")
                    model = gr.Dropdown(names, value="Slots, rank 128, fast", label="Model")
                    every = gr.Slider(1, 30, value=3, step=1, label="Detect every n-th frame")
                    limit = gr.Slider(30, 3000, value=600, step=30, label="Frames to process")
                    vrun = gr.Button("Run", variant="primary")
            annotated = gr.Video(label="Annotated", streaming=True, autoplay=True)
            stats = gr.Markdown()
            vrun.click(video, [clip, vtext, model, every, limit], [annotated, stats])

    return gr.mount_gradio_app(FastAPI(), page, path="/")


@app.function(image=web_image, cpu=0.125, memory=256, timeout=24 * 60 * 60, retries=0)
async def shutdown_at(deadline):
    """Independent supervisor: stop this whole app at `deadline` (Unix seconds)."""
    import asyncio

    from modal.client import _Client
    from modal_proto import api_pb2

    await asyncio.sleep(max(0.0, deadline - time.time()))
    client = await _Client.from_env()
    deployed = await modal.App.lookup.aio(APP)
    request = api_pb2.AppStopRequest(
        app_id=deployed.app_id, source=api_pb2.APP_STOP_SOURCE_PYTHON_CLIENT
    )
    await client._stub.AppStop(request)


if __name__ == "__main__":
    deadline = time.time() + LIFETIME_SECONDS
    with modal.enable_output():
        app.deploy()
    call = modal.Function.from_name(APP, "shutdown_at").spawn(deadline)
    record = {"app": APP, "deadline_unix": deadline, "shutdown_call": call.object_id}
    (HERE / "artifacts").mkdir(exist_ok=True)
    (HERE / "artifacts" / "demo-deploy.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    print(f"Web page: {modal.Function.from_name(APP, 'web').get_web_url()}")
