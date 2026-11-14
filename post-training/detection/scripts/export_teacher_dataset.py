"""Write teacher-labelled training images as Parquet shards for a Hugging Face dataset.

    python scripts/export_teacher_dataset.py --labels artifacts/teacher-train/google--gemini-4-carbon.jsonl \
        --images artifacts/teacher-images --out artifacts/teacher-dataset

Each row holds the stored JPEG, its source, size and the teacher's boxes ([x0, y0, x1,
y1] in pixels of that image) with COCO category names. Replies that failed are skipped.
"""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

FEATURES = {
    "image": {"_type": "Image"},
    "image_id": {"dtype": "string", "_type": "Value"},
    "source": {"dtype": "string", "_type": "Value"},
    "width": {"dtype": "int32", "_type": "Value"},
    "height": {"dtype": "int32", "_type": "Value"},
    "objects": [
        {
            "label": {"dtype": "string", "_type": "Value"},
            "box": {"feature": {"dtype": "float32", "_type": "Value"}, "_type": "Sequence"},
        }
    ],
    "model": {"dtype": "string", "_type": "Value"},
    "cost_usd": {"dtype": "float64", "_type": "Value"},
}
SCHEMA = pa.schema(
    [
        ("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ("image_id", pa.string()),
        ("source", pa.string()),
        ("width", pa.int32()),
        ("height", pa.int32()),
        ("objects", pa.list_(pa.struct([("label", pa.string()), ("box", pa.list_(pa.float32()))]))),
        ("model", pa.string()),
        ("cost_usd", pa.float64()),
    ],
    metadata={"huggingface": json.dumps({"info": {"features": FEATURES}})},
)


def export(labels, images, out, rows_per_shard):
    out = Path(out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    replies = [json.loads(line) for line in Path(labels).read_text().splitlines()]
    replies = [r for r in replies if "error" not in r]
    shards = (len(replies) + rows_per_shard - 1) // rows_per_shard
    for shard in range(shards):
        rows = []
        for reply in replies[shard * rows_per_shard : (shard + 1) * rows_per_shard]:
            name = f"{reply['image_id']}.jpg"
            rows.append(
                {
                    "image": {"bytes": (Path(images) / name).read_bytes(), "path": name},
                    "image_id": reply["image_id"],
                    "source": reply["image_id"].split("/")[0],
                    "width": reply["width"],
                    "height": reply["height"],
                    "objects": reply["objects"],
                    "model": reply["model"],
                    "cost_usd": reply["cost"],
                }
            )
        path = out / "data" / f"train-{shard:05d}-of-{shards:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
    return len(replies), shards


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows-per-shard", type=int, default=3000)
    args = parser.parse_args()
    print(export(args.labels, args.images, args.out, args.rows_per_shard))


if __name__ == "__main__":
    main()
