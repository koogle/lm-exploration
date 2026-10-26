import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from training_data.hub_files import read_rows

ROWS = [{"question": "a", "answer": 1}, {"question": "b", "answer": 2}]


class HubFileTests(unittest.TestCase):
    def test_reads_parquet_jsonl_and_json_rows_in_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pq.write_table(pa.Table.from_pylist(ROWS), root / "rows.parquet")
            (root / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ROWS) + "\n")
            (root / "rows.json").write_text(json.dumps(ROWS))
            for name in ("rows.parquet", "rows.jsonl", "rows.json"):
                self.assertEqual(read_rows(root / name), ROWS, name)

    def test_rejects_other_files(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "rows.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "JSON array"):
                read_rows(Path(folder) / "rows.json")
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                read_rows(Path(folder) / "rows.csv")


if __name__ == "__main__":
    unittest.main()
