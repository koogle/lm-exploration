import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training_data.cached_shards import CachedShards


class CachedShardsTests(unittest.TestCase):
    def test_local_shard_resume_and_old_shard_eviction(self):
        source = {"name": "code_python", "format": "json.gz", "repo": "test", "revision": "0" * 40}
        with tempfile.TemporaryDirectory() as temporary:

            def download(**kwargs):
                path = Path(kwargs["local_dir"]) / kwargs["filename"]
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    with gzip.open(path, "wt") as handle:
                        for i in range(3):
                            handle.write(json.dumps({"row": i}) + "\n")
                return str(path)

            reader = CachedShards(temporary)
            with patch("huggingface_hub.hf_hub_download", side_effect=download) as fetch:
                self.assertEqual(
                    list(reader(source, "a.gz", start=1)), [(1, {"row": 1}), (2, {"row": 2})]
                )
                self.assertTrue((Path(temporary) / "code_python/a.gz").exists())
                list(reader(source, "b.gz"))
                self.assertFalse((Path(temporary) / "code_python/a.gz").exists())
                self.assertTrue((Path(temporary) / "code_python/b.gz").exists())
                self.assertEqual(fetch.call_args.kwargs["revision"], source["revision"])


if __name__ == "__main__":
    unittest.main()
