import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import modal_live_publication as publication


class PublicationRefreshTests(unittest.TestCase):
    def test_warm_worker_refreshes_before_opening_new_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "new-segment.pt"
            with (
                patch.object(publication, "checked_checkpoint", return_value=checkpoint),
                patch.object(
                    publication.run_volume,
                    "reload",
                    side_effect=lambda: checkpoint.write_bytes(b"committed"),
                ) as reload,
            ):
                self.assertEqual(publication.refresh_checkpoint("run", "checkpoint"), checkpoint)
                reload.assert_called_once()

    def test_missing_checkpoint_remains_an_explicit_failure(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(
                publication, "checked_checkpoint", return_value=Path(temporary) / "missing.pt"
            ),
            patch.object(publication.run_volume, "reload"),
            self.assertRaises(FileNotFoundError),
        ):
            publication.refresh_checkpoint("run", "checkpoint")
