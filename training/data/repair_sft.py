"""Code-repair SFT rows (`repair-sft-v2`); no supplied code is executed during preparation.

The prompt gives the language, a task and broken code; the answer is the complete
corrected code (see `training_data.sft` for what is supervised).
"""

import re

from training_data.sft import SFTDataset


class RepairSFT(SFTDataset):
    FORMAT = "repair-sft-v2"
    FIELDS = ("language", "instruction", "broken", "fixed")

    @staticmethod
    def language(row):
        value = row.get("language", "")
        if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_+#.-]{0,39}", value):
            raise ValueError("language must be an explicit lowercase language identifier")
        return value

    def prompt(self, row):
        return (
            f"Repair the {self.language(row)} code to satisfy the task. "
            "Return only the complete corrected code.\n"
            f"Task: {row['instruction']}\nBroken code:\n{row['broken']}\nCorrected code:\n"
        )

    def answer(self, row):
        return row["fixed"]

    def bucket(self, row):
        return self.language(row)

    def fingerprints(self, row):
        return (row["broken"], row["fixed"])

    def check(self, row):
        if row["broken"].strip() == row["fixed"].strip():
            raise ValueError("Repair must change code")

    def check_splits(self, splits):
        languages = {split: {r["bucket"] for r in rows} for split, rows in splits.items()}
        missing = languages["train"] - languages["validation"]
        if missing:
            raise ValueError(f"Missing validation languages: {sorted(missing)}")


REPAIR = RepairSFT()
