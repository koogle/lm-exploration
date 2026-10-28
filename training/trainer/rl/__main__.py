"""Print an RL recipe's settings with every default filled in; no Modal, data or GPU.

uv run python -m trainer.rl --recipe rl/recipes/<recipe>.json
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from trainer.rl.settings import Settings

parser = argparse.ArgumentParser()
parser.add_argument("--recipe", required=True)
recipe = json.loads(Path(parser.parse_args().recipe).read_text())
print(json.dumps(asdict(Settings.from_dict(recipe["settings"])), indent=2))
