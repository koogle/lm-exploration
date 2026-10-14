"""Download Tiny Shakespeare to data/raw/ (ignored by Git) and verify its SHA-256.

From training/: uv run python -m training_data.fetching.fetch_tinyshakespeare
"""

import hashlib
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "370cbcd448eb7daf32f21a6be560b70e0b33c4e3/data/tinyshakespeare/input.txt"
)
SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
DESTINATION = Path(__file__).resolve().parents[1] / "raw" / "tinyshakespeare.txt"


def main():
    if DESTINATION.is_file() and hashlib.sha256(DESTINATION.read_bytes()).hexdigest() == SHA256:
        print(f"Already present: {DESTINATION}")
        return
    with urllib.request.urlopen(URL, timeout=60) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != SHA256:
        raise SystemExit("Downloaded file does not match the pinned SHA-256; not saved")
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    temporary = DESTINATION.with_suffix(".tmp")
    temporary.write_bytes(content)
    temporary.replace(DESTINATION)
    print(f"Saved {len(content):,} bytes to {DESTINATION}")


if __name__ == "__main__":
    main()
