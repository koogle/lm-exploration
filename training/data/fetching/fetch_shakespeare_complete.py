"""Download Project Gutenberg's Complete Works of Shakespeare (eBook #100) to data/raw/.

Verifies the pinned SHA-256 of the download, removes the Project Gutenberg header
and license footer, normalizes CRLF to LF and saves only the works themselves.
Gutenberg updates this eBook in place; a new revision fails the check instead of
silently changing the dataset.

From training/: uv run python -m training_data.fetching.fetch_shakespeare_complete
"""

import hashlib
import urllib.request
from pathlib import Path

URL = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
# Revision "Most recently updated: August 24, 2025".
SOURCE_SHA256 = "3cf4b3d44ee14cff4e14e78e2ad3318eff76f3f7f2afc3cee6bb925879110a37"
SHA256 = "bfb5a5764506115cbae33bfdf66014697ef69d06339f001db209d3f2a56c83fc"
START = "*** START OF THE PROJECT GUTENBERG EBOOK"
END = "*** END OF THE PROJECT GUTENBERG EBOOK"
DESTINATION = Path(__file__).resolve().parents[1] / "raw" / "shakespeare-complete.txt"


def strip_gutenberg(text):
    text = text.lstrip("﻿").replace("\r\n", "\n")
    start = text.index("\n", text.index(START)) + 1
    end = text.index(END)
    return text[start:end].strip("\n") + "\n"


def main():
    if DESTINATION.is_file() and hashlib.sha256(DESTINATION.read_bytes()).hexdigest() == SHA256:
        print(f"Already present: {DESTINATION}")
        return
    with urllib.request.urlopen(URL, timeout=60) as response:
        source = response.read()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise SystemExit("Downloaded file does not match the pinned SHA-256; not saved")
    content = strip_gutenberg(source.decode("utf-8")).encode("utf-8")
    if hashlib.sha256(content).hexdigest() != SHA256:
        raise SystemExit("Stripped text does not match the pinned SHA-256; not saved")
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    temporary = DESTINATION.with_suffix(".tmp")
    temporary.write_bytes(content)
    temporary.replace(DESTINATION)
    print(f"Saved {len(content):,} bytes to {DESTINATION}")


if __name__ == "__main__":
    main()
