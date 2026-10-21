"""Modal CLI entrypoint honoring the managed environment's verified CA bundle."""

import os
from pathlib import Path

import certifi

if os.environ.get("SSL_CERT_FILE") and Path(os.environ["SSL_CERT_FILE"]).is_file():
    # grpclib calls certifi.where() explicitly; retain TLS verification and proxy CA.
    certifi.where = lambda: os.environ["SSL_CERT_FILE"]

from modal.__main__ import main

if __name__ == "__main__":
    # The official entrypoint enables SDK output and CLI error reporting. Calling
    # only the Click group can silently suppress deployment and app-list output.
    main()
