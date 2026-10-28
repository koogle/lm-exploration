"""Running model-written code: only in a network-blocked Modal Sandbox, or locally for
tests of our own fixtures. See `README.md` in this folder.

- `execution.py`: builds each test program, runs batches in a `Sandbox` (or `run_local`).
- `runner.py`: runs inside the sandbox; isolates and limits each program.
"""

from training_data.sandbox.execution import PROGRAM_SECONDS, Sandbox, program, run_local

__all__ = ["PROGRAM_SECONDS", "Sandbox", "program", "run_local"]
