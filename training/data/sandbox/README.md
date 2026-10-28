# Code sandbox

Runs model-written Python against unit tests: `training_data.sandbox`.

[`execution.py`](execution.py) runs code only in a Modal Sandbox with
`block_network=True`, no volumes or secrets, and a timeout that ends at the caller's
launch deadline. [`runner.py`](runner.py) runs each program as `python -I -` in a fresh
directory with 5 s of CPU, 1 GB of memory and a 16 MB file-size limit, in its own
process group that is killed when it ends. Each concurrently running program has its own
unprivileged user ID and a private directory, so it cannot change another program's
files or the captured output; a program the runner cannot start fails on its own instead
of stopping the batch. (A first full teacher run stopped when one generated program
deleted the other programs' directories in `/tmp`.) The program imports the solution,
runs the tests, calls every `test*` function the tests define and prints a random per-
program nonce last; it passes only with exit code 0 and the nonce as its last line, so a
solution that exits early or skips the tests fails.

`run_local` runs a batch with the same runner in a local subprocess, for tests of our
own fixtures only.
