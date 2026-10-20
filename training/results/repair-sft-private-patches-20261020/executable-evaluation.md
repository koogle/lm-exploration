# Executable held-out evaluation after 20,000 private-patch SFT updates

Tested the final checkpoint on the same 18 synthetic held-out tasks used for the
pilot: absolute difference and maximum, each in Python, JavaScript, TypeScript,
Go, Rust, C, C++, Java and Bash. Each task has seven input/output test cases.
Generation uses the identical prompt, greedy BF16 decoding and 128-token cap.
No output cleanup or retries. Code executes in disposable network-blocked Modal
sandboxes; reference/broken grader calibration passed in all nine languages.

| Model | Correct tasks | Reached output comparison | Emitted EOS |
| --- | ---: | ---: | ---: |
| Original PR #26 base (rerun) | 0/18 | 0/18 | 0/18 |
| 200-update arithmetic pilot (historical) | 0/18 | 18/18 | 18/18 |
| 20,000-update private-patch SFT | 0/18 | 6/18 | 15/18 |

The latest model has 12 build/syntax failures and six runnable but incorrect
outputs. Duplicate definitions are common. Lower teacher-forced validation loss
has not yielded a passing task on this small diagnostic. This does not establish
performance on all held-out data: these are two synthetic task families, not
execution-verified patches against the private repositories. The planned full
corpus run has no checkpoint and is not included.

Full raw generations, observed/expected outputs, compiler diagnostics and dataset
hash are in [generation-evaluation.json](generation-evaluation.json).
[Modal evaluation](https://modal.com/apps/radiantai/main/ap-8Un24S9vsHg7RqRUv7kSDU).

## Return the absolute difference of a and b, plus 11. Preserve the solve function signature. Inputs are small signed integers.

Actual generated response:

```python
def solve(a, b):
    return a - b + 11

def solve(a, b):
    return a - b - 11

```

For inputs (-7, 2), expected 20, observed -20. The task fails.

## Return the larger of a and b, plus 11. Preserve the solve function signature. Inputs are small signed integers.

Actual generated response:

```python
def solve(a, b):
    return (a if a < b else b) + 11

def solve2(a, b):
    return (a if a < b else b) + 11

```

For inputs (-7, 2), expected 13, observed 4. The task fails.

