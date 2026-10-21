# Newest 124M base: executable repair evaluation

Evaluated `jfrick/124M-20.879B`: 124,389,632 parameters, 20,878,526,009
lifetime pretraining targets. Modal checkpoint:
`simple-llm-code60-unique-10b-20261020-r01-s04/step-000153342.pt`.
This is a base-model evaluation, not SFT repeated on the newer parent.

The identical 18 synthetic held-out tasks span two families (absolute difference
and maximum) and nine languages. Seven input/output cases per task; greedy BF16,
128-token cap, no output cleanup. Disposable network-blocked Modal sandboxes.
Reference/broken grader calibration passed in all nine languages.

| Model | Correct | Reached output comparison | EOS |
| --- | ---: | ---: | ---: |
| Older 10.879B-token base (rerun) | 0/18 | 0/18 | 0/18 |
| Newer 20.879B-token base | 0/18 | 3/18 | 1/18 |
| Older base + 20K private-patch SFT (previous evaluation) | 0/18 | 6/18 | 15/18 |

The newer base has 14 build/syntax failures, one runtime failure and three
runnable but incorrect outputs. It often preserves the original bug and continues
with repetitive text. More pretraining alone did not produce a successful repair
on this diagnostic. Results do not measure all held-out private patches or
establish performance after SFT on the newer parent.

[Full generations and diagnostics](generation-evaluation.json).
[Modal evaluation](https://modal.com/apps/radiantai/main/ap-YCppEhOQccrMq2GgHUiKTS).

## Actual Python output: maximum

The model starts with:

```python
def solve(a, b):
    return (a if a < b else b) + 11
```

It then appends repeated comments (`# Python 2`, `# Python 3`, ...) until the
128-token cap. For `a=-7, b=2`, the requested maximum plus 11 is 13; the output
is 4. The complete unedited response is stored in the JSON report.
