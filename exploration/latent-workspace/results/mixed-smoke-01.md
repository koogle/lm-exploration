# Conditioning experiment results

| Condition | Steps | All-masked NLL ↓ | Δ vs diffusion final ↓ | Greedy token accuracy | Block ms/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| diffusion:final | 5 | 17.1842 | +0.0000 | 0.00% | 7.43 |
| diffusion:all | 5 | 16.8165 | -0.3678 | 0.00% | 7.44 |
| diffusion:jspace | 5 | 18.5860 | +1.4018 | 0.00% | 9.27 |

Negative deltas favor the condition. These are exploratory point estimates,
not significance tests. All-masked NLL is not iterative sequence likelihood.
Latency does not imply a speedup at matched quality.

Model revision: `060db6499f32faf8b98477b0a26969ef7d8b9987`.
Source SHA-256: `401a9f139485274f63fbc1ad7959e62736c25c42bf311f4c4f25710969dfdc11`.
GPU: NVIDIA L40S. Elapsed: 67.6s.

## diffusion:final: example continuations

Prompt: The capital of France is

```text
 sich sich sich sich sich sich sich sich sich sich sich sich sich sich sich sich
```

Prompt: Once upon a time

```text
 sich sich sich sich sich sich sich sich sich sich sich sich sich sich sich sich
```

## diffusion:all: example continuations

Prompt: The capital of France is

```text
 sich sich sich sich sich sich sich sich sich sich sich sich sich sich sich sich
```

Prompt: Once upon a time

```text
 sich sich sich sich sich sich sich sich sich sich sich sich sich sich sich sich
```

## diffusion:jspace: example continuations

Prompt: The capital of France is

```text
 sich sich sich sich sich sich sich sich sich sich sich sich sich sich sich sich
```

Prompt: Once upon a time

```text
 sich sich sich sich sich sich sich sich sich sich sich sich sich sich sich sich
```
