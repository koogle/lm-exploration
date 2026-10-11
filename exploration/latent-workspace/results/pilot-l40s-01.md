# Conditioning experiment results

| Condition | Steps | All-masked NLL ↓ | Δ vs diffusion final ↓ | Greedy token accuracy | Block ms/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| parallel:final | 750 | 7.5224 | -0.0510 | 3.42% | 4.10 |
| diffusion:final | 750 | 7.5734 | +0.0000 | 3.22% | 6.96 |
| diffusion:layer12 | 750 | 7.5371 | -0.0363 | 3.03% | 6.90 |
| diffusion:multi | 750 | 7.5513 | -0.0221 | 3.42% | 6.99 |
| diffusion:attention12 | 750 | 7.5561 | -0.0174 | 3.52% | 6.90 |
| diffusion:attention_multi | 750 | 7.5786 | +0.0052 | 3.42% | 7.08 |
| diffusion:jspace | 750 | 7.6207 | +0.0473 | 1.76% | 10.60 |

Negative deltas favor the condition. These are exploratory point estimates,
not significance tests. All-masked NLL is not iterative sequence likelihood.
Latency does not imply a speedup at matched quality.

Model revision: `060db6499f32faf8b98477b0a26969ef7d8b9987`.
Source SHA-256: `a04ae893845c3e4af0404293d2e62ab6cc099b03d8264dc76c535674cd9b035c`.
GPU: NVIDIA L40S. Elapsed: 501.7s.

## parallel:final: example continuations

Prompt: The capital of France is

```text
 . . . . . . . . . . . . . . . .
```

Prompt: Once upon a time

```text
 . . . . . . . . . . . . . . . .
```

## diffusion:final: example continuations

Prompt: The capital of France is

```text
 . . . . . . . . . . . . . . . .
```

Prompt: Once upon a time

```text
 . . . . . . . . . . . . . . . .
```

## diffusion:layer12: example continuations

Prompt: The capital of France is

```text
 . . . . . . . . . . . . . . . .
```

Prompt: Once upon a time

```text
 . . . . . . . . . . . . . . . .
```

## diffusion:multi: example continuations

Prompt: The capital of France is

```text
 . . . . . . . . . . . . . . . .
```

Prompt: Once upon a time

```text
 . . . . . . . . . . . . . . . .
```

## diffusion:attention12: example continuations

Prompt: The capital of France is

```text
                
```

Prompt: Once upon a time

```text
                
```

## diffusion:attention_multi: example continuations

Prompt: The capital of France is

```text
                
```

Prompt: Once upon a time

```text
                
```

## diffusion:jspace: example continuations

Prompt: The capital of France is

```text
 . . . . . . . . . . . . . . . .
```

Prompt: Once upon a time

```text
 . . . . . . . . . . . . . . . .
```
