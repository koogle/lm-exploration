# data-position-memorize-01: initial data-budget study

> **Corpus audit correction:** the parent training pool contained 75 duplicate windows
> and one exact train/validation window overlap. These historical validation metrics
> are exploratory. The follow-up cleans training while keeping held-out tensors fixed.
> See [audit](data-study-01-window-audit.json). Tiny-set training-fit diagnostics remain
> training measurements, not evidence of held-out quality.


Completed: True. sealed; no test losses or accuracy computed.

All rows use the same backbone, architecture and seed. NLL is all-masked
conditional token loss, not iterative sequence likelihood. Single-seed
screening with 64 validation windows/domain; no significance claim.


| Unique blocks | Updates | Supervised masked tokens | Train NLL | Validation NLL | Prose | Code | Translation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 100 | 3962 | 0.419 | 15.902 | 14.911 | 15.887 | 16.907 |
| 96 | 500 | 19850 | 0.046 | 19.964 | 19.801 | 19.625 | 20.464 |
| 96 | 1000 | 39699 | 0.198 | 21.911 | 21.348 | 21.306 | 23.079 |
| 96 | 2000 | 79780 | 0.035 | 22.844 | 22.811 | 22.420 | 23.300 |

## Validation continuations: 96 windows

**prose**

```text
 tropical in , plugin coin of that denomination coin the that plugin coin the that denomination coin of that denomination coin of that denomination
```

**code**

```text
            prefix=,           	status = =
				status = =
 Youth = = =
 Youth = = =
 Youth = = =

```

**translation**

```text
 with various ,enary return of thatug with auch theenary coin of that denomination coin0 World:';
 coin of that denomination
```

