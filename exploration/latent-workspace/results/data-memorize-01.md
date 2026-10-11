# data-memorize-01: initial data-budget study

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
| 96 | 100 | 3962 | 3.757 | 11.090 | 10.265 | 11.174 | 11.830 |
| 96 | 500 | 19850 | 2.419 | 10.562 | 9.981 | 10.433 | 11.272 |
| 96 | 1000 | 39699 | 2.119 | 10.496 | 9.963 | 10.582 | 10.944 |
| 96 | 2000 | 79780 | 1.708 | 10.159 | 9.785 | 10.023 | 10.669 |

## Validation continuations: 96 windows

**prose**

```text
 earliest The The of The twenty . . to twenty to twenty in " in response War " " " " The song The
```

**code**

```text
        #                      =_results       .routing World = =alité = = = =
 = = = = = =
 =
```

**translation**

```text
 eine wollen ist eine die die sollten, daß die die, daß, die die die die, sollten die, die daß
```

