# data-residual-01: initial data-budget study

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
| 12288 | 100 | 3962 | 7.381 | 7.788 | 7.544 | 7.742 | 8.080 |
| 12288 | 500 | 19850 | 6.076 | 6.379 | 6.489 | 6.150 | 6.496 |
| 12288 | 2000 | 79780 | 5.771 | 6.149 | 6.244 | 6.020 | 6.183 |
| 12288 | 4000 | 160059 | 5.467 | 5.963 | 6.097 | 5.656 | 6.136 |

## Validation continuations: 12288 windows

**prose**

```text
 the the the the female female . . The the the the female the the the female the the the female the the the
```

**code**

```text
        self self self




def self self self app(self       app ' ' ' ' ' ' '
```

**translation**

```text
 Die Pr einen einen Reenenen von der der der Demenenen von der der, die dem der der
```

