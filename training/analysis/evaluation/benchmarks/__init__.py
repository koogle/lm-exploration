"""One module per benchmark: its prompt format and `score(model, rows, exemplars, settings,
shots) -> (scores, generated answers or None)`. `BENCHMARKS` maps config names to them."""

from . import arc, gsm8k, hellaswag, lambada, openbookqa, svamp, winogrande

BENCHMARKS = {
    "hellaswag": hellaswag,
    "arc_easy": arc,
    "arc_challenge": arc,
    "openbookqa": openbookqa,
    "winogrande": winogrande,
    "lambada": lambada,
    "gsm8k": gsm8k,
    "svamp": svamp,
}
