"""One module per benchmark: its prompt format and `score(model, rows, exemplars, settings,
shots) -> (scores, generated answers or None)`. `BENCHMARKS` maps config names to them.
A module with `SANDBOX = True` runs model-written code and takes the code sandbox as a
sixth argument."""

from . import (
    arc,
    gsm8k,
    hellaswag,
    humaneval_plus,
    lambada,
    mbpp_plus,
    openbookqa,
    svamp,
    winogrande,
)

BENCHMARKS = {
    "hellaswag": hellaswag,
    "arc_easy": arc,
    "arc_challenge": arc,
    "openbookqa": openbookqa,
    "winogrande": winogrande,
    "lambada": lambada,
    "gsm8k": gsm8k,
    "svamp": svamp,
    "mbpp_plus": mbpp_plus,
    "humaneval_plus": humaneval_plus,
}
