"""Reproducible sizing heuristics; this does not launch training or fit a law."""

import argparse
import json


def parameter_counts(layers, width, vocabulary):
    """Exact modern GPT count, including biases, norms and tied embeddings."""
    if any(type(x) is not int or x <= 0 for x in (layers, width, vocabulary)):
        raise ValueError("Dimensions must be positive integers")
    hidden = (8 * width + 2) // 3
    attention = 4 * width * width + 4 * width
    mlp = 3 * width * hidden + 2 * hidden + width
    backbone = layers * (attention + mlp + 2 * width) + width
    embedding = vocabulary * width
    return {
        "total": backbone + embedding,
        "backbone": backbone,
        "tied_embedding": embedding,
    }


def report(accepted_tokens=500_000_000):
    if type(accepted_tokens) is not int or accepted_tokens <= 0:
        raise ValueError("Accepted tokens must be a positive integer")
    candidates = []
    for vocabulary in (32768, 151936):
        for layers, heads, width in ((8, 6, 384), (10, 8, 512), (14, 12, 768)):
            counts = parameter_counts(layers, width, vocabulary)
            n = counts["total"]
            candidates.append(
                {
                    "layers": layers,
                    "heads": heads,
                    "width": width,
                    "vocabulary": vocabulary,
                    "parameters": counts,
                    "embedding_fraction": counts["tied_embedding"] / n,
                    "accepted_tokens_per_total_parameter": accepted_tokens / n,
                    "heuristic_20n_tokens_using_total_parameters": 20 * n,
                    "rough_student_training_flops_at_accepted_tokens": 6
                    * n
                    * accepted_tokens,
                }
            )
    return {
        "status": "planning_only_not_a_fitted_scaling_law",
        "reference": "https://arxiv.org/abs/2203.15556",
        "accepted_student_tokens": accepted_tokens,
        "caveats": [
            (
                "20 tokens/parameter is a rough from-scratch compute-optimal heuristic, "
                "not a capacity limit or a continued-training prescription."
            ),
            (
                "Total includes tied embeddings. Vocabulary-heavy architectures need "
                "empirical calibration; these are not paper-equivalent predictions."
            ),
            (
                "6ND omits teacher generation, verification, attention/context overhead, "
                "evaluation and implementation inefficiency; profile actual cost."
            ),
            (
                "Different tokenizers yield different token counts on the same text. "
                "Retokenize the corpus before comparing budgets."
            ),
            (
                "Accepted unique data, repeated exposures and loss-bearing tokens "
                "must be counted separately."
            ),
        ],
        "candidates": candidates,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-tokens", type=int, default=500_000_000)
    args = parser.parse_args()
    print(json.dumps(report(args.accepted_tokens), indent=2))


if __name__ == "__main__":
    main()
