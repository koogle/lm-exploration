"""Versioned Python/math prompt curriculum, preserved from the distillation sweep."""

import random

TOPICS = (
    "factorials and products",
    "binomial coefficients and Pascal's triangle",
    "greatest common divisors and least common multiples",
    "prime numbers and sieves",
    "modular arithmetic",
    "integer digit algorithms",
    "fractions and rational arithmetic",
    "integer square roots",
    "powers and exponentiation",
    "arithmetic progressions",
    "prefix sums and interval queries",
    "difference arrays",
    "sliding windows",
    "two-pointer algorithms",
    "binary search and boundary positions",
    "sorting and stable ties",
    "deduplication with order preservation",
    "frequency dictionaries",
    "set operations",
    "stack algorithms",
    "queue algorithms",
    "balanced parentheses",
    "string parsing",
    "substring matching",
    "run length encoding",
    "palindromes",
    "anagrams",
    "matrix traversal",
    "matrix multiplication",
    "grid neighbors and flood fill",
    "graph breadth-first search",
    "graph depth-first search",
    "topological sorting",
    "shortest paths with nonnegative weights",
    "connected components",
    "tree traversals",
    "memoization",
    "dynamic programming for sequences",
    "knapsack variants",
    "coin change",
    "interval merging",
    "scheduling with greedy algorithms",
    "recursive decomposition",
    "backtracking over small inputs",
    "permutations and combinations",
    "boolean logic and truth tables",
    "bit manipulation",
    "histograms and counting",
    "means medians and exact integer statistics",
    "polynomial evaluation",
    "base conversions",
    "finite state machines",
    "nested list processing",
    "dictionary inversion and grouping",
    "sparse vector operations",
    "distance metrics",
    "calendar arithmetic without date libraries",
    "rational probability on small finite spaces",
    "invariants in loops",
    "monotonic stacks",
    "union find",
    "bounded simulation",
)

CONTEXTS = (
    "a direct mathematical exercise",
    "a library utility",
    "a small scheduling problem",
    "an inventory problem",
    "a board game",
    "text processing",
    "a numeric puzzle",
    "a data cleaning task",
    "an event stream",
    "a resource allocation problem",
    "a classroom worked example",
    "an algorithm interview exercise",
)


def specification(worker, index):
    rng = random.Random(20261022 + worker * 10_000_000 + index)
    topic = TOPICS[rng.randrange(len(TOPICS))]
    context = CONTEXTS[rng.randrange(len(CONTEXTS))]
    level = rng.choice(("introductory", "intermediate", "intermediate", "advanced"))
    boundary = rng.choice(
        (
            "empty or zero inputs",
            "repeated values and ties",
            "inclusive versus exclusive boundaries",
            "negative values where meaningful",
            "single-element inputs",
            "off-by-one cases",
            "impossible or absent results",
        )
    )
    focus = rng.choice(
        (
            "Explain the loop invariant.",
            "Show a small trace of the changing state.",
            "Explain the mathematical definition before the code.",
            "Contrast a plausible wrong solution with the correct one.",
            "Explain why the base case matters.",
            "Show how the data structure represents the problem.",
        )
    )
    numbers = [rng.randint(2, 1000) for _ in range(4)]
    return (
        f"Create a self-contained {level} Python exercise about {topic}, framed as {context}. "
        f"Make {boundary} explicit in the contract. {focus} "
        f"Optional scenario constants: {numbers}. Variation seed: {worker}-{index}. "
        "Invent a concrete problem with a precise input/output contract; do not merely paraphrase the topic. "
        "Provide a complete function named solve, a concise explanation of why it works, and two worked examples. "
        "Use Python's arbitrary-precision integer semantics accurately. Only standard-library imports, no I/O, network or packages. "
        'Return only JSON with string fields "problem", "code", "explanation". Put worked examples in the explanation. '
        "Keep the complete answer under 900 tokens. Do not include the variation seed in the answer."
    )
