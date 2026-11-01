"""Exact three-route Shapley allocation of counterfactual log-odds changes.

This allocates effects among the measured cache interventions; it is not an
intrinsic measure of which layers computed a fact or how much information exists.
"""

from itertools import combinations
from math import factorial


def shapley(values):
    """`values` maps frozensets of K/C/A to measured intervention effects."""
    routes = "KCA"
    result = {}
    for route in routes:
        others = [r for r in routes if r != route]
        effect = 0.0
        for n in range(3):
            weight = factorial(n) * factorial(2 - n) / factorial(3)
            for subset in combinations(others, n):
                subset = frozenset(subset)
                effect += weight * (values[subset | {route}] - values[subset])
        result[route] = effect
    assert abs(sum(result.values()) - (values[frozenset(routes)] - values[frozenset()])) < 1e-7
    return result


def allocations(rows):
    cases = {}
    for row in rows:
        if row["mode"] not in ["none", "K", "C", "A", "KC", "KA", "CA", "KCA"]:
            continue
        key = (
            row["task"],
            tuple(row["values"]),
            row["template"],
            row["entity"],
            row["lag"],
            row["direction"],
        )
        cases.setdefault(key, {})[
            frozenset() if row["mode"] == "none" else frozenset(row["mode"])
        ] = row
    result = []
    for (task, values, template, entity, lag, direction), case in cases.items():
        if len(case) != 8:
            continue
        result.append(
            {
                "task": task,
                "values": values,
                "template": template,
                "entity": entity,
                "lag": lag,
                "direction": direction,
                "paired_correct": case[frozenset()]["paired_correct"],
                "clean_gap": case[frozenset()]["clean_gap"],
                "allocation": shapley({k: v["oriented_effect"] for k, v in case.items()}),
            }
        )
    return result
