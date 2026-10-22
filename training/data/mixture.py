"""Explicit loader selection; each implementation validates its own format."""


def open_mixture(manifest, *, loader, context, split="train", allow_unreviewed=False):
    if loader == "token_shards":
        from .token_stream import TokenMixture

        mixture = TokenMixture
    elif loader == "raw":
        from .raw_token_stream import RawTokenMixture

        mixture = RawTokenMixture
    elif loader == "live":
        from .live_stream import LiveTokenMixture

        mixture = LiveTokenMixture
    else:
        raise ValueError(f"Unsupported data loader: {loader!r}")
    return mixture(manifest, context=context, split=split, allow_unreviewed=allow_unreviewed)


def check_capacity(mixture, target_tokens):
    """Reject a token target the snapshot cannot supply within each source's pass cap.

    Exact per-source `target_tokens` budgets must cover every source and sum to the
    run's target; otherwise the weighted mixture must fit within the pass caps.
    """
    default = mixture.manifest["max_passes"]
    sources = mixture.sources
    budgets = {name: s["target_tokens"] for name, s in sources.items() if "target_tokens" in s}
    if budgets:
        if set(budgets) != set(sources) or sum(budgets.values()) != target_tokens:
            raise ValueError("Source targets must cover every source and sum to the token target")
        for name, budget in budgets.items():
            capacity = sources[name].get("max_passes", default)
            if budget > capacity * sources[name]["splits"]["train"]["loss_tokens"]:
                raise ValueError(f"Source {name} cannot supply its target within its pass cap")
        return
    capacity = min(
        s.get("max_passes", default)
        * s["splits"]["train"]["loss_tokens"]
        * 100
        / s["weight_percent"]
        for s in sources.values()
    )
    if target_tokens > capacity:
        raise ValueError("Snapshot cannot support target mixture within source pass caps")
