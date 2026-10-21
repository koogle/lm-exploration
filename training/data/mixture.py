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
