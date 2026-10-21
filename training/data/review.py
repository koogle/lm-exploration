"""One readiness gate shared by loaders and preparation preflight."""


def require_data_review(training_ready: bool, allow_unreviewed: bool = False) -> None:
    if not training_ready and not allow_unreviewed:
        raise ValueError("Preparation snapshot has pending audits; allow_unreviewed_data required")
