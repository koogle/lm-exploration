"""Pure restart policy: never reset a running campaign's clock or worker IDs."""


def resume_state(previous, config, now):
    if not previous:
        raise ValueError("Resume requires an existing campaign")
    for name in ["backend", "batch_size", "workers"]:
        if previous[name] != config[name]:
            raise ValueError(f"Cannot change {name} during resume")
    if previous.get("resume_epoch") == config["resume_epoch"]:
        return previous
    if previous.get("status") == "generating":
        raise ValueError("Another campaign window is active")
    deadline = config["resume_deadline"]
    if not now < deadline <= config["resume_requested_at"] + 8 * 3600:
        raise ValueError("Invalid bounded resume deadline")
    state = dict(previous)
    state["history"] = list(previous.get("history", [])) + [
        {
            k: previous.get(k)
            for k in ["status", "started_at", "stopped_at", "deadline", "archive_call"]
        }
    ]
    state.update(
        status="generating",
        resume_epoch=config["resume_epoch"],
        resumed_at=now,
        deadline=deadline,
        calls={},
        errors=[],
        archive_call=None,
    )
    state["resume_source_sha256"] = config.get("resume_source_sha256", {})
    state.pop("stopped_at", None)
    return state
