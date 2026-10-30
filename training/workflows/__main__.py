"""Resolve and inspect a workflow of any stage locally without importing Modal or launching
jobs: a post-training workflow prints its recipe with every setting filled in, a pipeline
its plan for `--run-id`."""

import argparse
import json

from .config import load_workflow, resolve_payload, workflow_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--model-config", default="")
    parser.add_argument("--dataset-config", default="")
    parser.add_argument("--run-id", default="pipeline-run")
    args = parser.parse_args()
    stage = workflow_stage(args.workflow)
    if stage == "pipeline":
        from .pipeline import load_pipeline

        print(json.dumps(load_pipeline(args.workflow, args.run_id), indent=2))
        return
    if stage != "pretrain":
        from .stages import load

        print(json.dumps(load(args.workflow), indent=2))
        return
    payload = load_workflow(
        args.workflow, model_config=args.model_config, dataset_config=args.dataset_config
    )
    _, _, training = resolve_payload(payload)
    print(json.dumps({**payload, "resolved_training": training.as_dict()}, indent=2))


if __name__ == "__main__":
    main()
