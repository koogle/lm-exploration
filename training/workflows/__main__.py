"""Resolve and inspect a workflow locally without importing Modal or launching jobs."""

import argparse
import json

from .config import load_workflow, resolve_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--model-config", default="")
    parser.add_argument("--dataset-config", default="")
    args = parser.parse_args()
    payload = load_workflow(
        args.workflow, model_config=args.model_config, dataset_config=args.dataset_config
    )
    policy, _, training = resolve_payload(payload)
    resolved = training if policy.family == "hybrid" else training.as_dict()
    print(json.dumps({**payload, "resolved_training": resolved}, indent=2))


if __name__ == "__main__":
    main()
