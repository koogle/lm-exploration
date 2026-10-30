"""Resolve and inspect a workflow of any stage locally without importing Modal or launching
jobs; a post-training workflow prints its recipe with every setting filled in."""

import argparse
import json

from .config import load_workflow, resolve_payload, workflow_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--model-config", default="")
    parser.add_argument("--dataset-config", default="")
    args = parser.parse_args()
    stage = workflow_stage(args.workflow)
    if stage == "sft":
        from trainer.sft_recipe import load_recipe

        print(json.dumps(load_recipe(args.workflow), indent=2))
        return
    if stage == "rl":
        from trainer.rl.recipe import load_recipe

        print(json.dumps(load_recipe(args.workflow), indent=2))
        return
    payload = load_workflow(
        args.workflow, model_config=args.model_config, dataset_config=args.dataset_config
    )
    _, _, training = resolve_payload(payload)
    print(json.dumps({**payload, "resolved_training": training.as_dict()}, indent=2))


if __name__ == "__main__":
    main()
