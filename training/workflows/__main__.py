"""Resolve and inspect a workflow locally without importing Modal or launching jobs: every
stage resolved, and which stages `--start`/`--end` select."""

import argparse
import json

from .plan import load_workflow, select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--run-id", default="workflow-run")
    parser.add_argument("--start", default="", help="first stage to run")
    parser.add_argument("--end", default="", help="last stage to run")
    args = parser.parse_args()
    plan = load_workflow(args.workflow, args.run_id)
    print(json.dumps({**plan, "selected": select(plan, args.start, args.end)}, indent=2))


if __name__ == "__main__":
    main()
