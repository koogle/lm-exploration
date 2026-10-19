"""Bounded corpus preparation CLI. Run from training/; no GPU required."""

import argparse
import json

from training_data.simple_llm import Stager, load_recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", default="tokenizer/recipe-v1.json")
    parser.add_argument("--output", required=True, help="Versioned persistent staging directory")
    parser.add_argument("--source", action="append", help="Default: all pinned sources")
    parser.add_argument("--max-records-per-source", type=int, default=100)
    parser.add_argument("--export", help="New snapshot directory; must not exist")
    args = parser.parse_args()
    recipe = load_recipe(args.recipe)
    names = args.source or [s["name"] for s in recipe["sources"]]
    if set(names) - {s["name"] for s in recipe["sources"]}:
        parser.error("Unknown source")
    stager = Stager(args.output, recipe)
    try:
        for name in names:
            count = stager.prepare(name, args.max_records_per_source)
            print(json.dumps({"source": name, "scanned_this_call": count}), flush=True)
        if args.export:
            stager.export(args.export)
    finally:
        stager.close()


if __name__ == "__main__":
    main()
