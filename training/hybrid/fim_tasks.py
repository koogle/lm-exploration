"""HumanEval-Infilling in each model's own fill-in-the-middle format, and HumanEval.

lm-evaluation-harness ships HumanEval-Infilling (`humaneval_infilling`) with its data,
test programs and execution-based pass@1, but prompts every model with the suffix
written before the prefix as plain text. These task configs keep the harness's data,
grading and stop rules and replace only the prompt with the model's FIM tokens:
`prefix_token prompt suffix_token suffix middle_token` (PSM), the format the model was
trained on. The completion is the middle; the harness runs prompt + middle + suffix
against the task's tests.

Single-line generation stops at the first newline, which the harness drops from the
completion; every single-line middle ends with one, so it is put back before the
program is assembled. Without it the middle runs into the suffix's first line.

`humaneval_config` is the harness's HumanEval with trailing whitespace stripped from
each prompt, as bigcode-evaluation-harness does: a prompt ending in a bare newline
splits the newline-plus-indentation token most code tokenizers learn, and SantaCoder,
for one, answers it with end-of-text.
"""

# (prefix, suffix, middle) tokens of each family's PSM format.
FIM_FORMATS = {
    "glm": ("<|code_prefix|>", "<|code_suffix|>", "<|code_middle|>"),
    "starcoder": ("<fim_prefix>", "<fim_suffix>", "<fim_middle>"),
    "santacoder": ("<fim-prefix>", "<fim-suffix>", "<fim-middle>"),
    "deepseek": ("<｜fim▁begin｜>", "<｜fim▁hole｜>", "<｜fim▁end｜>"),
    "qwen": ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"),
}

# The harness's dataset repository is a loading script, which `datasets` 4 no longer
# runs; its Hub parquet conversion holds the same splits. Pinned to that conversion's commit.
PARQUET = (
    "hf://datasets/loubnabnl/humaneval_infilling@448c2c3593ea013776cc5f7f7c0b34342b7b524e"
    "/{split}/test/0000.parquet"
)

# Harness dataset configs: (task suffix, dataset name, generation stop strings).
SPLITS = (
    ("single_line", "HumanEval-SingleLineInfilling", ["\n"]),
    ("multi_line", "HumanEval-MultiLineInfilling", []),
)


def _single_line_predictions(resps, docs):
    """prompt + middle + suffix, with the newline the single-line stop removed."""
    return [
        [doc["prompt"] + r + "\n" + doc["suffix"] for r in resp]
        for resp, doc in zip(resps, docs, strict=True)
    ]


def _strip_prompts(dataset):
    return dataset.map(lambda doc: {"prompt": doc["prompt"].rstrip()})


def humaneval_config():
    """The harness's HumanEval task with right-stripped prompts."""
    from lm_eval.tasks.humaneval import utils

    return {
        "task": "humaneval_stripped",
        "dataset_path": "openai/openai_humaneval",
        "unsafe_code": True,
        "output_type": "generate_until",
        "test_split": "test",
        "process_docs": _strip_prompts,
        "doc_to_text": "{{prompt}}",
        "doc_to_target": "{{test}}\ncheck({{entry_point}})",
        "metric_list": [
            {"metric": utils.pass_at_k, "aggregation": "mean", "higher_is_better": True, "k": [1]}
        ],
        "generation_kwargs": {
            "until": ["\nclass", "\ndef", "\n#", "\nif", "\nprint"],
            "max_gen_toks": 1024,
            "do_sample": False,
        },
        "repeats": 1,
        "num_fewshot": 0,
        "filter_list": [
            {
                "name": "create_test",
                "filter": [{"function": "custom", "filter_fn": utils.build_predictions}],
            }
        ],
        "metadata": {"version": 1.0},
    }


def task_configs(fim_format):
    """Harness task configs (dicts for `lm_eval.simple_evaluate`) for one FIM format."""
    from lm_eval.tasks.humaneval_infilling import utils

    prefix, suffix, middle = FIM_FORMATS[fim_format]
    configs = []
    for name, dataset, until in SPLITS:
        configs.append(
            {
                "task": f"humaneval_fim_{name}_{fim_format}",
                "dataset_path": "parquet",
                "dataset_kwargs": {"data_files": {"test": PARQUET.format(split=dataset)}},
                "unsafe_code": True,
                "output_type": "generate_until",
                "test_split": "test",
                "doc_to_text": f"{prefix}{{{{prompt}}}}{suffix}{{{{suffix}}}}{middle}",
                "doc_to_target": "{{test}}\ncheck({{entry_point}})",
                "metric_list": [
                    {
                        "metric": utils.pass_at_k,
                        "aggregation": "mean",
                        "higher_is_better": True,
                        "k": [1],
                    }
                ],
                "generation_kwargs": {"until": until, "max_gen_toks": 1024, "do_sample": False},
                "repeats": 1,
                "num_fewshot": 0,
                "filter_list": [
                    {
                        "name": "create_test",
                        "filter": [
                            {
                                "function": "custom",
                                "filter_fn": _single_line_predictions
                                if name == "single_line"
                                else utils.build_predictions,
                            }
                        ],
                    }
                ],
                "metadata": {"version": 1.0, "fim_format": fim_format},
            }
        )
    return configs
