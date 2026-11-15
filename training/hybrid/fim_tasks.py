"""HumanEval-Infilling in each model's own fill-in-the-middle format.

lm-evaluation-harness ships HumanEval-Infilling (`humaneval_infilling`) with its data,
test programs and execution-based pass@1, but prompts every model with the suffix
written before the prefix as plain text. These task configs keep the harness's data,
grading and stop rules and replace only the prompt with the model's FIM tokens:
`prefix_token prompt suffix_token suffix middle_token` (PSM), the format the model was
trained on. The completion is the middle; the harness runs prompt + middle + suffix
against the task's tests.
"""

# (prefix, suffix, middle) tokens of each family's PSM format.
FIM_FORMATS = {
    "glm": ("<|code_prefix|>", "<|code_suffix|>", "<|code_middle|>"),
    "starcoder": ("<fim_prefix>", "<fim_suffix>", "<fim_middle>"),
    "santacoder": ("<fim-prefix>", "<fim-suffix>", "<fim-middle>"),
    "deepseek": ("<｜fim▁begin｜>", "<｜fim▁hole｜>", "<｜fim▁end｜>"),
    "qwen": ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"),
}

# Harness dataset configs: (task suffix, dataset name, generation stop strings).
SPLITS = (
    ("single_line", "HumanEval-SingleLineInfilling", ["\n"]),
    ("multi_line", "HumanEval-MultiLineInfilling", []),
)


def task_configs(fim_format):
    """Harness task configs (dicts for `lm_eval.simple_evaluate`) for one FIM format."""
    from lm_eval.tasks.humaneval_infilling import utils

    prefix, suffix, middle = FIM_FORMATS[fim_format]
    configs = []
    for name, dataset, until in SPLITS:
        configs.append(
            {
                "task": f"humaneval_fim_{name}_{fim_format}",
                "dataset_path": "loubnabnl/humaneval_infilling",
                "dataset_name": dataset,
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
                        "filter": [{"function": "custom", "filter_fn": utils.build_predictions}],
                    }
                ],
                "metadata": {"version": 1.0, "fim_format": fim_format},
            }
        )
    return configs
