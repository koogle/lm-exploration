"""HumanEval and MBPP as chat problems in LiveCodeBench's problem layout.

`load(suite)` returns the test problems of a pinned release with the fields the
LiveCodeBench prompts and grader read (`livecodebench.question`, `grade`). A HumanEval
problem gives its prompt (signature and docstring) as starter code; the prompt also
precedes the answer when graded, so its imports and helper functions are defined, and
the check program is the task's own `check(entry_point)`. An MBPP problem states the
task with its test assertions, as the benchmark's prompts do, and the check program
is those assertions.
"""

SUITES = {
    "humaneval": (
        "openai/openai_humaneval",
        "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
        "openai_humaneval/test-00000-of-00001.parquet",
    ),
    "mbpp": (
        "google-research-datasets/mbpp",
        "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
        "full/test-00000-of-00001.parquet",
    ),
}


def humaneval(row):
    return {
        "question_id": row["task_id"],
        "question_content": "Complete the Python function below.",
        "starter_code": row["prompt"],
        "preamble": row["prompt"] + "\n",
        "check": f"{row['test']}\ncheck({row['entry_point']})\n",
    }


def mbpp(row):
    tests = "\n".join(row["test_list"])
    return {
        "question_id": f"mbpp/{row['task_id']}",
        "question_content": f"{row['text']}\n\nYour code should pass these tests:\n\n{tests}",
        "starter_code": "",
        "check": f"{row['test_setup_code']}\n{tests}\n",
    }


def load(suite):
    """The suite's problems; `difficulty` and `platform` name the suite for the summary."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    repo, revision, file = SUITES[suite]
    path = hf_hub_download(repo, file, repo_type="dataset", revision=revision)
    convert = {"humaneval": humaneval, "mbpp": mbpp}[suite]
    return [
        {**convert(row), "difficulty": suite, "platform": suite}
        for row in pq.read_table(path).to_pylist()
    ]
