"""Rolling log-likelihood of a text file, as bits per byte, for any harness model.

`text_config(path)` is a harness task over a JSONL file of `{"id", "text"}` rows (written
by `evaluation/fresh_text.py`): each row's whole text is scored with the harness's rolling
windows (`loglikelihood_rolling`), and the summed log-likelihood is divided by the text's
whitespace-separated words and UTF-8 bytes, the metrics of the harness's own `wikitext` task.
Every model scores the same bytes whatever its tokenizer, so `bits_per_byte` compares
models directly.
"""


def _text(doc):
    return doc["text"]


def _results(doc, results):
    (loglikelihood,) = results
    words, size = len(doc["text"].split()), len(doc["text"].encode())
    return {
        "word_perplexity": (loglikelihood, words),
        "byte_perplexity": (loglikelihood, size),
        "bits_per_byte": (loglikelihood, size),
    }


def text_config(path):
    """Harness task config (a dict for `lm_eval.simple_evaluate`) for the rows of `path`."""
    return {
        "task": "text_" + path.rsplit("/", 1)[-1].removesuffix(".jsonl").replace(".", "_"),
        "dataset_path": "json",
        "dataset_kwargs": {"data_files": {"test": path}},
        "output_type": "loglikelihood_rolling",
        "test_split": "test",
        "doc_to_text": "",
        "doc_to_target": _text,
        "process_results": _results,
        "metric_list": [
            {"metric": "word_perplexity"},
            {"metric": "byte_perplexity"},
            {"metric": "bits_per_byte"},
        ],
        "metadata": {"version": 1.0},
    }
