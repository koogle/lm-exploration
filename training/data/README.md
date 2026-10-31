# Training datasets

[EU institutional corpus exploration](eu/README.md) is a separate discovery and
extraction project with ten source candidates, bounded crawlers, agent research
notes and Parquet export. Its artifacts are not registered training inputs.

For the new frozen-tokenizer corpus, use the separate
[Simple LLM streaming pipeline](SIMPLE_LLM_STREAMING.md) and
[raw-corpus preparation guide](RAW_CORPUS.md). Its local preparation
CLI lives in [`prepare_simple_llm.py`](prepare_simple_llm.py); run
`uv run lm-prepare-simple-llm --help` (or
`uv run python -m training_data.prepare_simple_llm --help`) from `training/`.
The corresponding CPU Modal launcher lives in
[`../modal_scripts/modal_prepare_simple_llm.py`](../modal_scripts/modal_prepare_simple_llm.py).

The hybrid family's GLM data uses [`packed_shards.py`](packed_shards.py): uint32
shards with real EOS and per-document byte counts, a 13-gram benchmark screen, and
`PreparedMixture`, which packs documents into fixed windows and can read shards still
being prepared. It shares the source-scheduling rule (`mixture.deficit_source`) with
`TokenMixture`; recipes and the Modal launcher are in [`../hybrid/`](../hybrid/README.md).

## Chat format and chat SFT data

The GLM tokenizer is pinned in [`../tokenizer/glm.py`](../tokenizer/glm.py); models that
use it (the hybrid family, [`../hybrid/`](../hybrid/README.md)) read chat data through
the modules below. The prepared subsets are GLM token ids, so a family with another
tokenizer would prepare its own.

[`chat.py`](chat.py) maps OpenAI chat messages (`system`/`user`/`assistant`/`tool`,
assistant `tool_calls`, a `tools` list of function schemas) onto GLM-4.7-Flash's
own chat template, pinned by revision and SHA-256 like the tokenizer. Assistant
turns render without reasoning (`<|assistant|></think>`); calls are GLM's
`<tool_call>name<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>` and
results `<|observation|><tool_response>…</tool_response>`. `encode` returns token
ids and a loss mask that covers each assistant turn after its prompt plus the token
that ends it (the next role token, or the closing EOS); `parse_assistant` turns
generated text back into an OpenAI assistant message.

[`recipes/chat-sft-v1.json`](recipes/chat-sft-v1.json) takes the non-reasoning
(`no_think`) files of `HuggingFaceTB/smoltalk2` SFT, pinned by revision: general
chat and system prompts, rewriting and summarization, instruction following,
tables, science, OpenThoughts3 answers and two tool-calling sets (xLAM traces,
Hermes function calling). [`chat_sft.py`](chat_sft.py) converts each row to OpenAI
messages (Hermes `<tool_call>` JSON or Python dicts become `tool_calls`, multiple
`<tool_response>` blocks become separate `tool` turns, instructions around a
`<tools>` block join the system prompt, turns after the last assistant reply are
cut), drops rows with reasoning or unparsable calls, conversations longer than
the context and those sharing a 13-gram with the pretraining benchmark screen,
and holds out 0.5% by hash. [`modal_chat_sft_data.py`](../modal_scripts/modal_chat_sft_data.py) prepares
each file in its own call (`retries=0`, refuses work 24 hours after launch):

```sh
uv run --all-extras modal run -m modal_scripts.modal_chat_sft_data
```


`datasets.json` records each dataset's source, license review and eligibility;
[`registry.py`](registry.py) validates it with pydantic.
Paths in it are relative to this `data/` directory. Active training consumes the
prepared snapshots described above.


## Registered datasets

Both registered datasets are character-level Shakespeare learning corpora for
exercising training, checkpointing and sampling. Their text is not committed to
Git. To inspect them locally, run from
`training/`:

```sh
uv run python -m training_data.fetching.fetch_tinyshakespeare
uv run python -m training_data.fetching.fetch_shakespeare_complete
```

- `tiny-shakespeare` (~1.1 MB): the char-rnn benchmark, saved to
  `data/raw/tinyshakespeare.txt`.
- `shakespeare-complete` (~5.4 MB): Project Gutenberg eBook #100, saved to
  `data/raw/shakespeare-complete.txt` with the Gutenberg header and license footer
  removed and line endings normalized. Gutenberg revises this eBook in place, so
  a new revision fails the pinned check until the hashes are deliberately updated.

Each fetcher verifies a pinned SHA-256. Source and license notes are in
`datasets.json`. These are not selected research corpora or evidence of useful
language capability. Research corpus choices and language mixtures remain undecided.

## Add an eligible local text dataset

1. Establish the source's eligibility and any attribution, retention, redistribution
   and removal obligations before enabling training. Public availability alone is
   not eligibility. Keep record-level source/revision/license/filtering provenance
   separately when aggregating documents, and inspect for credentials and personal
   information. Do not commit private or unreviewed corpus content.
2. Write a fetch module in `fetching/` whose `main()` downloads one nonempty UTF-8
   text file to its `DESTINATION` under `raw/` and verifies a pinned SHA-256 (see
   the existing fetchers).
   `raw/` and `cache/` are ignored by Git and excluded from the Modal image. The
   historical character loader reads the whole file into RAM; this is not the
   active snapshot preparation pipeline.
3. Add an object to the `datasets` array in `datasets.json` with all required fields:

   | Field | Meaning |
   | --- | --- |
   | `id` | Unique lowercase slug: letters, digits, `_` and `-`, at most 64 characters |
   | `name` | Short display name |
   | `path` | Relative path under `data/`, no absolute paths, `..`, or escaping symlinks |
   | `description` | Content, intended use and limitations |
   | `source` | Authorship or source URL/repository and pinned revision; reference retained record-level provenance when applicable |
   | `license_notes` | Actual terms and eligibility review, obligations and limits; do not invent approvals |
   | `languages` | Nonempty array of language identifiers, e.g. `["en"]` |
   | `training_eligible` | Boolean; `true` only after the above review for the intended use |

   Do not add an undecided candidate source as an eligible entry.
4. Validate the registry entry locally. The retained historical recipe schema
   only recognizes the original Shakespeare IDs; there is no character-training
   launcher. For current training inputs, use the snapshot preparation guides above.

Dataset definitions and fetching scripts are tracked; `raw/`, `cache/` and
`../runs/` are not. Each research dataset needs its own evidence; this README is not a license
allowlist or an approval of the candidate sources in the training plan.
