"""Chat SFT data: conversations as token ids and loss masks, packed for training.

Subsets are in GLM's chat format unless their recipe names another tokenizer
(`eos_id`, `tokenizer`), such as the Qwen3 agent mixes of
`modal_scripts/modal_agent_mix.py`.

A recipe (data/recipes/chat-sft-*.json) names a Hugging Face dataset at a pinned revision
and its parquet files, each one subset named by its file stem; `subsets` adds
subsets prepared elsewhere (`prepared_in` another recipe's folder, or written into
this recipe's folder by another stage), each read `repeat` times per pass (a
fraction reads a seeded sample of that share). `convert` turns a smoltalk2 row
(Hermes-style tool calls in the text, tool schemas in
`chat_template_kwargs.xml_tools`) into OpenAI messages and tools; `prepare_subset` renders each conversation with `chat.encode`, drops those
longer than the context or sharing a 13-gram with the benchmark screen, holds out
a hash-selected fraction, and writes per subset through `SubsetWriter`, which also
serves sources streamed in rank order:

    <subset>.tokens.npy  uint32, every conversation ending with EOS
    <subset>.mask.npy    uint8, 1 where the token is a training target
    <subset>.docs.npy    int64 [conversations, 2]: start offset and length of each
                         training conversation (the arrays may hold more)
    <subset>.heldout.*   the same three arrays for the held-out conversations
    <subset>.json        counts

`ChatMixture` packs whole conversations, in a seeded shuffled order over every
subset, into windows of context + 1 tokens (the tail padded with EOS and ignored),
and serves (inputs, labels) where a label is the next token if it is a target and
-100 otherwise. A recipe's `replay` names a pretraining recipe and the fraction of
each batch's rows taken from it (`packed_shards.ReplayMixture`). Documents never mix: each conversation ends with EOS, which the
model's document mask respects.
"""

import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from training_data import chat

from tokenizer import glm

IGNORE_INDEX = -100
STOP_IDS = (
    glm.EOS_ID,
    glm.SPECIAL_TOKENS["<|user|>"],
    glm.SPECIAL_TOKENS["<|observation|>"],
)
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}
SAMPLE_CONVERSATIONS = (
    ([{"role": "user", "content": "What is the capital of France? Answer in one sentence."}], None),
    (
        [{"role": "user", "content": "Write a Python function that checks whether n is prime."}],
        None,
    ),
    (
        [
            {
                "role": "user",
                "content": "Pens cost $3 each. How much do 7 pens cost? Show your steps.",
            }
        ],
        None,
    ),
    ([{"role": "user", "content": "What's the weather in Berlin right now?"}], [WEATHER_TOOL]),
)
TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_RESPONSE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
TOOLS = re.compile(r"<tools>\s*([\[{].*?)\s*</tools>", re.DOTALL)


def _loads(text):
    """JSON, or the Python-literal dicts some Hermes rows use."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def _tools(xml_tools):
    """OpenAI tools from `xml_tools` blocks, and any instruction text around them."""
    tools, instructions = [], []
    for block in xml_tools or []:
        matches = TOOLS.findall(block) if isinstance(block, str) else []
        for match in matches or [block]:
            parsed = _loads(match) if isinstance(match, str) else match
            tools.extend(parsed if isinstance(parsed, list) else [parsed])
        if matches:
            rest = TOOLS.sub("", block).strip()
            if rest:
                instructions.append(rest)
    tools = [
        t if t.get("type") == "function" else {"type": "function", "function": t} for t in tools
    ]
    return tools, "\n\n".join(instructions)


def convert(row):
    """OpenAI {"messages", "tools"} from a smoltalk2 row; None when it cannot be parsed."""
    kwargs = row.get("chat_template_kwargs") or {}
    if kwargs.get("python_tools") or kwargs.get("enable_thinking"):
        return None
    try:
        tools, instructions = _tools(kwargs.get("xml_tools"))
        system = "\n\n".join(t for t in (kwargs.get("custom_instructions"), instructions) if t)
        messages = [{"role": "system", "content": system}] if system else []
        for message in row["messages"]:
            role, content = message["role"], message["content"] or ""
            if role == "assistant" and "<tool_call>" in content:
                calls = []
                for body in TOOL_CALL.findall(content):
                    call = _loads(body)
                    calls.append(
                        {
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(
                                    call.get("arguments", {}), ensure_ascii=False
                                ),
                            },
                        }
                    )
                text = TOOL_CALL.sub("", content).strip()
                messages.append({"role": "assistant", "content": text, "tool_calls": calls})
            elif role == "tool":
                responses = TOOL_RESPONSE.findall(content) or [content]
                messages.extend({"role": "tool", "content": r.strip()} for r in responses)
            elif role in ("system", "user", "assistant"):
                if "<think>" in content or "</think>" in content:
                    return None
                messages.append({"role": role, "content": content})
            else:
                return None
    except (ValueError, SyntaxError, KeyError, TypeError, AttributeError):
        return None
    # Turns after the last assistant reply have nothing to train on.
    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    if not messages:
        return None
    return {"messages": messages, "tools": tools}


def sample_prompts():
    """The sample conversations rendered up to the assistant's turn."""
    return tuple(
        chat.render(messages, tools, add_generation_prompt=True)
        for messages, tools in SAMPLE_CONVERSATIONS
    )


def subset_names(recipe):
    """The subsets this recipe prepares from its parquet files."""
    return [Path(file).stem for file in recipe.get("files", [])]


def subsets(recipe):
    """(name, prepared_in or None, repeat) of every subset the recipe trains on."""
    own = [(name, None, 1) for name in subset_names(recipe)]
    return own + [
        (s["name"], s.get("prepared_in"), s.get("repeat", 1)) for s in recipe.get("subsets", [])
    ]


def subset_folder(recipe, root, name):
    """The folder holding a subset: the recipe's own `root`, or a sibling recipe's."""
    prepared_in = next(p for n, p, _ in subsets(recipe) if n == name)
    return Path(root) if prepared_in is None else Path(root).parent / prepared_in


def is_heldout(messages, fraction):
    digest = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).digest()
    return int.from_bytes(digest[:8], "little") / 2**64 < fraction


def recipe_eos(recipe):
    """The token that ends a recipe's conversations and pads its windows: GLM's EOS,
    unless the recipe names its tokenizer's (`eos_id`)."""
    return recipe.get("eos_id", glm.EOS_ID)


def recipe_decoder(recipe):
    """Token ids -> text for a recipe's tokenizer: GLM's, unless the recipe names a
    Hugging Face tokenizer (`tokenizer`: {"repo", "revision"})."""
    if "tokenizer" not in recipe:
        tokenizer = glm.load()
        return lambda ids: tokenizer.decode(ids, skip_special_tokens=False)
    from tokenizers import Tokenizer

    source = recipe["tokenizer"]
    tokenizer = Tokenizer.from_pretrained(source["repo"], revision=source["revision"])
    return lambda ids: tokenizer.decode(ids, skip_special_tokens=False)


class SubsetWriter:
    """One subset written while conversations stream in: tokens and masks go to raw
    files, and `close` writes the layout above with the training conversations (all,
    or the first `count` by rank) and the held-out ones."""

    def __init__(self, folder, name):
        self.folder, self.name = Path(folder), name
        self.folder.mkdir(parents=True, exist_ok=True)
        self.files = {
            # Closed in `close`, after every conversation has been written.
            kind: open(self.folder / f"{name}.{kind}.raw", "wb")  # noqa: SIM115
            for kind in ("tokens", "mask")
        }
        self.position, self.rows = 0, []  # (rank, start, length, targets, held out)

    def write(self, ids, mask, heldout, rank=None):
        """Append one conversation; `rank` orders it for `close(count)` (default: arrival)."""
        np.asarray(ids, np.uint32).tofile(self.files["tokens"])
        np.asarray(mask, np.uint8).tofile(self.files["mask"])
        rank = len(self.rows) if rank is None else rank
        self.rows.append((rank, self.position, len(ids), int(np.sum(mask)), bool(heldout)))
        self.position += len(ids)

    def close(self, count=None):
        """Write the subset's arrays and `<name>.json`; returns the counts."""
        for handle in self.files.values():
            handle.close()
        self.rows.sort()
        trained = [r for r in self.rows if not r[4]]
        trained = trained if count is None else trained[:count]
        heldout = [r for r in self.rows if r[4]]
        arrays = {
            kind: np.memmap(self.folder / f"{self.name}.{kind}.raw", dtype=dtype, mode="r")
            if self.position else np.zeros(0, dtype)
            for kind, dtype in (("tokens", np.uint32), ("mask", np.uint8))
        }  # fmt: skip
        # Training conversations point into the full arrays; held-out ones get their own.
        for kind, array in arrays.items():
            np.save(self.folder / f"{self.name}.{kind}.npy", array)
            pieces = [np.asarray(array[s : s + n]) for _, s, n, _, _ in heldout]
            held = np.concatenate(pieces) if pieces else np.zeros(0, array.dtype)
            np.save(self.folder / f"{self.name}.heldout.{kind}.npy", held)
        docs = np.array([(start, length) for _, start, length, _, _ in trained], np.int64)
        np.save(self.folder / f"{self.name}.docs.npy", docs.reshape(-1, 2))
        lengths = np.array([length for _, _, length, _, _ in heldout], np.int64)
        starts = np.concatenate(([0], np.cumsum(lengths)[:-1])) if len(lengths) else lengths
        held_docs = np.stack((starts, lengths), 1) if len(lengths) else np.zeros((0, 2), np.int64)
        np.save(self.folder / f"{self.name}.heldout.docs.npy", held_docs.astype(np.int64))
        for kind in ("tokens", "mask"):
            (self.folder / f"{self.name}.{kind}.raw").unlink()
        meta = {
            "subset": self.name,
            "conversations": len(trained),
            "tokens": int(sum(r[2] for r in trained)),
            "target_tokens": int(sum(r[3] for r in trained)),
            "heldout_conversations": len(heldout),
            "heldout_tokens": int(sum(r[2] for r in heldout)),
            "heldout_target_tokens": int(sum(r[3] for r in heldout)),
            "stored_conversations": len(self.rows),
        }
        (self.folder / f"{self.name}.json").write_text(json.dumps(meta, indent=2) + "\n")
        return meta


def prepare_subset(recipe, name, rows, folder, tokenizer, screen=None, converter=convert):
    """Convert (`converter`: row -> {"messages", "tools"} or None), encode, screen and
    write one subset; returns its counts."""
    from training_data.packed_shards import contaminated

    writer = SubsetWriter(folder, name)
    dropped = {"unparsed": 0, "too_long": 0, "contaminated": 0}
    for row in rows:
        conversation = converter(row)
        if conversation is None:
            dropped["unparsed"] += 1
            continue
        try:
            ids, mask = chat.encode(tokenizer, conversation["messages"], conversation["tools"])
        except (ValueError, StopIteration, TypeError, KeyError):
            dropped["unparsed"] += 1
            continue
        if len(ids) > recipe["context"] + 1:
            dropped["too_long"] += 1
            continue
        if screen is not None and contaminated(ids, screen, recipe["screen_ngram"]):
            dropped["contaminated"] += 1
            continue
        writer.write(ids, mask, is_heldout(conversation["messages"], recipe["heldout_fraction"]))
    meta = writer.close() | {"dropped": dropped}
    (Path(folder) / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def _load(folder, stem):
    folder = Path(folder)
    return (
        np.load(folder / f"{stem}.tokens.npy", mmap_mode="r"),
        np.load(folder / f"{stem}.mask.npy", mmap_mode="r"),
        np.load(folder / f"{stem}.docs.npy"),
    )


def pack(conversations, context, eos=glm.EOS_ID):
    """Pack (tokens, mask) conversations into windows padded with `eos`; yields
    (inputs, labels) arrays."""
    window, targets = [], []
    for tokens, mask in conversations:
        if len(window) + len(tokens) > context + 1 and window:
            yield _finish(window, targets, context, eos)
            window, targets = [], []
        window.extend(tokens.tolist())
        targets.extend(mask.tolist())
    if window:
        yield _finish(window, targets, context, eos)


def _finish(window, targets, context, eos):
    pad = context + 1 - len(window)
    tokens = np.array(window + [eos] * pad, np.int64)
    trained = np.array(targets + [0] * pad, bool)
    labels = np.where(trained[1:], tokens[1:], IGNORE_INDEX)
    return tokens[:-1], labels


class ChatMixture:
    """Packed training windows over every subset of a prepared chat recipe, one pass.

    The order is a seeded permutation of all conversations (a subset with `repeat`
    appears that many times, its fractional part as a seeded sample); the state is
    the number of conversations consumed, so a resumed run continues exactly.
    """

    def __init__(self, recipe, root, seed=0):
        self.recipe, self.root, self.context = recipe, Path(root), recipe["context"]
        self.eos = recipe_eos(recipe)
        listed = subsets(recipe)
        self.subsets = [_load(subset_folder(recipe, root, name), name) for name, _, _ in listed]
        rng = np.random.default_rng(seed)
        index = []
        for s, ((_, _, docs), (_, _, repeat)) in enumerate(zip(self.subsets, listed, strict=True)):
            whole, part = int(repeat), repeat - int(repeat)
            chosen = list(range(len(docs))) * whole
            chosen += sorted(rng.choice(len(docs), round(part * len(docs)), replace=False))
            index.extend((s, int(i)) for i in chosen)
        order = rng.permutation(len(index))
        self.order = [index[i] for i in order]
        self.position = 0
        self.wait_seconds = 0

    def _conversations(self):
        while self.position < len(self.order):
            subset, i = self.order[self.position]
            tokens, mask, docs = self.subsets[subset]
            start, length = docs[i]
            self.position += 1
            yield (
                np.asarray(tokens[start : start + length]),
                np.asarray(mask[start : start + length]),
            )

    def next_batch(self, rows):
        inputs, labels, names = [], [], []
        stream = self._conversations()
        window, targets = [], []
        while len(inputs) < rows:
            conversation = next(stream, None)
            if conversation is None:
                if not window:
                    raise StopIteration
                tokens, mask = None, None
            else:
                tokens, mask = conversation
            if tokens is None or (len(window) + len(tokens) > self.context + 1 and window):
                x, y = _finish(window, targets, self.context, self.eos)
                inputs.append(x)
                labels.append(y)
                names.append("chat")
                window, targets = [], []
            if tokens is not None:
                window.extend(tokens.tolist())
                targets.extend(mask.tolist())
        # A conversation read past the last window is returned to the stream.
        if window:
            self.position -= 1
        return np.stack(inputs), np.stack(labels), names

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]


def pass_windows(mixture):
    """The windows one pass of a `ChatMixture` packs (from its position), counted from
    the conversation lengths alone, as `next_batch` packs them."""
    windows, used = 0, 0
    for subset, index in mixture.order[mixture.position :]:
        length = int(mixture.subsets[subset][2][index][1])
        if used and used + length > mixture.context + 1:
            windows, used = windows + 1, 0
        used += length
    return windows + bool(used)


def chat_heldout_windows(recipe, root, name, context, max_conversations):
    """One subset's held-out conversations packed whole into windows (see
    `train.heldout_windows`); None when the subset has no held-out conversations.

    The byte count is the UTF-8 length of the trained target tokens' text.
    """
    tokens, mask, docs = _load(subset_folder(recipe, root, name), f"{name}.heldout")
    conversations = [
        (np.asarray(tokens[s : s + n]), np.asarray(mask[s : s + n]))
        for s, n in docs[:max_conversations]
    ]
    if not conversations:
        return None
    windows = list(pack(conversations, context, recipe_eos(recipe)))
    inputs = np.stack([x for x, _ in windows])
    labels = np.stack([y for _, y in windows])
    targets = labels[labels != IGNORE_INDEX]
    total_bytes = len(recipe_decoder(recipe)(targets.tolist()).encode())
    return inputs, labels, total_bytes
