"""A hybrid export as the agent of PR-task episodes (`training_data.pr_task_agent`).

The conversation and the tools render in the GLM chat format (`training_data.chat`), the
export decodes greedily with `generate.generate_batch` in batches, and each reply is
parsed back into an OpenAI assistant message.
"""

from training_data import pr_task_agent


class LocalPolicy:
    """A hybrid export as the agent: GLM chat format with the tools, greedy decoding
    in batches of `rows`, each turn up to `max_new_tokens`."""

    def __init__(self, lm, tokenizer, max_new_tokens=1536, rows=32):
        self.lm, self.tokenizer = lm, tokenizer
        self.max_new_tokens, self.rows = max_new_tokens, rows

    def __call__(self, conversations):
        from training_data import chat, chat_sft

        from . import generate

        room = self.lm.config.block_size - self.max_new_tokens
        prompts = []
        for messages in conversations:
            text = chat.render(
                pr_task_agent.training_row(messages)["messages"],
                pr_task_agent.TOOLS,
                add_generation_prompt=True,
            )
            ids = self.tokenizer.encode(text, add_special_tokens=False).ids
            prompts.append(ids if len(ids) <= room else None)
        replies = [None] * len(conversations)
        order = sorted((i for i, p in enumerate(prompts) if p), key=lambda i: -len(prompts[i]))
        for start in range(0, len(order), self.rows):
            group = order[start : start + self.rows]
            outputs = generate.generate_batch(
                self.lm, [prompts[i] for i in group], self.max_new_tokens, chat_sft.STOP_IDS
            )
            for index, ids in zip(group, outputs, strict=True):
                text = self.tokenizer.decode(ids, skip_special_tokens=False)
                replies[index] = chat.parse_assistant(text)
        return replies
