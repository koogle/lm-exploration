"""A hybrid export as the agent of PR-task episodes (`training_data.pr_task_agent`).

The conversation renders in the GLM chat format (`training_data.chat`) in one of two
agent formats: "tools" (the harness's five tools) or "open-swe" (Open-SWE-Traces'
single bash tool and its PR-description prompt, `training_data.open_swe_format`, as
the agent-mix fine-tunes are trained). The export decodes with
`generate.generate_batch` in batches, greedily or sampled, and each reply is parsed
back into an OpenAI assistant message (in "open-swe", the submission command becomes
the harness's `submit`).
"""

from training_data import pr_task_agent

AGENT_FORMATS = ("tools", "open-swe")


class LocalPolicy:
    """A hybrid export as the agent: GLM chat format in `agent_format`, batches of
    `rows`, each turn up to `max_new_tokens`; greedy at temperature 0, otherwise
    sampled with `top_k` and nucleus `top_p` from `seed`."""

    def __init__(
        self,
        lm,
        tokenizer,
        max_new_tokens=1536,
        rows=32,
        agent_format="tools",
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        seed=0,
    ):
        import torch

        if agent_format not in AGENT_FORMATS:
            raise ValueError(f"Unknown agent format {agent_format}")
        self.lm, self.tokenizer = lm, tokenizer
        self.max_new_tokens, self.rows = max_new_tokens, rows
        self.agent_format, self.temperature = agent_format, temperature
        self.top_p, self.top_k = top_p, top_k
        device = next(lm.parameters()).device
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def render(self, messages):
        from training_data import chat, open_swe_format

        if self.agent_format == "open-swe":
            return chat.render(
                open_swe_format.conversation(messages),
                open_swe_format.TOOLS,
                add_generation_prompt=True,
            )
        return chat.render(
            pr_task_agent.training_row(messages)["messages"],
            pr_task_agent.TOOLS,
            add_generation_prompt=True,
        )

    def parse(self, text):
        from training_data import chat, open_swe_format

        message = chat.parse_assistant(text)
        if self.agent_format == "open-swe":
            return open_swe_format.mark_submission(message)
        return message

    def __call__(self, conversations):
        from training_data import chat_sft

        from . import generate

        room = self.lm.config.block_size - self.max_new_tokens
        prompts = []
        for messages in conversations:
            ids = self.tokenizer.encode(self.render(messages), add_special_tokens=False).ids
            prompts.append(ids if len(ids) <= room else None)
        replies = [None] * len(conversations)
        order = sorted((i for i, p in enumerate(prompts) if p), key=lambda i: -len(prompts[i]))
        for start in range(0, len(order), self.rows):
            group = order[start : start + self.rows]
            outputs = generate.generate_batch(
                self.lm,
                [prompts[i] for i in group],
                self.max_new_tokens,
                chat_sft.STOP_IDS,
                temperature=self.temperature,
                top_p=self.top_p,
                generator=self.generator,
                top_k=self.top_k,
            )
            for index, ids in zip(group, outputs, strict=True):
                text = self.tokenizer.decode(ids, skip_special_tokens=False)
                replies[index] = self.parse(text)
        return replies
