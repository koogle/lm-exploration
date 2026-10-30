"""HybridLM as an lm-evaluation-harness model.

Scoring (`loglikelihood`, `loglikelihood_rolling`) runs the model's own
document-masked forward pass on CUDA (FlexAttention, fla KDA). Each request is
`[EOS] + context + continuation`, the EOS standing for the end of a previous
document, left-truncated to the context length. Requests are grouped into a few
fixed shapes (rows x length, right-padded with EOS) so the compiled attention
reuses its graphs. Generation (`generate_until`) uses `generate.generate_batch`:
greedy, batched, left-padded with EOS, cut at the first stop string.
"""

import torch
from lm_eval.api.model import LM
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window
from torch.nn import functional as F

from . import generate
from . import tokenizer as glm

LENGTHS = (256, 512, 1024, 2048, 4096)
TOKENS_PER_BATCH = 32768


class HybridHarnessModel(LM):
    def __init__(self, lm, tokenizer, generation_rows=64):
        super().__init__()
        self.lm, self.tokenizer = lm, tokenizer
        self.eos = lm.config.eos_id
        self.context = lm.config.block_size
        self.generation_rows = generation_rows
        self._device = lm.token_embedding.weight.device

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    # Scoring.
    def _pair(self, context, continuation):
        # A context's trailing spaces belong to the continuation's first token.
        spaces = len(context) - len(context.rstrip())
        if spaces:
            context, continuation = context[:-spaces], context[-spaces:] + continuation
        whole = self.encode(context + continuation)
        prefix = self.encode(context)
        if whole[: len(prefix)] != prefix:
            return prefix, self.encode(continuation)
        return prefix, whole[len(prefix) :]

    @torch.no_grad()
    def _score(self, pairs):
        """(log-probability, is greedy) of each continuation token list after its context."""
        items = []
        for index, (context, continuation) in enumerate(pairs):
            tokens = ([self.eos] + context + continuation)[-(self.context + 1) :]
            items.append((index, tokens, len(continuation)))
        items.sort(key=lambda item: -len(item[1]))
        results = [None] * len(pairs)
        position = 0
        while position < len(items):
            length = next(n for n in LENGTHS if n >= len(items[position][1]) - 1)
            rows = TOKENS_PER_BATCH // length
            batch = items[position : position + rows]
            position += len(batch)
            inputs = torch.full((rows, length), self.eos, dtype=torch.long)
            for row, (_, tokens, _) in enumerate(batch):
                inputs[row, : len(tokens) - 1] = torch.tensor(tokens[:-1])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = self.lm.hidden_states(inputs.to(self.device))
            for row, (index, tokens, count) in enumerate(batch):
                end = len(tokens) - 1
                logits = F.linear(hidden[row, end - count : end], self.lm.output_weight).float()
                targets = torch.tensor(tokens[-count:], device=self.device)
                logprobs = logits.log_softmax(-1)
                total = logprobs.gather(-1, targets[:, None]).sum().item()
                greedy = bool((logits.argmax(-1) == targets).all())
                results[index] = (total, greedy)
        return results

    def loglikelihood(self, requests):
        return self._score([self._pair(*request.args) for request in requests])

    def loglikelihood_rolling(self, requests):
        pairs, owners = [], []
        for index, request in enumerate(requests):
            windows = get_rolling_token_windows(
                token_list=self.encode(request.args[0]),
                prefix_token=self.eos,
                max_seq_len=self.context,
                context_len=1,
            )
            for window in map(make_disjoint_window, windows):
                # The window's context begins with the prefix EOS, which _score adds.
                context = window[0][1:] if window[0][:1] == [self.eos] else window[0]
                pairs.append((context, window[1]))
                owners.append(index)
        totals = [0.0] * len(requests)
        for owner, (logprob, _) in zip(owners, self._score(pairs), strict=True):
            totals[owner] += logprob
        return totals

    # Generation.
    def generate_until(self, requests):
        outputs = [None] * len(requests)
        order = sorted(range(len(requests)), key=lambda i: -len(requests[i].args[0]))
        for start in range(0, len(order), self.generation_rows):
            group = order[start : start + self.generation_rows]
            settings = [requests[i].args[1] for i in group]
            limit = max(int(s.get("max_gen_toks", 256)) for s in settings)
            stops = [s.get("until") or [] for s in settings]
            stops = [[u] if isinstance(u, str) else list(u) for u in stops]
            prompts = [self.encode(requests[i].args[0])[-(self.context - limit) :] for i in group]

            def done(row, ids, stops=stops):
                text = self.tokenizer.decode(ids)
                return any(stop in text for stop in stops[row])

            generated = generate.generate_batch(self.lm, prompts, limit, (self.eos,), done=done)
            for row, index in enumerate(group):
                ids = [t for t in generated[row] if t != self.eos]
                text = self.tokenizer.decode(ids)
                for stop in stops[row]:
                    text = text.split(stop)[0]
                outputs[index] = text
        return outputs


def load(folder, device="cuda"):
    lm = generate.load_export(folder, dtype=torch.bfloat16, device=device)
    return HybridHarnessModel(lm, glm.load())
