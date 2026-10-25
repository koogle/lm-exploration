"""Draw the text "hello" through layer 0, head 0 of the 10M model with a KV cache.

From training/: uv run --with matplotlib python references/kv-cache/figures.py references/kv-cache
"""

import importlib.util
import math
import sys

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

spec = importlib.util.spec_from_file_location("model10m", "10m/model.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# Tiny Shakespeare character vocabulary (65 symbols), same as the published 10M model.
VOCAB = (
    ["\n", " ", "!", "$", "&", "'", ",", "-", ".", "3", ":", ";", "?"]
    + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
)
TEXT, PROMPT = "hello", 3
ids = [VOCAB.index(c) for c in TEXT]

torch.manual_seed(0)
config = m.ModelConfig(vocab_size=len(VOCAB), block_size=32, n_layer=2, n_head=2, n_embd=8)
model = m.GPT(config).eval()
with torch.no_grad():  # illustration: larger random qkv weights so scores are not all ~0
    for block in model.blocks:
        torch.nn.init.normal_(block.attention.qkv.weight, std=0.5)


def attention_step(attn, x, cache, record):
    batch, length, width = x.shape
    q, k, v = attn.qkv(x).chunk(3, -1)
    q, k, v = [
        t.view(batch, length, attn.n_head, width // attn.n_head).transpose(1, 2) for t in (q, k, v)
    ]
    past = 0 if cache["k"] is None else cache["k"].size(-2)
    q, k = m.rotary(q, offset=past), m.rotary(k, offset=past)
    if cache["k"] is not None:
        k, v = torch.cat([cache["k"], k], -2), torch.cat([cache["v"], v], -2)
    cache["k"], cache["v"] = k, v
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    mask = torch.ones(q.size(-2), k.size(-2), dtype=torch.bool).tril(diagonal=past)
    weights = scores.masked_fill(~mask, float("-inf")).softmax(-1)
    out = weights @ v
    if record is not None:
        record.update(
            past=past,
            q=q[0, 0],
            k=k[0, 0],
            v=v[0, 0],
            scores=scores[0, 0],
            mask=mask,
            weights=weights[0, 0],
            out=out[0, 0],
        )
    return attn.projection(out.transpose(1, 2).contiguous().view(batch, length, width))


def step(tokens, caches, record):
    x = model.token_embedding(tokens)
    for i, (block, cache) in enumerate(zip(model.blocks, caches)):
        x = x + attention_step(
            block.attention, block.attention_norm(x), cache, record if i == 0 else None
        )
        x = x + block.mlp(block.mlp_norm(x))
    return model.lm_head(model.final_norm(x))


caches = [{"k": None, "v": None} for _ in range(config.n_layer)]
passes = []
with torch.no_grad():
    chunks = [ids[:PROMPT]] + [[i] for i in ids[PROMPT:]]
    for chunk in chunks:
        record = {"new": chunk}
        logits = step(torch.tensor([chunk]), caches, record)
        passes.append(record)
    full = model(torch.tensor([ids]))[:, -1]
    assert torch.allclose(logits[:, -1], full, atol=1e-6), "cache result differs from full forward"
    embedding = model.token_embedding.weight[ids].numpy()

# ---------------------------------------------------------------- drawing
import os

OUT = sys.argv[1]
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
NEW, CACHED, BLOCKED, LINE = "#2a78d6", "#d9d8d3", "#f0efec", "#fcfcfb"
W, H = 1.15, 0.75  # cell size in data units
SCALE = 0.62  # inches per data unit


def canvas(width, height):
    fig = plt.figure(figsize=(width * SCALE, height * SCALE), facecolor=SURFACE)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(-height, 0)
    ax.axis("off")
    return fig, ax


def put(ax, x, y, s, size=12, color=INK, **kw):
    ax.text(x, y, s, fontsize=size, color=color, **kw)


def grid(ax, x, y, values, colors, rows=(), cols=(), fmt="{:+.2f}", title=None):
    """Matrix with top-left corner at (x, y); returns its right edge."""
    if title:
        put(ax, x, y + 0.55, title, 12.5, INK, va="bottom", weight="bold")
    for r, row in enumerate(values):
        for c, value in enumerate(row):
            face = colors[r][c]
            ax.add_patch(
                Rectangle((x + c * W, y - (r + 1) * H), W, H, facecolor=face, edgecolor=LINE, lw=2)
            )
            label = value if isinstance(value, str) else fmt.format(value)
            ax.text(
                x + (c + 0.5) * W,
                y - (r + 0.5) * H,
                label,
                ha="center",
                va="center",
                fontsize=10.5,
                color="white" if face == NEW else (MUTED if face == BLOCKED else INK),
            )
    for r, label in enumerate(rows):
        put(ax, x - 0.15, y - (r + 0.5) * H, label, 10.5, MUTED, ha="right", va="center")
    for c, label in enumerate(cols):
        put(ax, x + (c + 0.5) * W, y + 0.08, label, 10, MUTED, ha="center", va="bottom")
    return x + len(values[0]) * W


def op(ax, x, y, s):
    put(ax, x, y, s, 20, INK, ha="center", va="center")


def show(ch):
    return "␣" if ch == " " else ch


def legend(ax, x, y):
    for i, (face, label) in enumerate(
        [
            (NEW, "computed in this pass"),
            (CACHED, "read from KV cache"),
            (BLOCKED, "masked (future token)"),
        ]
    ):
        ax.add_patch(
            Rectangle((x + i * 6.2, y - 0.45), 0.6, 0.45, facecolor=face, edgecolor=CACHED, lw=0.5)
        )
        put(ax, x + i * 6.2 + 0.8, y - 0.22, label, 11, MUTED, va="center")


# ---- image 1: text -> tokens -> embeddings -> q/k/v
fig, ax = canvas(34, 15)
put(ax, 0.6, -1.0, 'Step 0: the text "hello" becomes tokens, then vectors', 20, va="center")
put(
    ax,
    0.6,
    -2.1,
    "Character-level 10M model, shrunk to width 8 with 2 heads, so each head works with "
    "D = 4 numbers. Untrained random weights (qkv drawn with std 0.5 instead of 0.02 so the scores are not all ~0);\nevery number in these images is a real output of training/10m/model.py, and the cached result matches a full no-cache forward.",
    11.5,
    MUTED,
    va="center",
)
for i, ch in enumerate(TEXT):
    x = 1 + i * 2.4
    ax.add_patch(
        FancyBboxPatch(
            (x, -4.4),
            1.8,
            1.2,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor="white",
            edgecolor=CACHED,
        )
    )
    put(ax, x + 0.9, -3.8, f"'{ch}'", 18, ha="center", va="center")
    put(ax, x + 0.9, -4.9, f"id {ids[i]}", 13, ha="center", va="center")
    put(ax, x + 0.9, -5.5, f"position {i}", 10.5, MUTED, ha="center", va="center")
put(
    ax,
    1,
    -2.85,
    "1. split into characters; id = index in the 65-symbol vocabulary",
    12,
    va="center",
)
put(ax, 1, -6.4, "2. each id selects one row of token_embedding (8 numbers):", 12, va="center")
grid(
    ax,
    3.2,
    -7.2,
    embedding.tolist(),
    [["white"] * 8 for _ in ids],
    rows=[f"'{c}' (id {i})" for c, i in zip(TEXT, ids)],
    cols=[f"e{j}" for j in range(8)],
)
put(
    ax,
    14.6,
    -6.4,
    "3. inside each attention layer, every token's 8 numbers go through:",
    12,
    va="center",
)
steps = [
    "RMSNorm",
    "qkv Linear: 8 → 24 numbers",
    "split: q (8), k (8), v (8)",
    "split each into 2 heads × 4",
    "RoPE: rotate q and k by position",
]
for i, s in enumerate(steps):
    ax.add_patch(
        FancyBboxPatch(
            (15, -7.6 - i * 1.25),
            8.4,
            0.85,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor="white",
            edgecolor=CACHED,
        )
    )
    put(ax, 19.2, -7.17 - i * 1.25, s, 11.5, ha="center", va="center")
put(
    ax,
    24.2,
    -9.6,
    "Result per token, per head:\n  q = what am I looking for?\n  k = what do I contain?\n"
    "  v = what do I hand over if picked?\n\nk and v of every past token are needed\nby every later token "
    "→ they get cached.\nq is only needed by its own token.",
    12,
    va="center",
)
put(
    ax,
    0.6,
    -13.4,
    "Generation plan: the prompt \"hel\" goes through in one pass (prefill). Then 'l' and 'o' are "
    "fed one at a time (decode), each reusing the cache.\nThe real next letters are fed instead of sampled "
    "ones, because the model is untrained.  The next images follow layer 0, head 0.",
    12,
    va="center",
)
fig.savefig(os.path.join(OUT, "kv1_tokens.png"), dpi=110, facecolor=SURFACE)
plt.close(fig)

# ---- images 2-4: one forward pass each
for n, rec in enumerate(passes):
    past, t_new, t_all = rec["past"], len(rec["new"]), rec["k"].shape[0]
    chars = [show(c) for c in TEXT[:t_all]]
    qpos = list(range(past, past + t_new))
    is_new = [p >= past for p in range(t_all)]
    keep = rec["mask"]

    def cell(r, c, keep=keep):
        return "white" if keep[r, c] else BLOCKED

    fig, ax = canvas(40, 18.6)
    if n == 0:
        title = "Pass 1 — prefill: feed 'h' 'e' 'l' together (cache empty, past = 0)"
        sub = "All 3 tokens are new, so q, k and v are computed for all of them. k and v are stored as the start of the cache."
    else:
        title = (
            f"Pass {n + 1} — decode: feed only '{TEXT[past]}' (past = {past} tokens already cached)"
        )
        sub = (
            f"Only the new token goes through the layer: 1 q, 1 k, 1 v. The new k and v rows are appended to "
            f"the cache ({past} → {t_all} rows). The grey rows are reused exactly as stored."
        )
    put(ax, 0.6, -1.0, title, 20, va="center")
    put(ax, 0.6, -1.85, sub, 12, MUTED, va="center")
    legend(ax, 0.6, -2.4)

    # row A: scores = Q @ K^T / sqrt(D), then mask
    yA = -4.6
    put(
        ax,
        0.6,
        yA + 1.35,
        "A.  How much should each new token attend to each cached token?",
        14,
        va="center",
    )
    x = grid(
        ax,
        3.0,
        yA,
        rec["q"].tolist(),
        [[NEW] * 4] * t_new,
        rows=[f"q{p} '{chars[p]}'" for p in qpos],
        cols=[f"d{j}" for j in range(4)],
        title="Q (new tokens)",
    )
    op(ax, x + 0.6, yA - 1.5, "@")
    x = grid(
        ax,
        x + 2.0,
        yA,
        rec["k"].T.tolist(),
        [[NEW if f else CACHED for f in is_new]] * 4,
        rows=[f"d{j}" for j in range(4)],
        cols=[f"k{p} '{c}'" for p, c in enumerate(chars)],
        title="Kᵀ (whole cache)",
    )
    op(ax, x + 1.0, yA - 1.5, "/ √4  =")
    x = grid(
        ax,
        x + 2.6,
        yA,
        rec["scores"].tolist(),
        [[cell(r, c) for c in range(t_all)] for r in range(t_new)],
        rows=[f"q{p}" for p in qpos],
        cols=[f"k{p} '{c}'" for p, c in enumerate(chars)],
        title="scores",
    )
    op(ax, x + 1.2, yA - 1.5, "mask →")
    masked = [
        [f"{rec['scores'][r, c]:+.2f}" if keep[r, c] else "−∞" for c in range(t_all)]
        for r in range(t_new)
    ]
    grid(
        ax,
        x + 2.7,
        yA,
        masked,
        [[cell(r, c) for c in range(t_all)] for r in range(t_new)],
        rows=[f"q{p}" for p in qpos],
        cols=[f"k{p} '{c}'" for p, c in enumerate(chars)],
        title=f"masked: tril(diagonal={past})",
    )

    # row B: weights @ V = output
    yB = -11.0
    put(ax, 0.6, yB + 1.35, "B.  softmax each row, then mix the cached values", 14, va="center")
    x = grid(
        ax,
        3.0,
        yB,
        rec["weights"].tolist(),
        [[cell(r, c) for c in range(t_all)] for r in range(t_new)],
        fmt="{:.2f}",
        rows=[f"q{p} '{chars[p]}'" for p in qpos],
        cols=[f"k{p} '{c}'" for p, c in enumerate(chars)],
        title="weights = softmax(masked)",
    )
    op(ax, x + 0.6, yB - 1.5, "@")
    x = grid(
        ax,
        x + 2.2,
        yB,
        rec["v"].tolist(),
        [[NEW if f else CACHED] * 4 for f in is_new],
        rows=[f"v{p} '{c}'" for p, c in enumerate(chars)],
        cols=[f"d{j}" for j in range(4)],
        title="V (whole cache)",
    )
    op(ax, x + 0.6, yB - 1.5, "=")
    x = grid(
        ax,
        x + 2.2,
        yB,
        rec["out"].tolist(),
        [[NEW] * 4] * t_new,
        rows=[f"out{p} '{chars[p]}'" for p in qpos],
        cols=[f"d{j}" for j in range(4)],
        title="output (new tokens)",
    )
    if n == 0:
        notes = [
            "Row q0 'h' can only see k0: −∞ hides 'e' and 'l' (its future), so its weight is 1.00 on itself.",
            "Each row of weights sums to 1. −∞ becomes exactly 0 after softmax.",
            "After this pass: K and V caches hold 3 rows. Q is discarded.",
        ]
    else:
        notes = [
            (
                "One query row, and every cached token is in its past, so nothing is masked "
                f"(tril with diagonal={past} keeps all {t_all} columns)."
            ),
            (
                f"Output: only 1 row, for '{TEXT[past]}'. The outputs for earlier tokens were produced "
                "in earlier passes and never change."
            ),
            f"After this pass: K and V caches hold {t_all} rows. This q is discarded.",
        ]
    for i, s in enumerate(notes):
        put(ax, 3.0, yB - 4.9 - i * 0.8, "• " + s, 12, INK, va="center")
    fig.savefig(os.path.join(OUT, f"kv{n + 2}_pass{n + 1}.png"), dpi=110, facecolor=SURFACE)
    plt.close(fig)
print("ok")
