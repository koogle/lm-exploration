"""Draw the KDA recurrence of training/hybrid/kda.py on a toy head, mapped to its source lines.

From training/: uv run --with matplotlib python references/kda/figures.py references/kda
"""

import inspect
import os
import sys

import matplotlib
import torch
from model_hybrid import documents, kda
from torch.nn import functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

# One head of width 4 with one-hot keys, so every row of S is one memory slot.
# Token 3 reuses token 1's key with a new value.
D = 4
ALPHA = torch.tensor([0.9, 0.95, 0.5, 0.8], dtype=torch.float64)
KEYS = [0, 1, 0]
VALUES = torch.tensor([[1, 0, 0.5, 0], [0, 1, 0, -0.5], [-1, 0, 0, 1]], dtype=torch.float64)
BETA_LOGITS = torch.tensor([2.1972245773, 2.1972245773, 40.0], dtype=torch.float64)  # .9, .9, 1

eye = torch.eye(D, dtype=torch.float64)
k = eye[KEYS][None, :, None]  # [B, T, H, K]
v = VALUES[None, :, None]
# kda_gate with A_log = 0 and dt_bias = 0 gives log(alpha) = -softplus(g); invert softplus.
g = (1 / ALPHA - 1).log().expand(1, len(KEYS), 1, D)
A_log, dt_bias = torch.zeros(1, dtype=torch.float64), torch.zeros(D, dtype=torch.float64)
decay = kda.kda_gate(g, A_log, dt_bias).double().exp()  # fp32 gate, as in kda_reference
beta = torch.sigmoid(BETA_LOGITS)[None, :, None]

# The loop body of kda_reference, keeping every intermediate for the figure.
state, states, steps = torch.zeros(1, 1, D, D, dtype=torch.float64), [], []
states.append(state[0, 0])
for t in range(len(KEYS)):
    state = state * decay[:, t, ..., None]
    k_t = k[:, t]
    predicted = torch.einsum("bhk,bhkv->bhv", k_t, state)
    error = v[:, t] - predicted
    update = torch.einsum("bhk,bhv->bhkv", beta[:, t, :, None] * k_t, error)
    steps.append(
        {
            "decayed": state[0, 0],
            "predicted": predicted[0, 0],
            "error": error[0, 0],
            "update": update[0, 0],
        }
    )
    state = state + update
    states.append(state[0, 0])
reads = [eye[KEYS[t]] @ states[t + 1] for t in range(len(KEYS))]

# kda_reference itself, with q = k, must give the same reads (it also scales q by D^-1/2).
positions = torch.arange(len(KEYS))[None]
reference = kda.kda_reference(k, k, v, g, BETA_LOGITS[None, :, None], A_log, dt_bias, positions)
torch.testing.assert_close(reference[0, :, 0], torch.stack(reads) * D**-0.5)
assert torch.allclose(decay[0, 0, 0], ALPHA) and beta[0, 2, 0] == 1

# One channel of short_convolution on two packed documents; document 2 starts at t = 4.
CONV_X = torch.tensor([1, 2, 3, 4, 10, 20, 30], dtype=torch.float64)
CONV_W = torch.tensor(
    [0.1, 0.2, 0.3, 1.0], dtype=torch.float64
)  # nn.Conv1d order: oldest tap first
CONV_T = 5  # token whose window is drawn
conv_positions = documents.document_positions(torch.tensor([[0] * 4 + [1] * 3]))


def mix(positions):
    taps = range(len(CONV_W))
    return torch.stack(
        [
            sum(CONV_W[-1 - s] * CONV_X[t - s] for s in taps if positions[0, t] >= s)
            for t in range(len(CONV_X))
        ]
    )


mixed, mixed_one_document = mix(conv_positions), mix(torch.arange(len(CONV_X))[None])
conv_out = kda.short_convolution(CONV_X[None, :, None], CONV_W[None, None], conv_positions)
torch.testing.assert_close(conv_out[0, :, 0], F.silu(mixed))

# Source lines are looked up in kda.py, so the figure follows the code.
SOURCE = inspect.getsource(kda).splitlines()


def line(fragment, after=0):
    for number, text in enumerate(SOURCE, 1):
        if number > after and fragment in text:
            return number, text.strip()
    raise ValueError(f"{fragment!r} not found in kda.py")


# ---------------------------------------------------------------- drawing
OUT = sys.argv[1]
SURFACE, PANEL, INK, MUTED, RULE = "#f7f6f2", "#ffffff", "#222222", "#666666", "#d8d6cf"
ACCENT, CODE_FACE, CODE_EDGE, CODE_INK = "#a85c12", "#eef2f8", "#c9d3e3", "#1d3b6e"
STEP_FACE, STEP_EDGE, READ_FACE, READ_EDGE = "#fff8e6", "#b58a1a", "#e5f3e1", "#4a9a3a"
WIDTH, HEIGHT, C = 1500, 1904, 32  # canvas and cell size in pixels at 100 dpi
SANS, MONO = "DejaVu Sans", "DejaVu Sans Mono"

fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), facecolor=SURFACE)
ax = fig.add_axes((0, 0, 1, 1))
ax.set_xlim(0, WIDTH)
ax.set_ylim(HEIGHT, 0)
ax.axis("off")


def put(x, y, s, px=14, color=INK, mono=False, **kw):
    """Text at baseline (x, y); px is the font size in canvas pixels."""
    kw.setdefault("va", "baseline")
    ax.text(x, y, s, fontsize=px * 0.72, color=color, family=MONO if mono else SANS, **kw)


def box(x, y, w, h, face, edge, radius=8, **kw):
    style = f"round,pad=0,rounding_size={radius}"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, facecolor=face, edgecolor=edge, **kw))


def arrow(x1, y1, x2, y2):
    props = {
        "arrowstyle": "-|>",
        "color": "#555555",
        "lw": 1.6,
        "mutation_scale": 14,
        "shrinkA": 0,
        "shrinkB": 0,
    }
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=props)


def fmt(value):
    if abs(value) < 1e-9:
        return "0"
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text.replace("0.", ".", 1) if text.lstrip("-").startswith("0.") else text


def color(value, scale=1.0):
    t = min(abs(value) / scale, 1.0)
    base = (217, 95, 60) if value > 0 else (47, 111, 208)
    rgb = tuple((255 + (c - 255) * t) / 255 for c in base)
    return rgb, "white" if t > 0.6 else INK


def cells(values, x, y, vertical=False, size=C, scale=1.0):
    for i, value in enumerate(float(v) for v in values):
        cx, cy = (x, y + i * size) if vertical else (x + i * size, y)
        face, ink = color(value, scale)
        ax.add_patch(Rectangle((cx, cy), size, size, facecolor=face, edgecolor="#bbbbbb", lw=0.8))
        label = fmt(value)
        put(
            cx + size / 2,
            cy + size / 2,
            label,
            10 if len(label) > 3 else 11,
            ink,
            True,
            ha="center",
            va="center",
        )


def matrix(values, x, y, title=None, highlight=None):
    for r in range(D):
        cells(values[r], x, y + r * C)
    ax.add_patch(Rectangle((x, y), D * C, D * C, fill=False, edgecolor="#666666", lw=1.2))
    if highlight is not None:
        ax.add_patch(
            Rectangle(
                (x - 2, y + highlight * C - 2),
                D * C + 4,
                C + 4,
                fill=False,
                edgecolor="#111111",
                lw=2.5,
            )
        )
    if title:
        put(x + D * C / 2, y - 10, title, 16, weight="bold", ha="center")


def chip(x, y, w, fragment, after=0, text=None):
    number, code = line(fragment, after)
    box(x, y - 17, w, 25, CODE_FACE, CODE_EDGE, radius=5)
    put(x + 8, y, f"L{number}", 12, "#999999", True)
    put(x + 46, y, text or code, 13, CODE_INK, True)
    return number


def badge(x, y, n):
    ax.add_patch(Circle((x, y - 5), 11, facecolor=ACCENT, edgecolor="none"))
    put(x, y - 5, str(n), 13, "white", weight="bold", ha="center", va="center")


def panel(y, h, title):
    box(30, y, 1440, h, PANEL, RULE, radius=10, lw=1.5)
    put(50, y + 30, title, 19, weight="bold")


def op(x, y, s):
    put(x, y, s, 26, "#444444", weight="bold", ha="center", va="center")


put(30, 42, "The KDA recurrence, mapped to training/hybrid/kda.py", 26, weight="bold")
put(
    30,
    68,
    f"Toy head with d = {D} and one-hot keys, so each row is one memory slot (real heads: d = 128, dense"
    " unit-length keys). Lnn = line in kda.py. Red = positive, blue = negative.",
    12.5,
    MUTED,
)

# Panel 1: the loop unrolled.
panel(
    86,
    400,
    "1 · kda_reference unrolled: the loop body runs once per token and rewrites state (= S)",
)
alpha = ", ".join(fmt(a) for a in ALPHA.tolist())
put(
    60,
    150,
    f"decay α = [{alpha}] per key channel (row), the same each step here; in the model α depends on the token",
    12.5,
    MUTED,
)
my, xs, sx = 210, [60, 420, 780, 1140], [204, 564, 924]
for i, x in enumerate(xs):
    matrix(states[i], x, my, "S₀ = 0" if i == 0 else f"S{'₀₁₂₃'[i]}")
for r in range(D):
    put(46, my + 20 + r * C, f"row {r + 1}", 12.5, MUTED, ha="right")
put(60, my + 150, "rows = key channels, cols = value channels", 12.5, MUTED)
chip(60, my + 178, 300, "state = q.new_zeros(", text="state = q.new_zeros(...)")
body = [
    line(f)[0]
    for f in (
        "state = state * keep",
        "error = v[:, t]",
        "state = state + torch.einsum",
        "outputs.append(",
    )
]
slots = ["e₁ (slot 1)", "e₂ (slot 2)", "e₁ (slot 1 again)"]
for t, x in enumerate(sx):
    box(x, my - 12, 200, 152, STEP_FACE, STEP_EDGE, lw=1.5)
    put(x + 100, my + 12, f"t = {t}  (token {t + 1})", 14, weight="bold", ha="center")
    put(x + 14, my + 38, f"k_t = {slots[t]}", 11.5, mono=True)
    put(
        x + 14,
        my + 60,
        f"v[:, t] = [{', '.join(fmt(a) for a in VALUES[t].tolist())}]",
        11.5,
        mono=True,
    )
    put(x + 14, my + 82, f"beta[:, t] = {fmt(beta[0, t, 0].item())}", 11.5, mono=True)
    put(x + 100, my + 110, "loop body", 12.5, MUTED, ha="center")
    put(x + 100, my + 127, " → ".join(f"L{n}" for n in body), 12, CODE_INK, True, ha="center")
    arrow(x - 14, my + 64, x - 2, my + 64)
    arrow(x + 200, my + 64, x + 214, my + 64)
notes = [
    ("v₁ written into slot 1,", "scaled by β = .9"),
    ("v₂ written into slot 2;", "slot 1 faded by α₁ = .9"),
    ("slot 1 now holds v₃:", "the old v₁ was replaced"),
]
for x, (first, second) in zip(xs[1:], notes, strict=True):
    put(x + 64, my + 158, first, 12.5, MUTED, ha="center")
    put(x + 64, my + 175, second, 12.5, MUTED, ha="center")
rx = 1290
arrow(1270, my + 64, rx - 4, my + 64)
box(rx, my - 12, 165, 152, READ_FACE, READ_EDGE, lw=1.5)
put(rx + 82, my + 12, "read, q[:, t] = e₁", 14, weight="bold", ha="center")
put(rx + 82, my + 36, "outputs[t] = qᵀ S₃", 12.5, mono=True, ha="center")
cells(reads[2], rx + 18, my + 50)
put(rx + 82, my + 104, "= v₃: the latest value", 12.5, MUTED, ha="center")
put(rx + 82, my + 121, "stored under key 1", 12.5, MUTED, ha="center")
chip(
    420,
    my + 220,
    1030,
    "for t in range(length)",
    text="for t in range(length):     one iteration per token; state is [batch, heads, K, V] = one d × d S per head",
)
stack = line("return torch.stack(outputs")[0]
put(
    60,
    462,
    f"Line {stack} stacks outputs into [B, T, H, V]. state never grows: the same d × d memory per head is reused for the whole document.",
    14,
)

# Panel 2: one iteration, zoomed in on token 3.
panel(
    506,
    620,
    "2 · One iteration of the loop, zoomed in on t = 2 (token 3): k = e₁, v₃ = [−1, 0, 0, 1], β = 1",
)
zy, ly, step = 610, 576, steps[2]
badge(70, ly, 1)
put(88, ly, "Forget", 15, ACCENT, weight="bold")
put(158, ly, "scale row i by α_i", 12.5, MUTED)
matrix(states[2], 60, zy, "S₂")
op(214, zy + 64, "×")
cells(ALPHA, 232, zy, vertical=True)
put(248, zy - 10, "α", 16, weight="bold", ha="center")
op(290, zy + 64, "=")
matrix(step["decayed"], 312, zy, "S′", highlight=0)
px = 500
badge(px + 10, ly, 2)
put(px + 28, ly, "Predict", 15, ACCENT, weight="bold")
put(px + 96, ly, "what does slot k hold now?", 12.5, MUTED)
for dy, label, values in (
    (22, "v̂ = kᵀS′", step["predicted"]),
    (66, "v₃", VALUES[2]),
    (116, "e = v₃ − v̂", step["error"]),
):
    put(px, zy + dy, label, 14, mono=True)
    cells(values, px + 100, zy + dy - 20)
ax.plot([px, px + 228], [zy + 86, zy + 86], color="#999999", lw=1)
put(px, zy + 150, "e = error in the code", 12.5, MUTED)
cx = 790
badge(cx + 10, ly, 3)
put(cx + 28, ly, "Correct", 15, ACCENT, weight="bold")
put(cx + 98, ly, "write only the error, scaled by β", 12.5, MUTED)
matrix(step["decayed"], cx, zy, "S′")
op(cx + 152, zy + 64, "+")
matrix(step["update"], cx + 176, zy, "β · k eᵀ")
op(cx + 328, zy + 64, "=")
matrix(states[3], cx + 350, zy, "S₃", highlight=0)
qx = 1310
badge(qx + 10, ly, 4)
put(qx + 28, ly, "Read", 15, ACCENT, weight="bold")
arrow(cx + 486, zy + 64, qx - 4, zy + 64)
put(qx, zy + 22, "o = qᵀ S₃", 14, mono=True)
cells(reads[2], qx, zy + 40)
put(qx, zy + 100, "q = e₁ reads slot 1", 12.5, MUTED)
cy = 790
put(
    60,
    cy,
    "The same four operations in kda_reference (state = S, k_t = k, decay[:, t] = α, beta[:, t] = β):",
    14,
    weight="bold",
)
code = [
    ("state = state * keep", "S′ = diag(α_t) S. keep is 0 at a document start, which zeroes S."),
    ("error = v[:, t]", "v̂ = kᵀS′ is the einsum; error = v − v̂."),
    ("state = state + torch.einsum", "outer product (β k) eᵀ added to S′."),
    ("outputs.append(", "o_t = qᵀ S_t, read after the update."),
]
for i, (fragment, note) in enumerate(code):
    y = cy + 38 + i * 36
    badge(72, y, i + 1)
    chip(92, y, 760, fragment)
    put(870, y, note, 12.5, MUTED)
slice_line = line("k_t = k[:, t]")[0]
put(
    60,
    cy + 184,
    f"L{slice_line} k_t = k[:, t] slices this token's key. b = batch, h = head, k = key channel, v = value channel in the einsum strings.",
    12.5,
    MUTED,
)
dy = 1010
ax.plot([50, 1450], [dy - 30, dy - 30], color="#e6e4dd", lw=1)
predict = line("error = v[:, t]")[0]
put(
    60,
    dy - 6,
    f"Why line {predict} matters: drop the v̂ subtraction and the update becomes plain linear attention (add k vᵀ)",
    14,
    weight="bold",
)
put(60, dy + 28, f"without L{predict}:  S′ row 1 + v₃", 14, mono=True)
cells(step["decayed"][0] + VALUES[2], 380, dy + 8)
put(530, dy + 28, "old v₁ and new v₃ blurred together", 12.5, MUTED)
put(60, dy + 62, f"with L{predict}:     S′ row 1 + e", 14, mono=True)
cells(states[3][0], 380, dy + 42)
put(530, dy + 62, "exactly v₃: the slot is overwritten, not piled up", 12.5, MUTED)
put(
    880, dy + 20, "S_t = diag(α_t) S_{t−1} + β_t k_t (v_t − k_tᵀ diag(α_t) S_{t−1})ᵀ", 13, mono=True
)
put(880, dy + 44, "o_t = q_tᵀ S_t", 13, mono=True)

# Panel 3: the short convolution on one channel.
py, CW, x0 = 1146, 48, 200
conv_def = line("def short_convolution(")[0]
panel(
    py,
    390,
    f"3 · short_convolution (L{conv_def}): before the recurrence, each channel of q, k and v mixes its last 4 tokens",
)
put(
    60,
    py + 56,
    "One channel, two documents packed in one row. Weights are in nn.Conv1d order, oldest tap first;"
    " the current token's weight is weight[:, 0, -1].",
    12.5,
    MUTED,
)
put(x0 + 2 * CW, py + 82, "document 1", 12.5, MUTED, ha="center")
put(x0 + 5.5 * CW, py + 82, "document 2", 12.5, MUTED, ha="center")
for t in range(len(CONV_X)):
    put(x0 + (t + 0.5) * CW, py + 98, f"t = {t}", 11, MUTED, ha="center")
    put(x0 + (t + 0.5) * CW, py + 174, str(int(conv_positions[0, t])), 12, mono=True, ha="center")
put(60, py + 132, "x[:, t, c]", 14, mono=True)
cells(CONV_X, x0, py + 106, size=CW, scale=30)
start = CONV_T - len(CONV_W) + 1
ax.add_patch(
    Rectangle(
        (x0 + start * CW - 2, py + 104),
        len(CONV_W) * CW + 4,
        CW + 4,
        fill=False,
        edgecolor="#111111",
        lw=2.5,
    )
)
put(60, py + 174, "positions", 14, mono=True)
put(60, py + 214, f"weight, t = {CONV_T}", 14, mono=True)
cells(CONV_W, x0 + start * CW, py + 186, size=CW)
for s in range(len(CONV_W)):
    if conv_positions[0, CONV_T] < s:
        cx = x0 + (CONV_T - s) * CW
        ax.plot(
            [cx + 6, cx + CW - 6], [py + 192, py + 186 + CW - 6], color="#c94a4a", lw=1.5, alpha=0.7
        )
        ax.plot(
            [cx + 6, cx + CW - 6], [py + 186 + CW - 6, py + 192], color="#c94a4a", lw=1.5, alpha=0.7
        )
terms = " + ".join(
    f"{fmt(CONV_W[-1 - s].item())}·{fmt(CONV_X[CONV_T - s].item()) if conv_positions[0, CONV_T] >= s else 0}"
    for s in range(len(CONV_W))
)
put(60, py + 258, f"t = {CONV_T}:  {terms} = {fmt(mixed[CONV_T].item())}", 13, mono=True)
masked = " and ".join(
    str(t) for t in range(start, CONV_T + 1) if conv_positions[0, CONV_T] < CONV_T - t
)
put(60, py + 276, f"taps at t = {masked} would read document 1, so they are zeroed", 12.5, MUTED)
put(60, py + 310, "before SiLU", 14, mono=True)
cells(mixed, x0, py + 288, size=CW, scale=40)
put(60, py + 360, "q[:, t, c]", 14, mono=True)
cells(conv_out[0, :, 0], x0, py + 338, size=CW, scale=40)
put(x0 + 7 * CW + 14, py + 360, "after SiLU", 12.5, MUTED)
ax.plot([x0 + 4 * CW] * 2, [py + 68, py + 104], color="#999999", lw=1.2, ls="--")
cx = 640
put(cx, py + 98, "The same steps in short_convolution:", 14, weight="bold")
steps_code = [
    ("out = x * weight[:, 0, -1]", "the current token"),
    ("for shift in range(1, width)", "shifts 1 … 3"),
    ("previous = F.pad(x", "x[t − shift], 0 before t = 0"),
    ("previous = previous.masked_fill(", "drop the previous document"),
    ("out = out + previous * weight", "add the weighted tap"),
    ("return F.silu(out)", "SiLU"),
]
for i, (fragment, note) in enumerate(steps_code):
    y = py + 132 + i * 34
    chip(cx, y, 590, fragment, conv_def)
    put(cx + 606, y, note, 12.5, MUTED)
calls = ", ".join(f"L{line(f'self.{n}_conv', line('def forward(')[0])[0]}" for n in "qkv")
put(cx, py + 338, f"{calls}: q, k and v each run this with their own weights.", 12.5, MUTED)
put(
    cx,
    py + 360,
    f"Without the document reset, t = 4 and 5 would read document 1:"
    f" {fmt(mixed_one_document[4].item())} and {fmt(mixed_one_document[5].item())} before SiLU.",
    12.5,
    MUTED,
)

# Panel 4: where the inputs come from.
panel(
    1556,
    332,
    "4 · Where q, k, v, α, β come from: KimiDeltaAttention.forward, then the top of kda_reference",
)
forward = line("def forward(")[0]
rows = [
    ("q, k, v", "self.q_proj(x)", forward, "projections, then short_convolution (panel 3)"),
    ("q, k", "q = F.normalize(", 0, "unit-length q and k, q scaled by d^−½"),
    ("α", "g = self.f_proj(x)", forward, "raw gate, low-rank projection"),
    ("α", "g = g.float() + dt_bias", 0, "inside kda_gate: add the per-channel bias"),
    ("α", "return -A_log", 0, "log of α, always ≤ 0; A_log sets each head's rate"),
    ("α", "decay = kda_gate(", 0, "α in (0, 1), one per key channel and token"),
    (
        "β",
        "beta = self.b_proj(x)",
        forward,
        f"then L{line('torch.sigmoid(beta')[0]}: sigmoid, one per head and token",
    ),
    (
        "loop",
        "out = kda_kernel(",
        forward,
        f"CUDA: fla chunk_kda, same maths in chunks; CPU: L{line('out = kda_reference(')[0]}",
    ),
    (
        "after",
        "values = values * torch.rsqrt(",
        forward,
        "RMSNorm per head, then × sigmoid(g_proj(x)) and o_proj",
    ),
]
for i, (symbol, fragment, after, note) in enumerate(rows):
    y = 1620 + i * 31
    put(60, y, symbol, 14, mono=True, weight="bold")
    chip(140, y, 720, fragment, after)
    put(876, y, note, 12.5, MUTED)

fig.savefig(os.path.join(OUT, "kda_recurrence.png"), dpi=150, facecolor=SURFACE)
