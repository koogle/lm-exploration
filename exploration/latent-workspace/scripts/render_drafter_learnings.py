"""Export the short research write-up and performance and scoring visuals as one page."""

import json
import re
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(".")
text = (
    root / "docs/implementing-a-diffusion-style-drafter-from-scratch-and-learnings.md"
).read_text()
paragraphs = text.split("\n\n")[1:6]
from matplotlib.patches import FancyBboxPatch

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.4, 3.9),
    layout="constrained",
    gridspec_kw={"width_ratios": [1.1, 1]},
)
r = json.loads((root / "results/large-curriculum-final.json").read_text())["reports"]["256"]
c = r["curve"]
a = axes[0]
colors = {"prose": "#4c78a8", "code": "#2a9d8f", "translation": "#d97732"}
for domain, color in colors.items():
    a.plot(
        [p["step"] / 1000 for p in c],
        [p["validation"]["by_domain"][domain]["mean_accepted_beyond_anchor"] for p in c],
        label=domain.capitalize(),
        color=color,
        linewidth=1.5,
    )
a.plot(
    [p["step"] / 1000 for p in c],
    [p["validation"]["accepted_beyond_anchor"] for p in c],
    label="Average (equal domain weights)",
    color="#202020",
    linewidth=2.3,
)
a.axvline(r["curriculum_transition"]["step"] / 1000, color="#999999", linestyle=":")
a.text(14, 0.4, "Translation added", fontsize=7, color="#555555")
a.scatter([70.585, 100], [4.03251, 4.00693], color="#202020", zorder=4)
a.annotate(
    "Selected: 4.033",
    (70.585, 4.03251),
    xytext=(39, 3.25),
    fontsize=8,
    arrowprops={"arrowstyle": "->", "color": "#555555"},
)
a.text(
    0.03, 0.98, "Final average: 4.007; late NLL +3.1%", transform=a.transAxes, fontsize=8, va="top"
)
a.set(
    title="Validation performance by domain and average",
    xlabel="Updates (thousands)",
    ylabel="Consecutive correct NEW tokens / 7",
    ylim=(0, 7),
)
a.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, 0.91), framealpha=0.9)
a.tick_params(labelsize=8)
a.title.set_fontsize(10)
a.grid(axis="y", alpha=0.15)

# Illustrative scoring example; not a measured model output.
a = axes[1]
a.set(xlim=(0, 6.3), ylim=(0, 4.8), title="How one draft block is scored")
a.title.set_fontsize(10)
a.axis("off")
a.text(0, 4.45, "Illustrative token IDs shown as words", fontsize=8, color="#555555")
a.text(1.03, 4.05, "Anchor", ha="center", fontsize=8)
a.text(3.62, 4.05, "Seven NEW draft tokens", ha="center", fontsize=8)
target = ["The", "cat", "sat", "on", "the", "mat", ".", "EOS"]
draft = ["The", "cat", "sat", "on", "the", "rug", ".", "EOS"]
for row, words, y in [("Qwen", target, 3.5), ("Draft", draft, 2.9)]:
    a.text(0, y + 0.18, row, fontsize=8, va="center")
    for i, word in enumerate(words):
        fill = "#e6e8eb" if i == 0 or i > 5 else "#cbe8db" if i < 5 else "#f3c8c0"
        x = 0.75 + i * 0.64
        a.add_patch(
            FancyBboxPatch(
                (x, y), 0.56, 0.36, boxstyle="round,pad=0.02", facecolor=fill, edgecolor="white"
            )
        )
        a.text(x + 0.28, y + 0.18, word, ha="center", va="center", fontsize=8)
a.text(0.75, 2.5, "Excluded", fontsize=7, color="#555555")
a.text(1.4, 2.5, "4 exact matches", fontsize=8, color="#18714b")
a.text(3.95, 2.5, "First error", fontsize=7, color="#a63b2b")
a.text(4.8, 2.5, "Not counted", fontsize=7, color="#555555")
a.text(0, 1.98, "Block score = 4 NEW (out of 7)", fontsize=11, weight="bold")
a.text(0, 1.5, "Average sampled blocks within each domain", fontsize=9)
a.text(0, 1.08, "Then average the three domain scores equally", fontsize=9)
a.text(0, 0.6, "Test: (5.01 prose + 4.96 code + 1.84 translation) / 3", fontsize=8)
a.text(0, 0.15, "= 3.94 NEW overall", fontsize=11, weight="bold")
fig.savefig(root / "results/drafter-learnings-charts.pdf")


def latex_escape(value):
    escapes = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "→": r"$\rightarrow$",
        "’": "'",
        "–": "--",
    }
    parts = re.split(r"\*\*(.*?)\*\*", value)
    return "".join(
        (r"\textbf{" if i % 2 else "")
        + "".join(escapes.get(char, char) for char in part)
        + ("}" if i % 2 else "")
        for i, part in enumerate(parts)
    )


body = "\n\n".join(latex_escape(p) for p in paragraphs[:3])
body += "\n\n" + r"\columnbreak" + "\n\n"
body += "\n\n".join(latex_escape(p) for p in paragraphs[3:])
source = (
    r"""% Generated from the Markdown write-up by scripts/render_drafter_learnings.py.
\documentclass[11pt,a4paper]{article}
\usepackage[margin=16mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern,microtype,multicol,graphicx,xcolor}
\usepackage[hidelinks]{hyperref}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlength{\columnsep}{7mm}
\definecolor{heading}{HTML}{274963}
\begin{document}
{\LARGE\bfseries Implementing and training a\\diffusion-style drafter\par}
{\large\color{heading} From scratch: learnings\par}
\vspace{2mm}
\begin{multicols}{2}
"""
    + body
    + r"""
\end{multicols}
\vspace{1mm}
\includegraphics[width=\linewidth]{drafter-learnings-charts.pdf}

{\footnotesize \textbf{Reading the charts.} The performance curve uses held-out teacher trajectories; live verified rollouts are measured separately. The scoring example is illustrative. Domain scores average sampled blocks (up to eight per prompt); the overall score weights domains equally. One seed; matching Qwen is not human answer quality.\par}
\vspace{1mm}
{\footnotesize \textbf{References.} \href{https://arxiv.org/abs/2602.06036}{DFlash (arXiv:2602.06036)}; \href{https://transformer-circuits.pub/2026/workspace/index.html}{Anthropic workspace / Jacobian lens}. Full reports and protocols are retained in the repository.\par}
\end{document}
"""
)
(root / "results/drafter-learnings.tex").write_text(source)
with tempfile.TemporaryDirectory() as build:
    subprocess.run(
        [
            "pdflatex",
            "-halt-on-error",
            "-interaction=nonstopmode",
            f"-output-directory={build}",
            "drafter-learnings.tex",
        ],
        cwd=root / "results",
        check=True,
        stdout=subprocess.DEVNULL,
    )
    log = Path(build, "drafter-learnings.log").read_text()
    if "Overfull" in log:
        raise RuntimeError("LaTeX overflow detected; inspect the page layout")
    pdf = Path(build, "drafter-learnings.pdf")
    info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    if not re.search(r"Pages:\s+1\b", info):
        raise RuntimeError("Write-up exceeds one page")
    (root / "results/drafter-learnings.pdf").write_bytes(pdf.read_bytes())
    subprocess.run(
        [
            "pdftoppm",
            "-singlefile",
            "-scale-to",
            "2000",
            "-png",
            str(pdf),
            str((root / "results/drafter-learnings").resolve()),
        ],
        check=True,
    )
print("Body words:", len(re.findall(r"\S+", " ".join(paragraphs))))
print("LaTeX: one page, no overfull boxes")
