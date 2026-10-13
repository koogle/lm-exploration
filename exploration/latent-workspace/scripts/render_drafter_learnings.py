"""Export the short research write-up and three evidence charts as one page."""

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
fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.35), layout="constrained")
a = axes[0]
a.bar(
    range(4),
    [0.384436, 0.484354, 0.519650, 0.530051],
    color=["#78909c", "#4c78a8", "#4c78a8", "#4c78a8"],
)
a.set(
    xticks=range(4),
    xticklabels=["Original", "+ Q/K norm", "5 layers", "5 depths"],
    ylim=(0, 0.65),
    title="Matched early architecture checks",
    ylabel="Validation NEW / 7",
)
a.tick_params(axis="x", rotation=20)
for i, v in enumerate([0.384, 0.484, 0.520, 0.530]):
    a.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
r = json.loads((root / "results/large-curriculum-final.json").read_text())["reports"]["256"]
c = [p for p in r["curve"] if p["step"] > r["curriculum_transition"]["step"]]
a = axes[1]
a.plot(
    [p["step"] / 1000 for p in c],
    [p["validation"]["accepted_beyond_anchor"] for p in c],
    label="Validation",
)
selected = next(p for p in c if p["step"] == 70585)
final = c[-1]
a.scatter(
    [selected["step"] / 1000, final["step"] / 1000],
    [
        selected["validation"]["accepted_beyond_anchor"],
        final["validation"]["accepted_beyond_anchor"],
    ],
    color=["#2a9d8f", "#d97732"],
    zorder=3,
)
a.annotate(
    "Selected 4.033",
    (70.585, 4.03251),
    xytext=(42, 5.15),
    fontsize=8,
    arrowprops={"arrowstyle": "->", "color": "#555555"},
)
a.annotate(
    "Final 4.007",
    (100, 4.00693),
    xytext=(64, 3.1),
    fontsize=8,
    arrowprops={"arrowstyle": "->", "color": "#555555"},
)
a.text(0.04, 0.06, "Late validation NLL: 2.105 → 2.171 (+3.1%)", transform=a.transAxes, fontsize=7)
a.set(
    title="Large long fit: modest late decline",
    xlabel="Updates (thousands)",
    ylabel="Correct NEW / 7",
    ylim=(0, 7),
)
r = json.loads((root / "results/qualification-long-196k-01.json").read_text())
a = axes[2]
keys = ["prose", "code", "translation"]
x = range(3)
a.bar(
    [i - 0.18 for i in x],
    [
        r["frozen"]["selected_validation"]["by_domain"][d]["mean_accepted_beyond_anchor"]
        for d in keys
    ],
    0.36,
    label="Validation",
)
a.bar(
    [i + 0.18 for i in x],
    [r["test_metrics"]["by_domain"][d]["mean_accepted_beyond_anchor"] for d in keys],
    0.36,
    label="Sealed test",
)
a.set(
    xticks=list(x),
    xticklabels=["Prose", "Code", "Translation"],
    ylim=(0, 7),
    title="Frozen held-out result",
    ylabel="Correct NEW / 7",
)
a.legend(fontsize=7)
for a in axes:
    a.tick_params(labelsize=8)
    a.title.set_fontsize(10)
    a.grid(axis="y", alpha=0.15)

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

{\footnotesize \textbf{Reading the charts.} NEW means consecutive correct draft tokens beyond the supplied anchor, out of seven. Middle/right share a metric and scale; the left panel shows earlier, weaker models. One seed; exposure budgets differ across experiments. Matching Qwen is not a human-quality score.\par}
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
