"""Plot the paired losses and convolution interventions from saved diagnostics."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("folder", type=Path)
args = parser.parse_args()
p = args.folder
j = json.loads((p / "analysis.json").read_text())
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
labels = ["climbmix", "code", "math", "weighted"]
for shift, v, c, label in [
    (-0.11, "none", "#16856b", "No convolution"),
    (0.11, "rope_triton", "#7952b3", "RoPE, no convolution"),
]:
    vals = j["paired_differences"][v]
    a = np.array([vals[k]["mean"] for k in labels])
    ci = np.array([vals[k]["ci95"] for k in labels])
    ax[0].errorbar(
        np.arange(4) + shift,
        a,
        yerr=np.array([a - ci[:, 0], ci[:, 1] - a]),
        fmt="o",
        capsize=3,
        color=c,
        label=label,
    )
ax[0].axhline(0, c="gray", lw=1)
ax[0].set_xticks(range(4), ["ClimbMix", "Code", "Math", "70/20/10"])
ax[0].set_ylabel("CE minus baseline (nats/token)")
ax[0].set_title("Paired held-out documents")
ax[0].legend(fontsize=8)
keys = ["remove_past_q_conv", "remove_past_k_conv", "remove_past_v_conv", "bf16_recurrence_inputs"]
a = np.array([j["interventions"]["baseline"][k] for k in keys])
ax[1].bar(range(4), a.mean(1), color="#4e708b")
ax[1].scatter(np.repeat(np.arange(4), 2), a.flatten(), c="black", s=15, zorder=3)
ax[1].set_xticks(range(4), ["q: no past", "k: no past", "v: no past", "BF16 inputs"], rotation=20)
ax[1].set_ylabel("CE increase after intervention")
ax[1].set_title("What trained baseline relies on")
for offset, kind, c in [
    (-0.24, "q_conv", "#4e708b"),
    (0, "k_conv", "#cb8d28"),
    (0.24, "v_conv", "#7952b3"),
]:
    data = []
    for seed in [17, 29]:
        r = json.loads((p / f"baseline-{seed}.json").read_text())
        data.extend(
            v["rms_by_lag_3_to_0"] for k, v in r["convolution_weights"].items() if k.endswith(kind)
        )
    ax[2].bar(np.arange(4) + offset, np.mean(data, 0), width=0.23, label=kind[0], color=c)
ax[2].set_xticks(range(4), ["t−3", "t−2", "t−1", "t"])
ax[2].set_ylabel("Learned tap RMS (layer/seed mean)")
ax[2].set_title("Queries emphasize now; k/v emphasize t−1")
ax[2].legend()
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.savefig(p / "diagnostics.png", dpi=180)
