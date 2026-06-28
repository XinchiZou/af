from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.2,
        "axes.linewidth": 0.85,
        "axes.edgecolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

groups = ["G1", "G2", "G3", "G4", "G5"]
methods = ["FlowKG", "KMDCL", "DiffKG", "KGRec", "KGAT"]

ndcg_user_results = {
    "FlowKG": [0.245, 0.251, 0.248, 0.255, 0.259],
    "KMDCL": [0.216, 0.221, 0.210, 0.226, 0.231],
    "DiffKG": [0.201, 0.211, 0.198, 0.220, 0.225],
    "KGRec": [0.143, 0.163, 0.147, 0.170, 0.174],
    "KGAT": [0.164, 0.170, 0.159, 0.176, 0.181],
}

ndcg_item_results = {
    "FlowKG": [0.246, 0.252, 0.249, 0.256, 0.260],
    "KMDCL": [0.217, 0.223, 0.217, 0.227, 0.232],
    "DiffKG": [0.202, 0.211, 0.201, 0.221, 0.225],
    "KGRec": [0.161, 0.166, 0.159, 0.172, 0.175],
    "KGAT": [0.165, 0.171, 0.161, 0.175, 0.180],
}

recall_user_results = {
    "FlowKG": [0.402, 0.412, 0.407, 0.418, 0.422],
    "KMDCL": [0.375, 0.382, 0.371, 0.386, 0.392],
    "DiffKG": [0.348, 0.361, 0.343, 0.372, 0.378],
    "KGRec": [0.255, 0.288, 0.262, 0.295, 0.301],
    "KGAT": [0.198, 0.207, 0.191, 0.215, 0.222],
}

recall_item_results = {
    "FlowKG": [0.405, 0.414, 0.408, 0.420, 0.425],
    "KMDCL": [0.376, 0.385, 0.378, 0.388, 0.395],
    "DiffKG": [0.349, 0.363, 0.350, 0.374, 0.379],
    "KGRec": [0.280, 0.291, 0.278, 0.298, 0.304],
    "KGAT": [0.201, 0.209, 0.195, 0.216, 0.221],
}

colors = {
    "FlowKG": "#D94A3A",
    "KMDCL": "#3F78A8",
    "DiffKG": "#59A89C",
    "KGRec": "#8B78B8",
    "KGAT": "#D6A84B",
}

panels = [
    (recall_user_results, "Recall@20", "User Cold-Start Group"),
    (ndcg_user_results, "NDCG@20", "User Cold-Start Group"),
    (recall_item_results, "Recall@20", "Item Sparsity Degree"),
    (ndcg_item_results, "NDCG@20", "Item Sparsity Degree"),
]

fig, axes = plt.subplots(2, 2, figsize=(3.55, 3.25))
x = np.arange(len(groups))
width = 0.135
offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * width

for ax, (results, ylabel, xlabel) in zip(axes.flat, panels):
    for offset, method in zip(offsets, methods):
        ax.bar(
            x + offset,
            results[method],
            width=width,
            color=colors[method],
            edgecolor="white",
            linewidth=0.25,
            alpha=1.0 if method == "FlowKG" else 0.88,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=7.4, labelpad=1.5, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=7.4, labelpad=1.5, fontweight="bold")
    ax.grid(
        True,
        linestyle="--",
        linewidth=0.55,
        color="#B0B0B0",
        alpha=0.72,
        zorder=0,
    )
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", width=0.75, length=2.2, pad=1.5, labelsize=6.7)

for ax in (axes[0, 0], axes[1, 0]):
    ax.set_ylim(0.18, 0.435)
    ax.set_yticks([0.20, 0.25, 0.30, 0.35, 0.40])

for ax in (axes[0, 1], axes[1, 1]):
    ax.set_ylim(0.13, 0.27)
    ax.set_yticks([0.14, 0.18, 0.22, 0.26])

legend_handles = [Patch(color=colors[method], label=method) for method in methods]
fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.55, 0.975),
    ncol=5,
    frameon=False,
    columnspacing=0.55,
    handlelength=0.95,
    handleheight=0.65,
    handletextpad=0.25,
    prop={"size": 6.7, "weight": "bold"},
)

fig.text(
    0.5,
    0.485,
    "(a) Performance w.r.t. cold-start user groups",
    ha="center",
    va="center",
    fontsize=8.0,
    fontweight="bold",
)
fig.text(
    0.5,
    0.032,
    "(b) Performance w.r.t. sparse item groups",
    ha="center",
    va="center",
    fontsize=8.0,
    fontweight="bold",
)

fig.subplots_adjust(
    left=0.115,
    right=0.985,
    top=0.905,
    bottom=0.150,
    wspace=0.34,
    hspace=0.55,
)

output_dir = Path(__file__).resolve().parent
fig.savefig(output_dir / "lastfm_sparsity_metrics.pdf", bbox_inches="tight")
fig.savefig(
    output_dir / "lastfm_sparsity_metrics.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

print(f"Saved to {output_dir / 'lastfm_sparsity_metrics.pdf'}")
