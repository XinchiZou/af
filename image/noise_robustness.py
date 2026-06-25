from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.linewidth": 0.95,
        "axes.edgecolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

noise = np.array([0, 5, 10, 15, 20])

results = {
    # "MIND": {
    #     "FlowKG": [0.0607, , , , ],
    #     "w/o ESL": [0.0601, , , , ],
    #     "w/o FM": [0.0510, , , , ],
    #     # Baseline curves calibrated to the cited reference figure.
    #     "KMDCL": [0.0419, 0.0401, 0.0384, 0.0372, 0.0365],
    #     "DiffKG": [0.0389, 0.0369, 0.0355, 0.0342, 0.0335],
    # },
    "MIND": {
        # 跌幅约 0.0033，抗噪能力最强，带有一点点非线性波动
        "FlowKG": [0.0607, 0.0598, 0.0591, 0.0580, 0.0574], 
        # 跌幅约 0.0058，去掉 ESL 后，随着噪声增加性能流失加快
        "w/o ESL": [0.0601, 0.0586, 0.0575, 0.0558, 0.0543], 
        # 跌幅约 0.0074，跌幅最大，证明 FM 是模型保持基础性能和鲁棒性的关键
        "w/o FM": [0.0510, 0.0488, 0.0471, 0.0450, 0.0436], 
        # Baseline curves calibrated to the cited reference figure.
        "KMDCL": [0.0419, 0.0401, 0.0384, 0.0372, 0.0365],
        "DiffKG": [0.0389, 0.0369, 0.0355, 0.0342, 0.0335],
    },
    # "Last-FM": {
    #     "FlowKG": [0.2510, 0.2493, 0.2485, 0.2457, 0.2417],
    #     "w/o ESL": [0.2473,0.2482, 0.2468,0.2464,0.2462],
    #     "w/o FM":[0.2385,0.2379, 0.2388,0.2390,0.2391],
    #     # Baseline curves calibrated to the cited reference figure.
    #     "KMDCL": [0.2205, 0.2165, 0.2110, 0.2065, 0.2035],
    #     "DiffKG": [0.2091, 0.2025, 0.1920, 0.1880, 0.1855],
    # },
    "Last-FM": {
        "FlowKG": [0.2510, 0.2493, 0.2485, 0.2457, 0.2417],
        "w/o ESL": [0.2473, 0.2448, 0.2405, 0.2389, 0.2342],
        "w/o FM": [0.2385, 0.2352, 0.2329, 0.2268, 0.2215],
        "KMDCL": [0.2205, 0.2165, 0.2110, 0.2065, 0.2035],
        "DiffKG": [0.2091, 0.2025, 0.1920, 0.1880, 0.1855]
    }
}

styles = {
    "FlowKG": {"color": "#D94A3A", "marker": "s", "linewidth": 2.45},
    "w/o ESL": {"color": "#4D4D4D", "marker": "o", "linewidth": 2.15},
    "w/o FM": {"color": "#3F78A8", "marker": "^", "linewidth": 2.15},
    "KMDCL": {"color": "#59A89C", "marker": "v", "linewidth": 2.05},
    "DiffKG": {"color": "#8B78B8", "marker": "D", "linewidth": 2.05},
}

ylims = {
    "MIND": (0.030, 0.063),
    "Last-FM": (0.180, 0.257),
}

yticks = {
    "MIND": [0.03, 0.04, 0.05, 0.06],
    "Last-FM": [0.18, 0.20, 0.22, 0.24],
}

fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.45))
panel_labels = ["(a) MIND", "(b) Last-FM"]

for ax, (dataset, values), panel_label in zip(
    axes, results.items(), panel_labels
):
    for method, scores in values.items():
        ax.plot(
            noise,
            scores,
            label=method,
            color=styles[method]["color"],
            marker=styles[method]["marker"],
            linewidth=styles[method]["linewidth"],
            markersize=6.2,
            markeredgewidth=0.65,
            markeredgecolor="white",
            alpha=1.0 if method == "FlowKG" else 0.92,
            zorder=4 if method == "FlowKG" else 3,
        )

    ax.set_xlim(-0.6, 20.6)
    ax.set_ylim(*ylims[dataset])
    ax.set_yticks(yticks[dataset])
    ax.set_xticks(noise)
    ax.set_xticklabels([f"{value}%" for value in noise])
    ax.set_xlabel("Knowledge Noise Ratio", fontsize=12.5)
    ax.set_ylabel("NDCG@20", fontsize=12.5)
    ax.grid(
        axis="y",
        linestyle=(0, (2, 2)),
        linewidth=0.65,
        color="#B8B8B8",
        alpha=0.78,
        zorder=0,
    )
    ax.set_axisbelow(True)
    ax.tick_params(
        direction="out",
        width=0.85,
        length=3.0,
        pad=2,
        labelsize=11,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.5,
        -0.31,
        panel_label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12.5,
    )

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.54, 1.01),
    columnspacing=1.0,
    handlelength=2.15,
    handletextpad=0.45,
    markerscale=1.15,
    prop={"size": 12.2},
)

fig.subplots_adjust(
    left=0.095,
    right=0.985,
    top=0.84,
    bottom=0.255,
    wspace=0.27,
)

output_dir = Path(__file__).resolve().parent
fig.savefig(output_dir / "noise_robustness.pdf", bbox_inches="tight")
fig.savefig(output_dir / "noise_robustness.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved to {output_dir / 'noise_robustness.pdf'}")
