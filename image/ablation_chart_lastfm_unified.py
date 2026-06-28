from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 6.2,
        "axes.linewidth": 0.75,
        "axes.edgecolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

blocks = [
    (
        "Skeleton vs Full",
        ["Coupled\nSkeleton", "Decoupled\nBase", "Full\nModel"],
        [0.3477, 0.3578, 0.4106],
        [0.1947, 0.2029, 0.2509],
    ),
    (
        "Base + Single Module",
        ["Base+SL", "Base+FM", "Base+CL"],
        [0.3665, 0.3751, 0.3873],
        [0.2081, 0.2152, 0.2234],
    ),
    (
        "Full - Single Module",
        ["w/o SL", "w/o FM", "w/o CL"],
        [0.4073, 0.3891, 0.3862],
        [0.2473, 0.2385, 0.2200],
    ),
]

labels = []
recall = []
ndcg = []
x_positions = []
groups = []
current = 0.0
for group_name, group_labels, group_recall, group_ndcg in blocks:
    start = len(x_positions)
    for i, label in enumerate(group_labels):
        labels.append(label)
        recall.append(group_recall[i])
        ndcg.append(group_ndcg[i])
        x_positions.append(current + i)
    end = len(x_positions) - 1
    groups.append((group_name, x_positions[start], x_positions[end]))
    current += len(group_labels) + 0.18

x = np.array(x_positions)
recall = np.array(recall)
ndcg = np.array(ndcg)

colors = {
    "Recall@20": "#3F78A8",
    "NDCG@20": "#D6A84B",
}

fig, ax = plt.subplots(figsize=(3.55, 1.78))
ax2 = ax.twinx()
bar_w = 0.34

x_left = x[0] - 0.62
x_right = x[-1] + 0.62
group_bounds = [x_left]
for i in range(1, len(groups)):
    group_bounds.append((groups[i - 1][2] + groups[i][1]) / 2)
group_bounds.append(x_right)

for i in range(len(groups)):
    bg_color = "#FFFFFF" if i == 1 else "#F4F4F4"
    ax.axvspan(group_bounds[i], group_bounds[i + 1], color=bg_color, zorder=0)
    if i > 0:
        ax.axvline(group_bounds[i], color="#BDBDBD", linewidth=0.55, linestyle=(0, (2, 2)), zorder=1)

bars1 = ax.bar(
    x - bar_w / 2,
    recall,
    width=bar_w,
    color=colors["Recall@20"],
    edgecolor="#333333",
    linewidth=0.25,
    label="Recall@20",
    zorder=3,
)
bars2 = ax2.bar(
    x + bar_w / 2,
    ndcg,
    width=bar_w,
    color=colors["NDCG@20"],
    edgecolor="#333333",
    linewidth=0.25,
    label="NDCG@20",
    zorder=3,
)

ax.set_xlim(x_left, x_right)
ax.set_ylim(0.335, 0.420)
ax2.set_ylim(0.185, 0.258)
ax.set_yticks([0.34, 0.38, 0.42])
ax2.set_yticks([0.19, 0.22, 0.25])
ax.set_ylabel("Recall@20", fontsize=6.4, labelpad=1.2, fontweight="bold")
ax2.set_ylabel("NDCG@20", fontsize=6.4, labelpad=1.2, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(
    labels,
    fontsize=5.5,
    rotation=27,
    ha="right",
    rotation_mode="anchor",
    fontweight="bold",
)
ax.tick_params(direction="out", width=0.65, length=2.0, pad=1.2, labelsize=5.6)
ax2.tick_params(direction="out", width=0.65, length=2.0, pad=1.2, labelsize=5.6)
ax.grid(axis="y", linestyle=(0, (2, 2)), linewidth=0.45, color="#B8B8B8", alpha=0.65, zorder=0)
ax.set_axisbelow(True)

for group_name, start, end in groups:
    ax.text(
        (start + end) / 2,
        0.4185,
        group_name,
        ha="center",
        va="top",
        fontsize=5.4,
        fontweight="bold",
        color="#666666",
        zorder=5,
    )

for spine in ("top",):
    ax.spines[spine].set_visible(False)
    ax2.spines[spine].set_visible(False)

handles = [bars1, bars2]
labels_legend = ["Recall@20", "NDCG@20"]
fig.legend(
    handles,
    labels_legend,
    loc="upper center",
    bbox_to_anchor=(0.52, 1.01),
    ncol=2,
    frameon=False,
    columnspacing=1.25,
    handlelength=1.2,
    handletextpad=0.35,
    prop={"size": 6.2, "weight": "bold"},
)

fig.subplots_adjust(left=0.115, right=0.885, top=0.84, bottom=0.31)

output_dir = Path(__file__).resolve().parent
fig.savefig(output_dir / "ablation_chart_lastfm_unified.pdf", bbox_inches="tight", pad_inches=0.015)
fig.savefig(output_dir / "ablation_chart_lastfm_unified.png", dpi=300, bbox_inches="tight", pad_inches=0.015)
plt.close(fig)

print(f"Saved to {output_dir / 'ablation_chart_lastfm_unified.pdf'}")
