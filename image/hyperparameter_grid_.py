from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "font.size": 11,
    "axes.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
})


DATA = {
    "MIND": {
        "rho": {
            "x": [0.2, 0.4, 0.6, 0.8],
            "recall": [0.0937, 0.0933, 0.0936, 0.0937],
            "ndcg": [0.0604, 0.0605, 0.0607, 0.0606],
        },
        "N": {
            "x": [1, 2, 3, 4, 5],
            "recall": [0.0945, 0.0945, 0.0945, 0.0945, 0.0945],
            "ndcg": [0.0613, 0.0613, 0.0613, 0.0612, 0.0612],
        },
        "tau": {
            "x": [1, 5, 20, 50, 100],
            "recall": [0.0578, 0.0695, 0.0852, 0.0876, 0.0911],
            "ndcg": [0.0363, 0.0435, 0.0535, 0.0560, 0.0584],
        },
        "tstart": {
            "x": [0.5, 0.6, 0.7, 0.8, 0.9],
            "recall": [0.0914, 0.0928, 0.0937, 0.0934, 0.0925],
            "ndcg": [0.0589, 0.0600, 0.0607, 0.0605, 0.0598],
        },
    },
    "ML-1M": {
        "rho": {
            "x": [0.2, 0.4, 0.6, 0.8],
            "recall": [0.3443, 0.3506, 0.3499, 0.3517],
            "ndcg": [0.3218, 0.3276, 0.3274, 0.3283],
        },
        "N": {
            "x": [1, 2, 3, 4, 5],
            "recall": [0.3515, 0.3517, 0.3515, 0.3516, 0.3515],
            "ndcg": [0.3281, 0.3283, 0.3282, 0.3282, 0.3282],
        },
        "tau": {
            "x": [0.1, 0.4, 0.7, 1.0],
            "recall": [0.2841, 0.3197, 0.3517, 0.3361],
            "ndcg": [0.2642, 0.2937, 0.3283, 0.3151],
        },
        "tstart": {
            "x": [0.5, 0.6, 0.7, 0.8, 0.9],
            "recall": [0.3484, 0.3502, 0.3511, 0.3517, 0.3506],
            "ndcg": [0.3246, 0.3265, 0.3275, 0.3283, 0.3270],
        },
    },
    "Last-FM": {
        "rho": {
            "x": [0.2, 0.4, 0.6, 0.8],
            "recall": [0.4058, 0.4085, 0.4106, 0.4064],
            "ndcg": [0.2483, 0.2499, 0.2510, 0.2481],
        },
        "N": {
            "x": [1, 2, 3, 4, 5],
            "recall": [0.4106, 0.4106, 0.4103, 0.4103, 0.4103],
            "ndcg": [0.2504, 0.2510, 0.2509, 0.2509, 0.2509],
        },
        "tau": {
            "x": [0.1, 0.4, 0.7, 1.0],
            "recall": [0.3777, 0.3986, 0.4100, 0.4073],
            "ndcg": [0.2256, 0.2454, 0.2499, 0.2462],
        },
        "tstart": {
            "x": [0.5, 0.6, 0.7, 0.8, 0.9],
            "recall": [0.4087, 0.4106, 0.4101, 0.4089, 0.4075],
            "ndcg": [0.2496, 0.2506, 0.2509, 0.2498, 0.2488],
        },
    },
    "Yelp": {
        "rho": {
            "x": [0.2, 0.4, 0.6, 0.8],
            "recall": [0.1221, 0.1232, 0.1227, 0.1228],
            "ndcg": [0.0686, 0.0691, 0.0688, 0.0688],
        },
        "N": {
            "x": [1, 2, 3, 4, 5],
            "recall": [0.1229, 0.1230, 0.1230, 0.1232, 0.1231],
            "ndcg": [0.0690, 0.0691, 0.0690, 0.0691, 0.0691],
        },
        "tau": {
            "x": [0.1, 0.4, 0.7, 1.0],
            "recall": [0.0740, 0.1003, 0.1181, 0.1232],
            "ndcg": [0.0418, 0.0569, 0.0666, 0.0691],
        },
        "tstart": {
            "x": [0.5, 0.6, 0.7, 0.8, 0.9],
            "recall": [0.1209, 0.1217, 0.1226, 0.1232, 0.1229],
            "ndcg": [0.0676, 0.0682, 0.0687, 0.0689, 0.0691],
        },
    },
}


PARAMS = [
    ("rho", r"Impact of $\rho$"),
    ("N", r"Impact of $N$"),
    ("tau", r"Impact of $\tau$"),
    ("tstart", r"Impact of $t_{\mathrm{start}}$"),
]


def _set_nice_yticks(
    ax,
    values,
    *,
    tick_target: int = 5,
    lower_pad_ratio: float = 0.18,
    upper_pad_ratio: float = 0.25,
    step_base: float = 0.0001,
) -> None:
    v_min, v_max = float(min(values)), float(max(values))
    span = v_max - v_min
    lower_pad = span * lower_pad_ratio
    upper_pad = span * upper_pad_ratio
    if lower_pad == 0:
        lower_pad = max(1e-4, abs(v_max) * 0.01)
    if upper_pad == 0:
        upper_pad = max(1e-4, abs(v_max) * 0.01)

    y0, y1 = v_min - lower_pad, v_max + upper_pad
    raw_step = (y1 - y0) / max(1, tick_target - 1)
    if raw_step <= 0 or not np.isfinite(raw_step):
        step = step_base
    else:
        step = max(math.ceil(raw_step / step_base) * step_base, step_base)

    start = math.floor(y0 / step) * step
    end = math.ceil(y1 / step) * step
    ticks = np.arange(start, end + step * 0.5, step)
    while len(ticks) < 4 and step > step_base:
        step = step / 2.0
        start = math.floor(y0 / step) * step
        end = math.ceil(y1 / step) * step
        ticks = np.arange(start, end + step * 0.5, step)
    ticks = np.round(ticks / step_base) * step_base

    ax.set_ylim(start, end)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))


def _plot_panel(ax_left, spec, *, title: str, show_legend: bool, compact: bool) -> None:
    x_labels = spec["x"]
    x_pos = np.arange(len(x_labels))
    recall = spec["recall"]
    ndcg = spec["ndcg"]

    color_ndcg = "#F07F2F"
    color_recall = "#3A66B7"

    ax_right = ax_left.twinx()
    ax_left.set_zorder(1)
    ax_right.set_zorder(2)
    ax_right.patch.set_visible(False)

    bar_width = 0.52 if len(x_labels) >= 5 else 0.46
    bars = ax_left.bar(
        x_pos,
        ndcg,
        width=bar_width,
        color=color_ndcg,
        label="N@20",
        zorder=2,
    )
    line_recall, = ax_right.plot(
        x_pos,
        recall,
        marker="x",
        markersize=6.0 if compact else 8.0,
        markeredgewidth=1.6 if compact else 2.2,
        linestyle=(0, (4.5, 2.2)),
        linewidth=1.9 if compact else 2.5,
        dash_capstyle="round",
        color=color_recall,
        label="R@20",
        zorder=5,
    )

    ax_left.set_xticks(x_pos)
    ax_left.set_xticklabels([str(x) for x in x_labels], fontweight="bold")

    is_main_n_panel = (not compact) and ("Impact of $N$" in title)
    upper_pad = 0.25 if is_main_n_panel else (0.45 if compact else 0.58)

    _set_nice_yticks(
        ax_left,
        ndcg,
        tick_target=4 if compact else 5,
        upper_pad_ratio=upper_pad,
    )
    _set_nice_yticks(
        ax_right,
        recall,
        tick_target=4 if compact else 5,
        upper_pad_ratio=upper_pad,
    )

    left_ticks = ax_left.get_yticks()
    left_ylim = ax_left.get_ylim()
    right_ylim = ax_right.get_ylim()
    for y_l in left_ticks:
        ratio = (y_l - left_ylim[0]) / (left_ylim[1] - left_ylim[0])
        y_r = right_ylim[0] + ratio * (right_ylim[1] - right_ylim[0])
        ax_right.axhline(
            y_r,
            linestyle=(0, (3.2, 2.4)),
            linewidth=1.1 if compact else 1.8,
            color="#D7D7D7",
            alpha=1.0,
            zorder=1,
        )

    spine_width = 1.5 if compact else 2.0
    tick_width = 1.5 if compact else 2.0
    for spine in ax_left.spines.values():
        spine.set_linewidth(spine_width)
        spine.set_color("black")
    for spine in ax_right.spines.values():
        spine.set_linewidth(spine_width)
        spine.set_color("black")

    tick_size = 5.2 if compact else 9.0
    x_tick_size = 6.0 if compact else 10.5
    ax_left.tick_params(axis="x", width=tick_width, length=4.0, pad=4, labelsize=x_tick_size, color="black", labelcolor="black")
    ax_left.tick_params(axis="y", width=tick_width, length=4.0, pad=3, labelsize=tick_size, color="black", labelcolor=color_ndcg)
    ax_right.tick_params(axis="y", width=tick_width, length=4.0, pad=3, labelsize=tick_size, color="black", labelcolor=color_recall)

    if show_legend or compact:
        legend = ax_right.legend(
            handles=[bars, line_recall],
            labels=["N@20", "R@20"],
            loc="upper right",
            bbox_to_anchor=(0.98, 1.02 if compact else (0.98 if is_main_n_panel else 1.01)),
            ncol=2,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            edgecolor="gray",
            columnspacing=0.25 if compact else 0.35,
            handletextpad=0.20 if compact else 0.25,
            borderpad=0.18 if compact else 0.25,
            handlelength=1.6 if compact else 1.8,
            fontsize=8.0 if compact else None,
        )
        legend.get_frame().set_linewidth(1.4 if compact else 1.8)
        for text in legend.get_texts():
            text.set_fontweight("bold")

    ax_left.set_title(title, fontweight="bold", pad=5)


def compose_main_grid(img_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 4.6))
    specs = DATA["ML-1M"]
    for idx, (ax, (key, title)) in enumerate(zip(axes.flat, PARAMS)):
        _plot_panel(
            ax,
            specs[key],
            title=f"({chr(ord('a') + idx)}) {title}",
            show_legend=True,
            compact=False,
        )

    fig.subplots_adjust(left=0.10, right=0.90, top=0.92, bottom=0.10, wspace=0.58, hspace=0.42)
    out_path = img_dir / "hyperparameter_grid.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def compose_all_grid(img_dir: Path) -> None:
    fig = plt.figure(figsize=(8.9, 5.9))
    outer = fig.add_gridspec(2, 2, wspace=0.20, hspace=0.26)
    datasets = ["MIND", "ML-1M", "Last-FM", "Yelp"]

    for idx, (key, title) in enumerate(PARAMS):
        outer_cell = outer[idx // 2, idx % 2]
        title_ax = fig.add_subplot(outer_cell)
        title_ax.set_title(
            f"({chr(ord('a') + idx)}) {title}",
            fontsize=13,
            fontfamily="Times New Roman",
            fontweight="bold",
            pad=14,
        )
        title_ax.axis("off")

        inner = outer_cell.subgridspec(2, 2, wspace=0.50, hspace=0.38)
        for d_idx, dataset in enumerate(datasets):
            ax = fig.add_subplot(inner[d_idx // 2, d_idx % 2])
            _plot_panel(
                ax,
                DATA[dataset][key],
                title=dataset,
                show_legend=False,
                compact=True,
            )
            ax.title.set_fontsize(7.5)
            ax.title.set_fontfamily("Times New Roman")

    fig.subplots_adjust(left=0.065, right=0.935, top=0.95, bottom=0.055)

    out_path = img_dir / "hyperparameter_grid_all.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    img_dir = Path(__file__).resolve().parent
    compose_main_grid(img_dir)
    compose_all_grid(img_dir)
    print("Saved: hyperparameter_grid.pdf/png and hyperparameter_grid_all.pdf/png")


if __name__ == "__main__":
    main()
