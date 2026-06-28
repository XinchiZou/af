import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = base_dir / path
    return path


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Draw latent-space scatter plots.")
    parser.add_argument("--npz", type=str, default="latest_scatter.npz")
    parser.add_argument("--out_file", type=str, default="scatter.png")
    parser.add_argument("--font", type=str, default="Times New Roman")
    parser.add_argument("--font_size", type=float, default=18.0)
    parser.add_argument("--title_size", type=float, default=24.0)
    parser.add_argument("--legend_size", type=float, default=17.0)
    parser.add_argument("--point_size", type=float, default=40.0)
    parser.add_argument("--alpha", type=float, default=0.70)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    npz_path = _resolve_path(args.npz, script_dir)
    out_path = _resolve_path(args.out_file, script_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = _load_npz(npz_path)
    enc_2d = data["encoder_2d"]
    flow_2d = data["flow_2d"]
    labels = data["labels"].astype(str)
    categories = data["categories"].astype(str).tolist()

    legend_name_map = {
        "sports": "Sports",
        "autos": "Autos",
        "foodanddrink": "Food & Drink",
        "food_and_drink": "Food & Drink",
        "food and drink": "Food & Drink",
    }
    preferred_order = ["sports", "autos", "foodanddrink"]
    if all(c in categories for c in preferred_order):
        categories = preferred_order

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [args.font, "Times New Roman", "Times", "DejaVu Serif"],
            "font.size": args.font_size,
            "axes.labelsize": args.font_size,
            "legend.fontsize": args.legend_size,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "#333333",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    palette = {
        "sports": "#2B83BA",
        "autos": "#A6C8E5",
        "foodanddrink": "#F28E2B",
    }
    fallback_cmap = plt.get_cmap("tab20")
    cat_colors = {
        cat: palette.get(cat, fallback_cmap(i % fallback_cmap.N))
        for i, cat in enumerate(categories)
    }

    def _set_padded_limits(ax, points: np.ndarray, pad_ratio: float = 0.045) -> None:
        x_min, x_max = float(points[:, 0].min()), float(points[:, 0].max())
        y_min, y_max = float(points[:, 1].min()), float(points[:, 1].max())
        x_pad = max((x_max - x_min) * pad_ratio, 1e-4)
        y_pad = max((y_max - y_min) * pad_ratio, 1e-4)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    def _plot_on_ax(ax, points: np.ndarray, subtitle: str) -> None:
        for cat in categories:
            idx = np.where(labels == cat)[0]
            pts = points[idx]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=args.point_size,
                alpha=args.alpha,
                color=cat_colors[cat],
                label=legend_name_map.get(cat, cat),
                edgecolors="white",
                linewidths=0.16,
                rasterized=True,
            )

        _set_padded_limits(ax, points)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.5,
            -0.035,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=args.title_size,
            fontweight="bold",
        )

        for spine in ax.spines.values():
            spine.set_linewidth(0.95)
            spine.set_color("#333333")

        legend = ax.legend(
            loc="upper left",
            frameon=True,
            framealpha=0.9,
            fancybox=False,
            edgecolor="#B8B8B8",
            borderaxespad=0.55,
            borderpad=0.42,
            labelspacing=0.62,
            handletextpad=0.42,
            markerscale=0.95,
            prop={"weight": "bold", "size": args.legend_size},
        )
        legend.get_frame().set_linewidth(0.75)

    fig, axes = plt.subplots(1, 2, figsize=(15.8, 5.55))
    _plot_on_ax(axes[0], enc_2d, "(a) Initial Semantic Embeddings")
    _plot_on_ax(axes[1], flow_2d, "(b) Flow-Refined Embeddings")

    plt.tight_layout()
    plt.subplots_adjust(top=0.985, bottom=0.13, wspace=0.055)

    print(f"Saving scatter plot to: {out_path}")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
