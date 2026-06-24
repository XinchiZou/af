import matplotlib.pyplot as plt
import numpy as np
import math

from matplotlib import rcParams

# === 全局风格：与 figure_group.py 保持一致 ===
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 20

def plot_kg_noise_alleviation_relative_recall_and_ndcg():
    # === 1. 数据准备 ===
    datasets = ['MIND', 'ML-1M', 'Last-FM', 'Yelp']

    drop_ratios_recall = {
        'FlowKG':  [0.46, 0.33, 0.43, 0.49],
        'KMDCL':   [1.68, 0.96, 1.21, 2.20],
        'DiffKG':  [3.98, 2.47, 3.27, 4.85],
        'KGRec':   [5.80, 2.85, 4.95, 6.90],
        'LightKG': [6.55, 3.25, 5.40, 7.60],
    }

    # --- NDCG Drop Data ---
    drop_ratios_ndcg = {
        'FlowKG':  [0.64, 0.42, 0.60, 0.73],
        'KMDCL':   [1.24, 1.23, 1.39, 2.76],
        'DiffKG':  [3.90, 2.58, 3.20, 5.24],
        'KGRec':   [6.10, 3.15, 5.05, 8.20],
        'LightKG': [6.85, 3.60, 5.65, 8.95],
    }

    def to_relative_metric(drop_ratios: dict[str, list[float]]):
        return {model: [1 - (d / 100) for d in drops] for model, drops in drop_ratios.items()}

    relative_recall = to_relative_metric(drop_ratios_recall)
    relative_ndcg = to_relative_metric(drop_ratios_ndcg)

    # === 2. 绘图设置（对齐 figure_group.py 的风格） ===
    methods = ['FlowKG', 'KMDCL', 'DiffKG', 'KGRec', 'LightKG']

    # 设置 SIGIR 风格配色 (从深到浅，或冷暖对比)
    # FlowKG 用最醒目的颜色 (红色/深红)，其他用蓝色系/灰色系区分
    # colors = {
    #     'FlowKG':  '#FCB2AF',  # Red (Ours)
    #     'KMDCL':   '#9BDFDF',  # Orange (Strong Baseline)
    #     'DiffKG':  '#FFE2CE',  # Green
    #     'KGRec':   '#C4D8E9',  # Blue
    #     'LightKG': '#BEBCDF',  # Purple
    # }
    colors = {
        'FlowKG':  '#FF9B9B',  # Red (Ours)
        'KMDCL':   '#96C291',  # Orange (Strong Baseline)
        'DiffKG':  '#FFDBAA',  # Green
        'KGRec':   '#FFB7B7',  # Blue
        'LightKG': '#8CC0DE',  # Purple
    }


    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    def reorder_legend_row_major(handles, labels, ncol: int):
        """Reorder legend entries so they display left-to-right then top-to-bottom.

        Matplotlib fills multi-column legends in a column-major way by default.
        This converts an input order (assumed row-major) into the order needed
        to render as row-major.
        """

        if ncol <= 1:
            return handles, labels

        count = len(handles)
        if count == 0:
            return handles, labels

        nrows = int(math.ceil(count / ncol))
        new_handles = [None] * count
        new_labels = [None] * count

        for r in range(nrows):
            for c in range(ncol):
                row_major_index = r * ncol + c
                if row_major_index >= count:
                    continue

                column_major_index = c * nrows + r
                if column_major_index >= count:
                    continue

                new_handles[column_major_index] = handles[row_major_index]
                new_labels[column_major_index] = labels[row_major_index]

        filtered = [(h, l) for h, l in zip(new_handles, new_labels) if h is not None]
        if not filtered:
            return handles, labels

        out_handles, out_labels = zip(*filtered)
        return list(out_handles), list(out_labels)

    bar_width = 0.15
    x = np.arange(len(datasets))

    def plot_panel(ax, values_by_method, ylabel: str, show_legend: bool):
        ax.grid(axis='y', linestyle='--', linewidth=0.6, color='gray', alpha=0.5, zorder=0)
        for i, method in enumerate(methods):
            ax.bar(
                x + i * bar_width,
                values_by_method[method],
                width=bar_width,
                label=method,
                color=colors[method],
                edgecolor='black',
                linewidth=0.8,
                zorder=3,
            )

        ax.set_ylabel(ylabel, fontsize=20, fontweight='bold')
        ax.set_xticks(x + bar_width * ((len(methods) - 1) / 2))
        ax.set_xticklabels(datasets, fontsize=20, fontweight='bold')
        # sharey=True 时右图默认不显示 y 轴刻度标签，这里强制两边都显示
        ax.tick_params(axis='y', labelsize=20, labelleft=True)
        ax.set_ylim(0.85, 1.05)

        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            # Ensure legend reads left-to-right, then top-to-bottom.
            handles, labels = reorder_legend_row_major(handles, labels, ncol=3)
            ax.legend(
                handles,
                labels,
                loc='upper left',
                frameon=True,
                edgecolor='black',
                ncol=3,
                fontsize=20,
                columnspacing=1.0,
                handletextpad=0.5,
            )

    # 左：Recall；右：NDCG
    plot_panel(axes[0], relative_recall, 'Relative Recall', show_legend=True)
    plot_panel(axes[1], relative_ndcg, 'Relative NDCG', show_legend=True)

    fig.tight_layout(rect=[0, 0.10, 1, 1])

    # 子图标题放在下方：使用 axes 坐标系，确保 (a)/(b) 与子图位置一致（不会被 tight_layout 影响错位）
    labels = ['(a) Relative Recall', '(b) Relative NDCG']
    for ax, label in zip(axes, labels):
        ax.text(
            0.5,
            -0.22,
            label,
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontsize=28,
            clip_on=False,
        )
    plt.savefig('noise_robustness.png', dpi=300, bbox_inches='tight')
    plt.savefig('noise_robustness.pdf', dpi=300, bbox_inches='tight')
    print('File saved: noise_robustness.png')
    plt.show()


plot_kg_noise_alleviation_relative_recall_and_ndcg()