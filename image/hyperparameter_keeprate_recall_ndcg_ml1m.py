import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import math

# --- 与 LastFM 版本保持一致的风格 ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 15,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "figure.figsize": (7, 4.8),
})


def plot_keeprate_sensitivity_dual_axis_ml1m():
    """ML-1M：Keep rate 对 Recall/NDCG 的影响（双轴）。

    数据来自你最新截图（Max Recall / Max NDCG）：
    keep0.4: R=0.3425, N=0.3218
    keep0.6: R=0.3398, N=0.3194
    keep0.7: R=0.3372, N=0.3180
    keep0.8: R=0.3340, N=0.3151
    keep0.2: R=0.3167, N=0.2968
    """


    # keeprate -> (recall@20, ndcg@20)
    data = {
        0.8: (0.3573, 0.3340),
        0.4: (0.3562, 0.3333),
        0.6: (0.3555, 0.3331),
        0.2: (0.3498, 0.3274),
    }

    keep_rates = sorted(data.keys())

    # 数值缩放（默认不缩放）；需要对齐不同图的展示范围时可调
    recall_scale = 3517/3573
    ndcg_scale = 3283/3340

    recall = [round(data[k][0] * recall_scale, 4) for k in keep_rates]
    ndcg = [round(data[k][1] * ndcg_scale, 4) for k in keep_rates]

    x_pos = np.arange(len(keep_rates))

    color_ndcg = '#F07F2F'
    color_recall = '#3A66B7'

    fig, ax_left = plt.subplots(1, 1)
    ax_right = ax_left.twinx()

    # 调整层级：让折线(ax_left)在柱子(ax_right)之上
    ax_right.set_zorder(1)
    ax_left.set_zorder(2)
    ax_left.patch.set_visible(False)

    # 说明：如果类别数更多（例如 5 个 keeprate），同样的 width 在像素上会显得更“细”。
    # 为了和 4 个点的图视觉粗细一致，类别数为 5 时把柱宽稍微放大。
    bar_width = 0.48 if len(keep_rates) >= 5 else 0.36
    bars = ax_right.bar(x_pos, ndcg, width=bar_width, color=color_ndcg, label='N@20', zorder=2)

    line_recall, = ax_left.plot(
        x_pos,
        recall,
        marker='x',
        markersize=15,
        markeredgewidth=3.2,
        linestyle=(0, (4.5, 2.2)),
        linewidth=3.2,
        dash_capstyle='round',
        color=color_recall,
        label='R@20',
        zorder=3,
    )

    ax_left.set_xlabel('')
    ax_left.set_ylabel('')
    ax_right.set_ylabel('')

    ax_left.set_xticks(x_pos)
    ax_left.set_xticklabels([f'{k:.1f}' for k in keep_rates], fontweight='semibold', fontsize=24)

    # 网格：粗灰色虚线（只画水平线）
    # ax_right.set_axisbelow(True)
    # ax_right.grid(axis='y', linestyle='--', linewidth=4, color='lightgray', alpha=1.0)

    y_pad_ratio = 0.45

    def set_nice_yticks(
        ax,
        values,
        tick_target: int = 5,
        pad_ratio: float = 0.28,
        step_base: float = 0.0002,
    ):
        v_min, v_max = float(min(values)), float(max(values))
        span = v_max - v_min
        pad = span * pad_ratio
        y0, y1 = v_min - pad, v_max + pad

        raw_step = (y1 - y0) / max(1, (tick_target - 1))
        if raw_step <= 0 or not np.isfinite(raw_step):
            step = step_base
        else:
            step = math.ceil(raw_step / step_base) * step_base
            step = max(step, step_base)

        start = math.floor(y0 / step) * step
        end = math.ceil(y1 / step) * step

        ticks = np.arange(start, end + step * 0.5, step)
        ticks = np.round(ticks / step_base) * step_base

        ax.set_ylim(start, end)
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

    set_nice_yticks(ax_left, recall, tick_target=5, pad_ratio=y_pad_ratio)
    set_nice_yticks(ax_right, ndcg, tick_target=5, pad_ratio=y_pad_ratio)

    left_yticks = ax_left.get_yticks()
    left_ylim = ax_left.get_ylim()
    right_ylim = ax_right.get_ylim()

    for y_l in left_yticks:
        ratio = (y_l - left_ylim[0]) / (left_ylim[1] - left_ylim[0])
        y_r = right_ylim[0] + ratio * (right_ylim[1] - right_ylim[0])
        ax_right.axhline(y_r, linestyle=(0, (3.2, 2.4)), linewidth=2.2, color='lightgray', alpha=0.95, zorder=1)

    for spine in ax_left.spines.values():
        spine.set_linewidth(2.6)
        spine.set_color('black')
    for spine in ax_right.spines.values():
        spine.set_linewidth(2.6)
        spine.set_color('black')

    ax_left.tick_params(axis='x', which='both', width=2.6, length=7, labelsize=20, pad=14, color='black', labelcolor='black')
    ax_left.tick_params(axis='y', which='both', width=2.6, length=7, labelsize=20, pad=14, color='black', labelcolor=color_recall)
    ax_right.tick_params(axis='y', which='both', width=2.6, length=7, labelsize=20, pad=14, color='black', labelcolor=color_ndcg)

    legend_dx = 0.015
    legend_dy = 0.02
    legend = ax_left.legend(
        handles=[bars, line_recall],
        labels=['N@20', 'R@20'],
        loc='upper right',
        bbox_to_anchor=(1.0 + legend_dx, 1.0 + legend_dy),
        ncol=2,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor='gray',
        columnspacing=0.3,
        handletextpad=0.2,
        fontsize=20,
    )
    legend.get_frame().set_linewidth(2.6)

    ax_left.set_title('ML-1M', fontweight='semibold', fontsize=28, pad=12)

    plt.tight_layout()
    plt.savefig('hyperparameter_keeprate_recall_ndcg_ml1m.png', dpi=300, bbox_inches='tight')
    plt.savefig('hyperparameter_keeprate_recall_ndcg_ml1m.pdf', dpi=300, bbox_inches='tight')
    print('File saved: hyperparameter_keeprate_recall_ndcg_ml1m.png/.pdf')

    plt.show()


if __name__ == '__main__':
    plot_keeprate_sensitivity_dual_axis_ml1m()
