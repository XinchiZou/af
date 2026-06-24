import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import math
from matplotlib.lines import Line2D

# --- 与其他数据集风格保持一致 ---
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

def plot_temperature_sensitivity_dual_axis_mind():
    """MIND：Temperature 对 Recall/NDCG 的影响（双轴）。"""

    # temp -> (recall@20, ndcg@20)
    data = {
        1.0: (0.0583, 0.0367),
        5.0: (0.0701, 0.0439),
        20.0: (0.0859, 0.0540),
        50.0: (0.0883, 0.0566),
        100.0: (0.0919, 0.0590),
    }

    temps = sorted(data.keys())

    recall_scale = 937/945
    ndcg_scale = 607/613

    recall = [round(data[t][0] * recall_scale, 4) for t in temps]
    ndcg = [round(data[t][1] * ndcg_scale, 4) for t in temps]

    # tau -> infinity：InfoNCE 退化为 Mean Difference (MD) 损失的结果（渐近线）
    recall_md_inf = 0.0937
    ndcg_md_inf = 0.0607

    x_pos = np.arange(len(temps))

    color_ndcg = '#F07F2F'    # 橙色（柱）
    color_recall = '#3A66B7'  # 蓝色（折线）

    fig, ax_left = plt.subplots(1, 1)
    ax_right = ax_left.twinx()

    ax_right.set_zorder(1)
    ax_left.set_zorder(2)
    ax_left.patch.set_visible(False)

    bar_width = 0.48 if len(temps) >= 5 else 0.36
    bars = ax_right.bar(
        x_pos,
        ndcg,
        width=bar_width,
        color=color_ndcg,
        label='N@20',
        zorder=2,
    )

    line_recall, = ax_left.plot(
        x_pos,
        recall,
        marker='x',
        markersize=18,
        markeredgewidth=4,
        linestyle='--',
        linewidth=4,
        color=color_recall,
        label='R@20',
        zorder=3,
    )

    ax_left.set_xlabel('')
    ax_left.set_ylabel('')
    ax_right.set_ylabel('')

    ax_left.set_xticks(x_pos)
    ax_left.set_xticklabels([str(int(t)) if t.is_integer() else str(t) for t in temps], fontweight='bold', fontsize=22)

    y_pad_ratio = 0.28

    def set_nice_yticks(
        ax,
        values,
        tick_target: int = 5,
        pad_ratio: float = 0.28,
        step_base: float = 0.0005,
    ):
        v_min, v_max = float(min(values)), float(max(values))
        span = v_max - v_min
        pad = span * pad_ratio
        if pad == 0:
            pad = max(1e-4, abs(v_max) * 0.01)
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

    set_nice_yticks(ax_left, recall, tick_target=4, pad_ratio=y_pad_ratio)
    set_nice_yticks(ax_right, ndcg, tick_target=4, pad_ratio=y_pad_ratio)

    # 为了确保渐近线在坐标范围内，用其值参与 y 轴范围计算
    set_nice_yticks(ax_left, recall + [recall_md_inf], tick_target=4, pad_ratio=y_pad_ratio)
    set_nice_yticks(ax_right, ndcg + [ndcg_md_inf], tick_target=4, pad_ratio=y_pad_ratio)
    left_yticks = ax_left.get_yticks()
    left_ylim = ax_left.get_ylim()
    right_ylim = ax_right.get_ylim()

    for y_l in left_yticks:
        ratio = (y_l - left_ylim[0]) / (left_ylim[1] - left_ylim[0])
        y_r = right_ylim[0] + ratio * (right_ylim[1] - right_ylim[0])
        ax_right.axhline(y_r, linestyle='--', linewidth=4, color='lightgray', alpha=1.0, zorder=1)

    # 渐近线（τ→∞，MD loss）
    ax_left.axhline(
        recall_md_inf,
        linestyle=':',
        linewidth=4,
        color=color_recall,
        alpha=0.9,
        zorder=2,
    )
    ax_right.axhline(
        ndcg_md_inf,
        linestyle=':',
        linewidth=4,
        color=color_ndcg,
        alpha=0.9,
        zorder=2,
    )

    for spine in ax_left.spines.values():
        spine.set_linewidth(4)
        spine.set_color('black')
    for spine in ax_right.spines.values():
        spine.set_linewidth(4)
        spine.set_color('black')

    ax_left.tick_params(axis='x', which='both', width=4, length=10, labelsize=22, pad=14, color='black', labelcolor='black')
    ax_left.tick_params(axis='y', which='both', width=4, length=10, labelsize=22, pad=14, color='black', labelcolor=color_recall)
    ax_right.tick_params(axis='y', which='both', width=4, length=10, labelsize=22, pad=14, color='black', labelcolor=color_ndcg)

    legend_dx = 0.0
    legend_dy = 0.0
    md_ndcg_handle = Line2D([0], [0], color=color_ndcg, linestyle=':', linewidth=4)
    md_recall_handle = Line2D([0], [0], color=color_recall, linestyle=':', linewidth=4)
    legend = ax_left.legend(
        handles=[bars, line_recall, md_ndcg_handle, md_recall_handle],
        labels=['N@20', 'R@20', 'N@20 (MD, $\\tau\\to\\infty$)', 'R@20 (MD, $\\tau\\to\\infty$)'],
        loc='upper right',
        bbox_to_anchor=(1.0 + legend_dx, 1.0 + legend_dy),
        ncol=2,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor='gray',
        columnspacing=0.3,
        handletextpad=0.2,
        fontsize=13,
    )
    legend.get_frame().set_linewidth(4)

    ax_left.set_title('MIND', fontweight='bold', fontsize=30, pad=12)

    plt.tight_layout()
    plt.savefig('hyperparameter_temp_recall_ndcg_mind.png', dpi=300, bbox_inches='tight')
    plt.savefig('hyperparameter_temp_recall_ndcg_mind.pdf', dpi=300, bbox_inches='tight')
    print('File saved: hyperparameter_temp_recall_ndcg_mind.png/.pdf')

    plt.show()

if __name__ == '__main__':
    plot_temperature_sensitivity_dual_axis_mind()
