import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import math

# --- 与其他 hyperparameter_* 脚本保持一致的风格 ---
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


def plot_steps_sensitivity_dual_axis_mind():
    """MIND：Steps 对 Recall/NDCG 的影响（双轴）。

    数据来自你截图（Max Recall / Max NDCG）：
    step1: R=0.0945, N=0.0613
    step2: R=0.0945, N=0.0613
    step3: R=0.0945, N=0.0613
    step4: R=0.0945, N=0.0612
    step5: R=0.0945, N=0.0612
    """

    # step -> (recall@20, ndcg@20)
    data = {
        1: (0.0945, 0.0613),
        2: (0.0945, 0.0613),
        3: (0.0945, 0.0613),
        4: (0.0945, 0.0612),
        5: (0.0945, 0.0612),
    }

    steps = sorted(data.keys())

    recall = [float(data[s][0]) for s in steps]
    ndcg = [float(data[s][1]) for s in steps]

    # 让 x 轴等距显示（类别轴）
    x_pos = np.arange(len(steps))

    # 配色：橙色柱 + 蓝色虚线X标记折线（与现有图一致）
    color_ndcg = '#F07F2F'    # 橙色（柱）
    color_recall = '#3A66B7'  # 蓝色（折线）

    fig, ax_left = plt.subplots(1, 1)
    ax_right = ax_left.twinx()

    # twinx 场景层级：折线在柱子上
    ax_right.set_zorder(1)
    ax_left.set_zorder(2)
    ax_left.patch.set_visible(False)

    bar_width = 0.48 if len(steps) >= 5 else 0.36
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
        markersize=15,
        markeredgewidth=3.2,
        linestyle=(0, (4.5, 2.2)),
        linewidth=3.2,
        dash_capstyle='round',
        color=color_recall,
        label='R@20',
        zorder=4,
    )

    ax_left.set_xlabel('')
    ax_left.set_ylabel('')
    ax_right.set_ylabel('')

    ax_left.set_xticks(x_pos)
    ax_left.set_xticklabels([str(s) for s in steps], fontweight='semibold', fontsize=24)

    y_pad_ratio = 1

    def set_nice_yticks(
        ax,
        values,
        tick_target: int = 5,
        pad_ratio: float = 0.28,
        step_base: float = 0.0001,
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

    # MIND 数值更小，这里 step_base 调得更细一些，避免刻度过稀
    set_nice_yticks(ax_left, recall, tick_target=5, pad_ratio=y_pad_ratio, step_base=0.0001)
    set_nice_yticks(ax_right, ndcg, tick_target=5, pad_ratio=y_pad_ratio, step_base=0.0001)

    # 对齐双轴网格：用左轴刻度映射到右轴画水平虚线
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

    legend_dx = 0.01
    legend_dy = 0.01
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

    ax_left.set_title('MIND', fontweight='semibold', fontsize=28, pad=12)

    plt.tight_layout()

    plt.savefig('hyperparameter_steps_recall_ndcg_mind.png', dpi=300, bbox_inches='tight')
    plt.savefig('hyperparameter_steps_recall_ndcg_mind.pdf', dpi=300, bbox_inches='tight')
    print('File saved: hyperparameter_steps_recall_ndcg_mind.png/.pdf')

    plt.show()


if __name__ == '__main__':
    plot_steps_sensitivity_dual_axis_mind()
