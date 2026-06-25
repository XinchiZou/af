import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import math

# --- 与 hyperparameter.py / hyperparameter_steps_recall_ndcg.py 保持一致的风格 ---
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


def plot_steps_sensitivity_dual_axis_yelp():
    """Yelp Steps 对 Recall/NDCG 的影响（双轴）。

    数据来自你最新截图：
    step1: R=0.1166, N=0.0653
    step2: R=0.1170, N=0.0653
    step3: R=0.1164, N=0.0653
    step4: R=0.1170, N=0.0655
    """

    # step -> (recall@20, ndcg@20)
    data = {
        1: (0.1244, 0.0696),
        2: (0.1245, 0.0697),
        3: (0.1245, 0.0696),
        4: (0.1247, 0.0697),
        5: (0.1246, 0.0697),
    }

    steps = sorted(data.keys())

    # 数值缩放（默认不缩放）；需要对齐不同图的展示范围时可调
    recall_scale = 1232/1247
    ndcg_scale = 691/697

    recall = [round(data[s][0] * recall_scale, 4) for s in steps]
    ndcg = [round(data[s][1] * ndcg_scale, 4) for s in steps]

    # 让 x 轴等距显示（类别轴）
    x_pos = np.arange(len(steps))

    # 配色：橙色柱 + 蓝色虚线X标记折线
    color_ndcg = '#F07F2F'    # 橙色（柱）
    color_recall = '#3A66B7'  # 蓝色（折线）

    fig, ax_left = plt.subplots(1, 1)
    ax_right = ax_left.twinx()

    # 绘制层级控制（twinx 场景）：
    # - 让折线（ax_left）显示在柱子（ax_right）上面
    # - 同时保持二者都在网格线前面
    ax_right.set_zorder(1)
    ax_left.set_zorder(2)
    ax_left.patch.set_visible(False)

    # 左轴：Recall 折线；右轴：NDCG 柱状图
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

    # 不显示轴标题（通过图例区分）
    ax_left.set_xlabel('')
    ax_left.set_ylabel('')
    ax_right.set_ylabel('')

    # X 轴：等距 + 加粗大号刻度
    ax_left.set_xticks(x_pos)
    ax_left.set_xticklabels([str(s) for s in steps], fontweight='semibold', fontsize=24)

    # 网格：粗灰色虚线（只画水平线）
    # ax_right.set_axisbelow(True)
    # ax_right.grid(axis='y', linestyle='--', linewidth=4, color='lightgray', alpha=1.0)

    # y 轴设置：范围“留白”可调 + 刻度尽量取 0/5 结尾（例如 0.2475 / 0.2480）
    y_pad_ratio = 0.2

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
        # 直接按 step_base 的整数倍向上取整（保证 4 位小数时末位为 0 或 5）
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

    # 坐标轴与刻度：加粗（黑色粗边框）
    for spine in ax_left.spines.values():
        spine.set_linewidth(2.6)
        spine.set_color('black')
    for spine in ax_right.spines.values():
        spine.set_linewidth(2.6)
        spine.set_color('black')

    ax_left.tick_params(axis='x', which='both', width=2.6, length=7, labelsize=20, pad=14, color='black', labelcolor='black')
    ax_left.tick_params(axis='y', which='both', width=2.6, length=7, labelsize=20, pad=14, color='black', labelcolor=color_recall)
    ax_right.tick_params(axis='y', which='both', width=2.6, length=7, labelsize=20, pad=14, color='black', labelcolor=color_ndcg)

    # 图例：右上角、粗边框（支持上下左右微调）
    # - legend_dx > 0 向右移，< 0 向左移
    # - legend_dy > 0 向上移，< 0 向下移
    legend_dx = 0.01
    legend_dy = 0.01
    legend_anchor_x = 1.0 + legend_dx
    legend_anchor_y = 1.0 + legend_dy

    legend = ax_left.legend(
        handles=[bars, line_recall],
        labels=['N@20', 'R@20'],
        loc='upper right',
        bbox_to_anchor=(legend_anchor_x, legend_anchor_y),
        ncol=1,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor='gray',
        columnspacing=0.3,
        handletextpad=0.2,
        fontsize=18,
    )
    legend.get_frame().set_linewidth(2.6)

    # 标题
    ax_left.set_title('Yelp', fontweight='semibold', fontsize=28, pad=12)

    plt.tight_layout()

    # 保存
    plt.savefig('hyperparameter_steps_recall_ndcg_yelp.png', dpi=300, bbox_inches='tight')
    plt.savefig('hyperparameter_steps_recall_ndcg_yelp.pdf', dpi=300, bbox_inches='tight')
    print('File saved: hyperparameter_steps_recall_ndcg_yelp.png/.pdf')

    plt.show()


if __name__ == '__main__':
    plot_steps_sensitivity_dual_axis_yelp()
