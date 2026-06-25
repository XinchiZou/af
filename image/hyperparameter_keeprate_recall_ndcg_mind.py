import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import math

# --- 与 hyperparameter_keeprate_recall_ndcg_lastfm.py 风格保持一致 ---
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


def plot_keeprate_sensitivity_dual_axis_mind():
    """MIND：Keep rate 对 Recall/NDCG 的影响（双轴）。

    数据来自你给的截图：
    keep0.6: R=0.0944, N=0.0612
    keep0.8: R=0.0945, N=0.0611
    keep0.2: R=0.0945, N=0.0609
    keep0.4: R=0.0941, N=0.0610
    """

    # keeprate -> (recall@20, ndcg@20)
    data = {
        0.2: (0.0945, 0.0609),
        0.4: (0.0941, 0.0610),
        0.6: (0.0944, 0.0612),
        0.8: (0.0945, 0.0611),
    }

    keep_rates = sorted(data.keys())
    # 数值缩放（默认不缩放）；需要对齐不同图的展示范围时可调
    recall_scale = 937/945
    ndcg_scale = 607/612
    recall = [round(data[k][0] * recall_scale, 4) for k in keep_rates]
    ndcg = [round(data[k][1] * ndcg_scale, 4) for k in keep_rates]

    # 让 x 轴等距显示（类别轴），但保留真实 keeprate 作为标签
    x_pos = np.arange(len(keep_rates))

    # 配色：对齐 temperature/keeprate 图风格（橙色柱 + 蓝色虚线X标记折线）
    color_ndcg = '#F07F2F'    # 橙色（柱）
    color_recall = '#3A66B7'  # 蓝色（折线）

    fig, ax_left = plt.subplots(1, 1)
    ax_right = ax_left.twinx()

    # 调整层级：让折线(ax_left)在柱子(ax_right)之上
    ax_right.set_zorder(1)
    ax_left.set_zorder(2)
    ax_left.patch.set_visible(False)

    # 右轴：NDCG 柱状图
    bar_width = 0.48 if len(keep_rates) >= 5 else 0.36
    bars = ax_right.bar(
        x_pos,
        ndcg,
        width=bar_width,
        color=color_ndcg,
        label='N@20',
        zorder=2,
    )

    # 左轴：Recall 折线
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

    # y 轴范围与刻度：自动取整到固定步长，保证 4 位小数整齐
    y_pad_ratio = 0.36

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

    # 对齐双轴的水平网格线（以左轴为基准映射到右轴）
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

    # 图例：右上角、粗边框
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
    plt.savefig('hyperparameter_keeprate_recall_ndcg_mind.png', dpi=300, bbox_inches='tight')
    plt.savefig('hyperparameter_keeprate_recall_ndcg_mind.pdf', dpi=300, bbox_inches='tight')
    print('File saved: hyperparameter_keeprate_recall_ndcg_mind.png/.pdf')

    plt.show()


if __name__ == '__main__':
    plot_keeprate_sensitivity_dual_axis_mind()
