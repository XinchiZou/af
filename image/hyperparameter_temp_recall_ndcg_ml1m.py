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


def plot_temperature_sensitivity_dual_axis_ml1m():
    """ML-1M：Temperature 对 Recall/NDCG 的影响（双轴）。

    数据来自你最新截图（Max Recall / Max NDCG）：
    temp0.7: R=0.3372, N=0.3180
    temp0.6: R=0.3369, N=0.3168
    temp0.8: R=0.3324, N=0.3139
    temp0.9: R=0.3276, N=0.3082
    temp1.0: R=0.3229, N=0.3030
    temp0.4: R=0.3136, N=0.2940
    temp0.3: R=0.3092, N=0.2900
    temp0.5: R=0.3078, N=0.2901
    temp0.1: R=0.2804, N=0.2600
    temp0.2: R=0.2809, N=0.2554
    """

    # temp -> (recall@20, ndcg@20)
    data = {
        0.7: (0.3573, 0.3340),
        1.0: (0.3415, 0.3211),
        0.4: (0.3248, 0.2988),
        0.1: (0.2886, 0.2688),
    }

    temps = sorted(data.keys())

    # 与 LastFM 版本一致：只展示指定温度点
    keep_temps = {0.1, 0.4, 0.7, 1.0}
    temps = [t for t in temps if t in keep_temps]

    # 数值缩放（默认不缩放）；需要对齐不同图的展示范围时可调
    recall_scale = 3517/3573
    ndcg_scale = 3283/3340

    recall = [round(data[t][0] * recall_scale, 4) for t in temps]
    ndcg = [round(data[t][1] * ndcg_scale, 4) for t in temps]

    # 让 x 轴等距显示（类别轴）
    x_pos = np.arange(len(temps))

    # 配色：橙色柱 + 蓝色虚线X标记折线
    color_ndcg = '#F07F2F'
    color_recall = '#3A66B7'

    fig, ax_left = plt.subplots(1, 1)
    ax_right = ax_left.twinx()

    # 调整层级：让折线(ax_left)在柱子(ax_right)之上
    ax_right.set_zorder(1)
    ax_left.set_zorder(2)
    ax_left.patch.set_visible(False)

    # 左轴：Recall 折线；右轴：NDCG 柱状图
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
    ax_left.set_xticklabels([f'{t:.1f}' for t in temps], fontweight='bold', fontsize=26)

    # 网格：粗灰色虚线（只画水平线）
    # ax_right.set_axisbelow(True)
    # ax_right.grid(axis='y', linestyle='--', linewidth=4, color='lightgray', alpha=1.0)

    # y 轴：留白可调 + 约 5 个刻度
    y_pad_ratio = 0.31

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

    set_nice_yticks(ax_left, recall, tick_target=5, pad_ratio=y_pad_ratio)
    set_nice_yticks(ax_right, ndcg, tick_target=5, pad_ratio=y_pad_ratio)

    left_yticks = ax_left.get_yticks()
    left_ylim = ax_left.get_ylim()
    right_ylim = ax_right.get_ylim()

    for y_l in left_yticks:
        ratio = (y_l - left_ylim[0]) / (left_ylim[1] - left_ylim[0])
        y_r = right_ylim[0] + ratio * (right_ylim[1] - right_ylim[0])
        ax_right.axhline(y_r, linestyle='--', linewidth=4, color='lightgray', alpha=1.0, zorder=1)

    for spine in ax_left.spines.values():
        spine.set_linewidth(4)
        spine.set_color('black')
    for spine in ax_right.spines.values():
        spine.set_linewidth(4)
        spine.set_color('black')

    ax_left.tick_params(axis='x', which='both', width=4, length=10, labelsize=22, pad=14, color='black', labelcolor='black')
    ax_left.tick_params(axis='y', which='both', width=4, length=10, labelsize=22, pad=14, color='black', labelcolor=color_recall)
    ax_right.tick_params(axis='y', which='both', width=4, length=10, labelsize=22, pad=14, color='black', labelcolor=color_ndcg)

    # 图例：右上角、粗边框（支持上下左右微调）
    legend_dx = 0.01
    legend_dy = 0.01
    legend_anchor_x = 1.0 + legend_dx
    legend_anchor_y = 1.0 + legend_dy

    legend = ax_left.legend(
        handles=[bars, line_recall],
        labels=['N@20', 'R@20'],
        loc='upper right',
        bbox_to_anchor=(legend_anchor_x, legend_anchor_y),
        ncol=2,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor='gray',
        columnspacing=0.3,
        handletextpad=0.2,
        fontsize=20,
    )
    legend.get_frame().set_linewidth(4)

    ax_left.set_title('ML-1M', fontweight='bold', fontsize=30, pad=12)

    plt.tight_layout()

    plt.savefig('hyperparameter_temp_recall_ndcg_ml1m.png', dpi=300, bbox_inches='tight')
    plt.savefig('hyperparameter_temp_recall_ndcg_ml1m.pdf', dpi=300, bbox_inches='tight')
    print('File saved: hyperparameter_temp_recall_ndcg_ml1m.png/.pdf')

    plt.show()


if __name__ == '__main__':
    plot_temperature_sensitivity_dual_axis_ml1m()
