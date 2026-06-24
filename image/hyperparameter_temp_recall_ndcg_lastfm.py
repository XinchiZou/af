import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import math

# --- 与 hyperparameter.py 保持一致的风格 ---
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


def plot_temperature_sensitivity_dual_axis():
    # 温度（temp）、Max Recall、Max NDCG：按你最新截图数据
    data = {
        0.7: (0.4100, 0.2499),
        1.0: (0.4073, 0.2462),
        0.4: (0.3986, 0.2454),
        0.1: (0.3777, 0.2256),
    }

    temps = sorted(data.keys())
    # 只展示指定的温度点
    # keep_temps = {0.1, 0.2, 0.5,0.7, 1.0}
    keep_temps = {0.1, 0.4, 0.7, 1.0}
    # keep_temps = {0.2, 0.4,0.6, 0.8, 1.0}
    temps = [t for t in temps if t in keep_temps]

    # 数值缩放（默认不缩放）；需要对齐不同图的展示范围时可调
    recall_scale = 1
    ndcg_scale = 1

    recall = [round(data[t][0] * recall_scale, 4) for t in temps]
    ndcg = [round(data[t][1] * ndcg_scale, 4) for t in temps]

    # 让 x 轴等距显示（类别轴），但保留真实 temp 作为标签
    x_pos = np.arange(len(temps))

    # 配色：对齐你给的参考图风格（橙色柱 + 蓝色虚线X标记折线）
    color_ndcg = '#F07F2F'    # 橙色（柱）
    color_recall = '#3A66B7'  # 蓝色（折线）

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

    # 参考图里没有轴标题，这里也去掉轴标签（通过图例区分）
    ax_left.set_xlabel('')
    ax_left.set_ylabel('')
    ax_right.set_ylabel('')

    # X 轴：等距 + 加粗大号刻度（与参考图一致）
    ax_left.set_xticks(x_pos)
    ax_left.set_xticklabels([f'{t:.1f}' for t in temps], fontweight='bold', fontsize=26)

    # 网格：粗灰色虚线（只画水平线）
    # ax_right.set_axisbelow(True)
    # ax_right.grid(axis='y', linestyle='--', linewidth=4, color='lightgray', alpha=1.0)

    # 左右 y 轴：范围“留白”可调 + 刻度尽量取 0/5 结尾（例如 0.2450 / 0.2475）
    # y_pad_ratio 越大，上下留白越多
    y_pad_ratio = 0.27

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
        # 取消 _nice_step 相关规则：直接按 step_base 的整数倍向上取整
        # step_base=0.0005 可保证 4 位小数时末位为 0 或 5。
        if raw_step <= 0 or not np.isfinite(raw_step):
            step = step_base
        else:
            step = math.ceil(raw_step / step_base) * step_base
            step = max(step, step_base)

        start = math.floor(y0 / step) * step
        end = math.ceil(y1 / step) * step

        # 生成刻度并做一次 rounding，避免浮点误差导致出现 0.244999 之类
        ticks = np.arange(start, end + step * 0.5, step)
        ticks = np.round(ticks / step_base) * step_base

        ax.set_ylim(start, end)
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

    # 左轴为 Recall，右轴为 NDCG
    set_nice_yticks(ax_left, recall, tick_target=5, pad_ratio=y_pad_ratio)
    set_nice_yticks(ax_right, ndcg, tick_target=5, pad_ratio=y_pad_ratio)

    left_yticks = ax_left.get_yticks()
    left_ylim = ax_left.get_ylim()
    right_ylim = ax_right.get_ylim()

    for y_l in left_yticks:
        ratio = (y_l - left_ylim[0]) / (left_ylim[1] - left_ylim[0])
        y_r = right_ylim[0] + ratio * (right_ylim[1] - right_ylim[0])
        ax_right.axhline(y_r, linestyle='--', linewidth=4, color='lightgray', alpha=1.0, zorder=1)

    # 坐标轴与刻度：加粗（参考图黑色粗边框）
    for spine in ax_left.spines.values():
        spine.set_linewidth(4)
        spine.set_color('black')
    for spine in ax_right.spines.values():
        spine.set_linewidth(4)
        spine.set_color('black')

    # x 轴刻度保持黑色；左右 y 轴刻度数字使用对应颜色（刻度线仍为黑色）
    ax_left.tick_params(axis='x', which='both', width=4, length=10, labelsize=22, pad=14, color='black', labelcolor='black')
    ax_left.tick_params(axis='y', which='both', width=4, length=10, labelsize=22, pad=14, color='black', labelcolor=color_recall)
    ax_right.tick_params(axis='y', which='both', width=4, length=10, labelsize=22, pad=14, color='black', labelcolor=color_ndcg)

    # 图例：右上角、粗边框
    # 图例位置微调：
    # - legend_dx > 0 向右移，< 0 向左移
    # - legend_dy > 0 向上移，< 0 向下移
    # 建议每次改 0.01~0.05 这样的小步长试。
    legend_dx = 0.01
    legend_dy = 0.01
    legend_anchor_x = 1.0 + legend_dx
    legend_anchor_y = 1.0 + legend_dy
    legend = ax_left.legend(
        handles=[bars, line_recall],
        labels=['N@20', 'R@20'],
        loc='upper right',
        # 通过 bbox_to_anchor 锚定到轴坐标系 (x,y)，再用 dx/dy 做上下左右微调
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

    # 标题：顶端居中、加粗
    ax_left.set_title('Last-FM', fontweight='bold', fontsize=30, pad=12)

    plt.tight_layout()

    # 保存
    plt.savefig('hyperparameter_temp_recall_ndcg_lastfm.png', dpi=300, bbox_inches='tight')
    plt.savefig('hyperparameter_temp_recall_ndcg_lastfm.pdf', dpi=300, bbox_inches='tight')
    print('File saved: hyperparameter_temp_recall_ndcg_lastfm.png/.pdf')

    plt.show()


if __name__ == '__main__':
    plot_temperature_sensitivity_dual_axis()
