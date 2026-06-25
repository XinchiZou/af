import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

# ACL 风格配置
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 13,
    "figure.figsize": (30, 6.5),
    "axes.linewidth": 1.1,
    "axes.edgecolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def plot_pareto_ndcg_time_params_v1():
    # =========================
    # 1) 数据准备
    # =========================
    performance_data = {
        'Last-FM': {'DiffKG': 0.2091, 'KMDCL': 0.2205, 'FlowKG': 0.2509, 'LightKG': 0.2430, 'KGRec': 0.1556},
        'MIND':   {'DiffKG': 0.0389, 'KMDCL': 0.0419, 'FlowKG': 0.0607, 'LightKG': 0.0276, 'KGRec': 0.0319},
        'Yelp':   {'DiffKG': 0.0518, 'KMDCL': 0.0546, 'FlowKG': 0.0691, 'LightKG': 0.0458, 'KGRec': 0.0476},
        'ML-1M':  {'DiffKG': 0.2875, 'KMDCL': 0.3059, 'FlowKG': 0.3283, 'LightKG': 0.3132, 'KGRec': 0.1901}
    }

    time_data = {
        'Last-FM': {'DiffKG': 2.123,   'KMDCL': 2.804,   'FlowKG': 0.667,   'LightKG': 0.2,   'KGRec': 1.291},
        'MIND':   {'DiffKG': 144.320, 'KMDCL': 234.000, 'FlowKG': 166.500, 'LightKG': 252.0, 'KGRec': 660.0},
        'Yelp':   {'DiffKG': 10.200,  'KMDCL': 18.658,  'FlowKG': 9.700,   'LightKG': 13.4,  'KGRec': 15.874},
        'ML-1M':  {'DiffKG': 10.78,   'KMDCL': 11.904,  'FlowKG': 10.638,  'LightKG': 11.5,  'KGRec': 10.12}
    }

    params_data = {
        'Last-FM': {'DiffKG': 19483932,  'KMDCL': 28849932,  'FlowKG': 765934,   'LightKG': 715835,  'KGRec': 744000},
        'MIND':   {'DiffKG': 108795263, 'KMDCL': 158340263, 'FlowKG': 9679086,  'LightKG': 8624131, 'KGRec': 9657152},
        'Yelp':   {'DiffKG': 52533718,  'KMDCL': 76709718,  'FlowKG': 4180910,  'LightKG': 4137867, 'KGRec': 4158976},
        'ML-1M':  {'DiffKG': 14301887,  'KMDCL': 21030887,  'FlowKG': 860526,   'LightKG': 817105,  'KGRec': 838592}
    }

    models = ['FlowKG', 'KMDCL', 'DiffKG', 'LightKG', 'KGRec']
    datasets = ['MIND', 'ML-1M', 'Last-FM', 'Yelp']

    # =========================
    # 2) 样式映射
    # =========================
    markers = {
        'KGRec': 'o',
        'DiffKG': 's',
        'KMDCL': 'D',
        'LightKG': '^',
        'FlowKG': '*'
    }

    # 参数量只体现在颜色深度上：点大小固定
    base_size = 285

    # 统一颜色映射范围（单位：Millions）
    all_params_m = []
    for ds in datasets:
        for m in models:
            all_params_m.append(params_data[ds][m] / 1e6)
    vmin = float(min(all_params_m))
    vmax = float(max(all_params_m))

    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('cividis')

    # =========================
    # 3) 画布：1x4
    # =========================
    fig, axes = plt.subplots(1, 4)
    axes = np.ravel(axes)

    # 线性轴：左右范围在 min/max 基础上，按 (max-min) 的固定倍数扩展
    def _auto_linear_xlim(times_list, pad_ratio: float = 0.2, pad_min: float = 0):
        t_min = float(min(times_list))
        t_max = float(max(times_list))
        span = max(t_max - t_min, 1e-6)
        pad = max(span * pad_ratio, pad_min)
        return max(0.0, t_min - pad), t_max + pad

    def _auto_linear_ticks(x_left: float, x_right: float):
        span = max(x_right - x_left, 1e-6)
        # 若范围很窄（如 9~14s），线性轴刻度只显示“偶数”
        if x_left >= 1 and span <= 20:
            step = 2
            # 边界附近也尽量给出刻度（避免 8.x~12.x 只显示 10/12）
            start = int(np.floor(x_left / step) * step)
            end = int(np.ceil(x_right / step) * step)
            if start <= end:
                ticks = list(range(start, end + 1, step))
            else:
                # 范围太窄时兜底：取中间最近的偶数
                mid = (x_left + x_right) / 2
                ticks = [int(np.round(mid / step) * step)]
        else:
            # 目标刻度数量 5~7 个
            raw_step = span / 6.0
            nice_steps = [0.5, 1, 2, 5, 10, 20, 50, 100]
            step = min(nice_steps, key=lambda s: abs(s - raw_step))
            start = np.floor(x_left / step) * step
            end = np.ceil(x_right / step) * step
            ticks = np.arange(start, end + step * 0.5, step).tolist()
        # 整数优先显示为整数
        labels = []
        for t in ticks:
            if abs(t - round(t)) < 1e-8:
                labels.append(str(int(round(t))))
            else:
                labels.append(f"{t:g}")
        return ticks, labels

    # log 轴：左右范围在 log10 空间按固定 decades 扩展（只在范围不窄时使用）
    def _auto_log_xlim(times_list, pad_decades: float = 0.2):
        t_min = float(min(times_list))
        t_max = float(max(times_list))
        t_min = max(t_min, 1e-6)
        t_max = max(t_max, t_min * 1.01)

        log_min = np.log10(t_min)
        log_max = np.log10(t_max)
        pad = max(pad_decades, 0.0)
        x_left = 10 ** (log_min - pad)
        x_right = 10 ** (log_max + pad)

        # 合理钳制，避免极端范围
        x_left = max(x_left, 0.3)
        x_right = min(x_right, 5000)

        if x_right <= x_left * 1.05:
            x_right = x_left * 1.2

        return x_left, x_right

    def _auto_log_ticks(x_left: float, x_right: float):
        def _fmt(x: float) -> str:
            if x >= 100:
                return f"{int(round(x))}"
            if x >= 10:
                return f"{x:.0f}"
            if x >= 1:
                return f"{x:.0f}"
            return f"{x:g}"

        # 若范围很窄（例如 9~14s），使用“整数刻度”避免出现 9×10^0 这种形式
        ratio = x_right / max(x_left, 1e-12)
        if ratio <= 2.5 and x_left >= 1:
            left_i = int(np.floor(x_left))
            right_i = int(np.ceil(x_right))
            ticks = list(range(left_i, right_i + 1))
            # 控制刻度数量，避免太密
            if len(ticks) > 8:
                step = int(np.ceil(len(ticks) / 8))
                ticks = ticks[::step]
                if ticks[-1] != right_i:
                    ticks.append(right_i)
            labels = [str(int(t)) for t in ticks]
            return ticks, labels

        log_left = np.log10(x_left)
        log_right = np.log10(x_right)
        k_min = int(np.floor(log_left))
        k_max = int(np.ceil(log_right))

        candidates = []
        for k in range(k_min, k_max + 1):
            for m in (1, 2, 5):
                candidates.append(m * (10 ** k))

        ticks = [t for t in candidates if x_left <= t <= x_right]
        # 确保边界附近有刻度（尤其 <1s 的 0.5）
        if x_left <= 0.5 <= x_right and 0.5 not in ticks:
            ticks = [0.5] + ticks
        ticks = sorted(set(ticks))
        labels = [_fmt(t) for t in ticks]
        return ticks, labels

    def _pretty_time(t: float) -> str:
        return f"{t:.3g}"

    def draw_chart(ax, dataset_name: str):
        scatters = []

        # 提取数据
        times = [time_data[dataset_name][m] for m in models]
        ndcgs = [performance_data[dataset_name][m] for m in models]
        params_m = [params_data[dataset_name][m] / 1e6 for m in models]

        y_min = min(ndcgs)
        y_max = max(ndcgs)
        y_pad = max((y_max - y_min) * 0.28, 0.005)
        y_bottom = y_min - y_pad
        y_top = y_max + y_pad
        y_bounds = {
            'ML-1M': (y_min - y_pad * 1.15, y_max + y_pad * 1.45),
            'Last-FM': (y_min - y_pad * 1.05, y_max + y_pad * 1.35),
        }
        if dataset_name in y_bounds:
            y_bottom, y_top = y_bounds[dataset_name]

        # X 轴：范围很窄（例如 9~14s）时使用线性轴显示真实值；否则用 log-x
        t_min = float(min(times))
        t_max = float(max(times))
        ratio = t_max / max(t_min, 1e-12)
        use_linear_x = (ratio <= 2.5 and t_min >= 1)

        if use_linear_x:
            x_left, x_right = _auto_linear_xlim(times)
            ticks, labels = _auto_linear_ticks(x_left, x_right)
        else:
            x_left, x_right = _auto_log_xlim(times)
            ticks, labels = _auto_log_ticks(x_left, x_right)

        if dataset_name == 'ML-1M':
            x_left, x_right = 9.75, 12.15
            ticks, labels = [10, 11, 12], ['10', '11', '12']
        elif dataset_name == 'Yelp':
            x_left, x_right = 8.2, 19.7
            ticks, labels = [10, 12, 14, 16, 18], ['10', '12', '14', '16', '18']
        elif dataset_name == 'Last-FM':
            x_left, x_right = 0.45, 3.55
            ticks, labels = [0.5, 1, 2, 3], ['0.5', '1', '2', '3']

        # 用时间排序给标注分配不同的“上下”错位，减少重叠（保持严格正上/正下）
        order = np.argsort(times)
        rank_by_model = {models[order[j]]: j for j in range(len(models))}

        for i, m in enumerate(models):
            is_ours = (m == 'FlowKG')
            line_w = 2.5 if is_ours else 1.0
            edge_c = 'black'

            sc = ax.scatter(
                times[i],
                ndcgs[i],
                c=[params_m[i]],
                norm=norm,
                cmap=cmap,
                s=(base_size * (1.15 if is_ours else 1.0)),
                marker=markers[m],
                edgecolors='#D94A3A' if is_ours else edge_c,
                linewidths=line_w,
                zorder=8,
                alpha=0.96
            )
            scatters.append(sc)

            # 标注严格放在点的正上/正下
            x_off = 0
            ha = 'center'
            label_adjust = {
                ('ML-1M', 'FlowKG'): (-9, 14, 11),
                ('ML-1M', 'DiffKG'): (-21, 7, 11),
                ('ML-1M', 'LightKG'): (-15, 16, 11),
                ('ML-1M', 'KMDCL'): (-17, 17, 11),
                ('Last-FM', 'DiffKG'): (-18, 14, 12),
                ('Last-FM', 'KMDCL'): (-18, 15, 12),
            }

            # 轻微上下错位，缓解近邻重叠（尤其 ML-1M）
            rank = rank_by_model.get(m, 0)
            # name_jitter = -10 if (rank % 2 == 1) else 0
            # param_jitter = 10 if (rank % 2 == 1) else 0
            name_jitter = 0
            param_jitter = 0
            label_fontsize = 13
            if (dataset_name, m) in label_adjust:
                name_offset_override, param_offset_override, label_fontsize = label_adjust[(dataset_name, m)]
            else:
                name_offset_override = None
                param_offset_override = None
            # --- 文本标注 1: 模型名 + 时间 (正下方) ---
            display_text = f"{m}\n({_pretty_time(times[i])}s)"
            y_offset_name = name_offset_override if name_offset_override is not None else -15 + name_jitter
            ax.annotate(
                display_text,
                (times[i], ndcgs[i]),
                xytext=(x_off, y_offset_name),
                textcoords='offset points',
                ha=ha,
                va='top',
                fontsize=label_fontsize,
                fontweight='semibold',
                color='#222222',
                zorder=12
            )

            # --- 文本标注 2: 参数量 (正上方) ---
            p_txt = f"{params_m[i]:.2f}M"
            y_offset_param = param_offset_override if param_offset_override is not None else 13 + param_jitter
            # if is_ours:
            #     y_offset_param = 20 + param_jitter
            # 顶部空间不足时，把参数标注放到点下方
            if ndcgs[i] >= y_top - (y_top - y_bottom) * 0.10:
                y_offset_param = -18 + name_jitter
            param_va = 'bottom' if y_offset_param >= 0 else 'top'
            ax.annotate(
                p_txt,
                (times[i], ndcgs[i]),
                xytext=(x_off, y_offset_param),
                textcoords='offset points',
                ha=ha,
                va=param_va,
                fontsize=label_fontsize,
                color='#333333',
                zorder=12
            )

        # 坐标轴设置
        ax.set_xscale('linear' if use_linear_x else 'log')
        ax.set_xlim(x_left, x_right)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        if use_linear_x:
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            # 增加 log 轴的 minor ticks 用于更密的竖向网格线；minor 标签仍然隐藏
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
            ax.xaxis.set_minor_formatter(NullFormatter())

        ax.grid(True, linestyle=(0, (2.5, 2.5)), linewidth=0.8, color='#B8B8B8', alpha=0.65, which='major')
        ax.grid(True, axis='x', linestyle=(0, (2, 2.8)), linewidth=0.55, color='#B8B8B8', alpha=0.45, which='minor')
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)
            spine.set_color('#333333')
        ax.tick_params(direction='out', width=1.0, length=4, pad=6, colors='#333333')
        ax.set_title(dataset_name, fontweight='bold', y=1.02, fontsize=22)
        ax.set_ylim(y_bottom, y_top)

        # --- 箭头（只放在第一个子图上） ---
        if dataset_name == 'MIND':
            # 分别控制：箭头(起点/终点) 与 文字(位置)
            arrow_end = (0.02, 0.98)    # 箭头尖端位置
            arrow_start = (0.12, 0.90)  # 箭头起点位置
            text_pos = (0.02, 0.88)     # 文字位置

            # 只画箭头（不附带文字）
            ax.annotate(
                "",
                xy=arrow_end,
                xytext=arrow_start,
                xycoords='axes fraction',
                textcoords='axes fraction',
                arrowprops=dict(facecolor='#555555', edgecolor='#555555', arrowstyle='->', lw=1.25),
            )

            # 单独画文字（不影响箭头方向）
            ax.text(
                text_pos[0],
                text_pos[1],
                "High Efficiency\nHigh Accuracy",
                transform=ax.transAxes,
                fontsize=12,
                color='#555555',
                ha='left',
                va='top'
            )

        return scatters[0]

    # 绘制四个子图（每个子图单独设置，方便分别调整格式）
    ax_mind, ax_ml1m, ax_lastfm, ax_yelp = axes

    sc_mind = draw_chart(ax_mind, 'MIND')
    _ = draw_chart(ax_ml1m, 'ML-1M')
    _ = draw_chart(ax_lastfm, 'Last-FM')
    _ = draw_chart(ax_yelp, 'Yelp')
    first_scatter = sc_mind

    # 1x4：每个子图单独设置 label
    ax_mind.set_xlabel('Training Time (Seconds)', fontsize=20, fontweight='bold')
    ax_ml1m.set_xlabel('Training Time (Seconds)', fontsize=20, fontweight='bold')
    ax_lastfm.set_xlabel('Training Time (Seconds)', fontsize=20, fontweight='bold')
    ax_yelp.set_xlabel('Training Time (Seconds)', fontsize=20, fontweight='bold')
    ax_mind.set_ylabel('NDCG@20', fontsize=20, fontweight='bold')

    # 共享 Colorbar + 留出底部空间放总图例
    plt.subplots_adjust(right=0.9, bottom=0.25, top=0.80, wspace=0.22)
    cbar_ax = fig.add_axes([0.92, 0.22, 0.012, 0.62])
    cbar = fig.colorbar(first_scatter, cax=cbar_ax)
    cbar.set_label('Parameter Count (Millions)', rotation=270, labelpad=18, fontweight='bold', fontsize=20)
    cbar.set_ticks([1, 10, 100])
    cbar.ax.set_yticklabels(['1', '10', '100'])
    cbar.ax.tick_params(labelsize=18, width=0.9, length=3.5)
    cbar.outline.set_linewidth(0.9)

    # 图例（沿用原风格表达）
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#8F8F8F', label='FlowKG', markersize=18, markeredgecolor='#D94A3A', markeredgewidth=1.9),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#8F8F8F', label='KMDCL', markersize=14, markeredgecolor='#333333', markeredgewidth=1.1),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#8F8F8F', label='DiffKG', markersize=14, markeredgecolor='#333333', markeredgewidth=1.1),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#8F8F8F', label='LightKG', markersize=14, markeredgecolor='#333333', markeredgewidth=1.1),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#8F8F8F', label='KGRec', markersize=14, markeredgecolor='#333333', markeredgewidth=1.1),
    ]
    # 每张子图右上角都放一个图例（逐个设置）
    ax_mind.legend(
        handles=legend_elements,
        loc='upper right',
        ncol=1,
        frameon=True,
        fancybox=False,
        edgecolor='black',
        fontsize=12,
        borderpad=0.5,
        labelspacing=1,     # 行间距（上下更疏）
        columnspacing=2.0,    # 两列更“宽”
        handletextpad=0.8,    # 图标-文字间距
        handlelength=1.6,     # 图标“把手”长度
    )
    ax_ml1m.legend(
        handles=legend_elements,
        loc='upper right',
        ncol=1,
        frameon=True,
        fancybox=False,
        edgecolor='black',
        fontsize=12,
        borderpad=0.5,
        labelspacing=1,
        columnspacing=2.0,
        handletextpad=0.8,
        handlelength=1.6,
    )
    ax_lastfm.legend(
        handles=legend_elements,
        loc='upper right',
        ncol=1,
        frameon=True,
        fancybox=False,
        edgecolor='black',
        fontsize=12,
        borderpad=0.5,
        labelspacing=1,
        columnspacing=2.0,
        handletextpad=0.8,
        handlelength=1.6,
    )
    ax_yelp.legend(
        handles=legend_elements,
        loc='upper right',
        ncol=1,
        frameon=True,
        fancybox=False,
        edgecolor='black',
        fontsize=12,
        borderpad=0.5,
        labelspacing=1,
        columnspacing=2.0,
        handletextpad=0.8,
        handlelength=1.6,
    )

    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.46, 1.0),
        ncol=5,
        frameon=False,
        fancybox=False,
        fontsize=18,
        borderpad=0.0,
        labelspacing=0.6,
        columnspacing=2.0,
        handletextpad=0.62,
        handlelength=2.3,
    )

    plt.savefig('efficiency_pareto.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig('efficiency_pareto.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    plot_pareto_ndcg_time_params_v1()
