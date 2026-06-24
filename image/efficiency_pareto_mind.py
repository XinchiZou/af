import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

# 统一画图风格（论文用）
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 13,
    "figure.figsize": (7.8, 6.5),
})


def main():
    # =========================
    # 1) 数据（与总图一致）
    # =========================
    performance_data = {
        'Last-FM': {'DiffKG': 0.2091, 'KMDCL': 0.2205, 'FlowKG': 0.2509, 'LightKG': 0.2430, 'KGRec': 0.1556},
        'MIND':    {'DiffKG': 0.0389, 'KMDCL': 0.0419, 'FlowKG': 0.0607, 'LightKG': 0.0276, 'KGRec': 0.0319},
        'Yelp':    {'DiffKG': 0.0518, 'KMDCL': 0.0546, 'FlowKG': 0.0691, 'LightKG': 0.0458, 'KGRec': 0.0476},
        'ML-1M':   {'DiffKG': 0.2875, 'KMDCL': 0.3059, 'FlowKG': 0.3283, 'LightKG': 0.3132, 'KGRec': 0.1901},
    }

    time_data = {
        'Last-FM': {'DiffKG': 2.123,   'KMDCL': 2.804,   'FlowKG': 0.667,   'LightKG': 0.2,   'KGRec': 1.291},
        'MIND':    {'DiffKG': 144.320, 'KMDCL': 234.000, 'FlowKG': 166.500, 'LightKG': 252.0, 'KGRec': 660.0},
        'Yelp':    {'DiffKG': 10.200,  'KMDCL': 18.658,  'FlowKG': 9.700,   'LightKG': 13.4,  'KGRec': 15.874},
        'ML-1M':   {'DiffKG': 10.78,   'KMDCL': 11.904,  'FlowKG': 10.638,  'LightKG': 11.5,  'KGRec': 10.12},
    }

    params_data = {
        'Last-FM': {'DiffKG': 19483932,  'KMDCL': 28849932,  'FlowKG': 765934,   'LightKG': 715835,  'KGRec': 744000},
        'MIND':    {'DiffKG': 108795263, 'KMDCL': 158340263, 'FlowKG': 9679086,  'LightKG': 8624131, 'KGRec': 9657152},
        'Yelp':    {'DiffKG': 52533718,  'KMDCL': 76709718,  'FlowKG': 4180910,  'LightKG': 4137867, 'KGRec': 4158976},
        'ML-1M':   {'DiffKG': 14301887,  'KMDCL': 21030887,  'FlowKG': 860526,   'LightKG': 817105,  'KGRec': 838592},
    }

    models = ['FlowKG', 'KMDCL', 'DiffKG', 'LightKG', 'KGRec']
    dataset = 'MIND'

    markers = {
        'KGRec': 'o',
        'DiffKG': 's',
        'KMDCL': 'D',
        'LightKG': '^',
        'FlowKG': '*',
    }

    base_size = 320
    cmap = plt.cm.get_cmap('coolwarm')

    # 重要：颜色刻度范围“按当前子图”自适应（只用本数据集的参数量）
    _params_m_for_scale = [params_data[dataset][m] / 1e6 for m in models]
    vmin = float(min(_params_m_for_scale))
    vmax = float(max(_params_m_for_scale))
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-6

    # =========================
    # 2) 轴范围/刻度（保持原逻辑）
    # =========================
    def _auto_linear_xlim(times_list, pad_ratio: float = 0.2):
        t_min = float(min(times_list))
        t_max = float(max(times_list))
        span = max(t_max - t_min, 1e-6)
        pad = span * pad_ratio
        return max(0.0, t_min - pad), t_max + pad

    def _auto_linear_ticks(x_left: float, x_right: float):
        span = max(x_right - x_left, 1e-6)
        if x_left >= 1 and span <= 20:
            step = 2
            start = int(np.floor(x_left / step) * step)
            end = int(np.ceil(x_right / step) * step)
            ticks = list(range(start, end + 1, step)) if start <= end else [int(np.round(((x_left + x_right) / 2) / step) * step)]
        else:
            raw_step = span / 6.0
            nice_steps = [0.5, 1, 2, 5, 10, 20, 50, 100]
            step = min(nice_steps, key=lambda s: abs(s - raw_step))
            start = np.floor(x_left / step) * step
            end = np.ceil(x_right / step) * step
            ticks = np.arange(start, end + step * 0.5, step).tolist()

        labels = [str(int(round(t))) if abs(t - round(t)) < 1e-8 else f"{t:g}" for t in ticks]
        return ticks, labels

    def _auto_log_xlim(times_list, pad_decades: float = 0.2):
        t_min = max(float(min(times_list)), 1e-6)
        t_max = max(float(max(times_list)), t_min * 1.01)
        log_min = np.log10(t_min)
        log_max = np.log10(t_max)
        x_left = 10 ** (log_min - pad_decades)
        x_right = 10 ** (log_max + pad_decades)
        return max(x_left, 0.3), min(x_right, 5000)

    def _auto_log_ticks(x_left: float, x_right: float):
        def _fmt(x: float) -> str:
            if x >= 100:
                return f"{int(round(x))}"
            if x >= 1:
                return f"{x:.0f}"
            return f"{x:g}"

        ratio = x_right / max(x_left, 1e-12)
        if ratio <= 2.5 and x_left >= 1:
            left_i = int(np.floor(x_left))
            right_i = int(np.ceil(x_right))
            ticks = list(range(left_i, right_i + 1))
            if len(ticks) > 8:
                step = int(np.ceil(len(ticks) / 8))
                ticks = ticks[::step]
                if ticks[-1] != right_i:
                    ticks.append(right_i)
            return ticks, [str(int(t)) for t in ticks]

        log_left = np.log10(x_left)
        log_right = np.log10(x_right)
        k_min = int(np.floor(log_left))
        k_max = int(np.ceil(log_right))

        candidates = [m * (10 ** k) for k in range(k_min, k_max + 1) for m in (1, 2, 5)]
        ticks = [t for t in candidates if x_left <= t <= x_right]
        if x_left <= 0.5 <= x_right and 0.5 not in ticks:
            ticks = [0.5] + ticks
        ticks = sorted(set(ticks))
        return ticks, [_fmt(t) for t in ticks]

    # 时间显示：3 位有效数字（重要）
    def _pretty_time(t: float) -> str:
        return f"{t:.3g}"

    # =========================
    # 3) 绘图（单子图）
    # =========================
    fig, ax = plt.subplots(1, 1)

    times = [time_data[dataset][m] for m in models]
    ndcgs = [performance_data[dataset][m] for m in models]
    params_m = [params_data[dataset][m] / 1e6 for m in models]

    y_min, y_max = min(ndcgs), max(ndcgs)
    y_pad = max((y_max - y_min) * 0.25, 0.005)
    y_bottom, y_top = y_min - y_pad, y_max + y_pad

    t_min, t_max = float(min(times)), float(max(times))
    use_linear_x = (t_max / max(t_min, 1e-12) <= 2.5 and t_min >= 1)
    if use_linear_x:
        x_left, x_right = _auto_linear_xlim(times)
        ticks, labels = _auto_linear_ticks(x_left, x_right)
    else:
        x_left, x_right = _auto_log_xlim(times)
        ticks, labels = _auto_log_ticks(x_left, x_right)

    order = np.argsort(times)
    rank_by_model = {models[order[j]]: j for j in range(len(models))}

    first_scatter = None
    for i, m in enumerate(models):
        is_ours = (m == 'FlowKG')
        sc = ax.scatter(
            times[i],
            ndcgs[i],
            c=[params_m[i]],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            s=(base_size * (1.5 if is_ours else 1.0)),
            marker=markers[m],
            edgecolors='black',
            linewidth=1.0,
            zorder=8,
            alpha=0.85,
        )
        if first_scatter is None:
            first_scatter = sc

        # 重要：标注“模型名+时间”在点下方
        ax.annotate(
            f"{m}\n({_pretty_time(times[i])}s)",
            (times[i], ndcgs[i]),
            xytext=(0, -15),
            textcoords='offset points',
            ha='center',
            va='top',
            fontsize=14,
            fontweight='bold',
            zorder=12,
        )

        # 重要：参数量标注在点上方（空间不足就放到下方）
        y_offset_param = 13
        if ndcgs[i] >= y_top - (y_top - y_bottom) * 0.10:
            y_offset_param = -18
        ax.annotate(
            f"{params_m[i]:.2f}M",
            (times[i], ndcgs[i]),
            xytext=(0, y_offset_param),
            textcoords='offset points',
            ha='center',
            va=('bottom' if y_offset_param >= 0 else 'top'),
            fontsize=14,
            fontweight='bold',
            color='#333',
            zorder=12,
        )

    ax.set_xscale('linear' if use_linear_x else 'log')
    ax.set_xlim(x_left, x_right)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)

    if use_linear_x:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
        ax.xaxis.set_minor_formatter(NullFormatter())

    ax.grid(True, linestyle='--', alpha=0.4, which='major')
    ax.grid(True, axis='x', linestyle='--', alpha=0.25, which='minor')
    ax.set_title(dataset, fontweight='bold', y=1.02, fontsize=22)
    ax.set_ylim(y_bottom, y_top)

    # 只在 MIND 图上画箭头（保持原图含义）
    ax.annotate(
        "",
        xy=(0.02, 0.98),
        xytext=(0.12, 0.90),
        xycoords='axes fraction',
        textcoords='axes fraction',
        arrowprops=dict(facecolor='gray', arrowstyle='->', lw=1.5),
    )
    ax.text(
        0.02,
        0.88,
        "High Efficiency\nHigh Accuracy",
        transform=ax.transAxes,
        fontsize=12,
        color='gray',
        ha='left',
        va='top',
    )

    ax.set_xlabel('Training Time (Seconds)', fontsize=20, fontweight='bold')
    ax.set_ylabel('NDCG@20', fontsize=20, fontweight='bold')

    # Colorbar（参数量）
    cbar = fig.colorbar(first_scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Parameter Count (Millions)', rotation=270, labelpad=18, fontweight='bold', fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    # 图例顺序：FlowKG, KMDCL, DiffKG, LightKG, KGRec
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', label='FlowKG', markersize=15, markeredgecolor='k'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', label='KMDCL', markersize=11, markeredgecolor='k'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', label='DiffKG', markersize=11, markeredgecolor='k'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', label='LightKG', markersize=11, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='KGRec', markersize=11, markeredgecolor='k'),
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        frameon=True,
        fancybox=False,
        edgecolor='black',
        fontsize=12,
        borderpad=0.5,
        labelspacing=1,
        handletextpad=0.8,
        handlelength=1.6,
    )

    plt.savefig('efficiency_pareto_mind.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig('efficiency_pareto_mind.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
