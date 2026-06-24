import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms

from matplotlib import rcParams

# === 全局风格：对齐 figure_group.py ===
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 20

# 说明：该消融图 x 轴标签较多，为避免拥挤，后续仍会在局部设置较小的 xtick 字号。

def plot_mechanism_analysis_lastfm(show_metrics: bool = False):
    # === 1. 数据准备 (LastFM Dataset) ===

# # --- Group 1: Architecture Evolution (骨架与完全体) ---
# # Logic: Coupled 最低 -> Decoupled 稍好 (基础提升) -> Full SOTA
#     g1_labels = ['Coupled\nSkeleton', 'Decoupled\nSkeleton(Base)', 'Full\nModel']
#     g1_recall = [0.3420, 0.3520, 0.4039] 
#     g1_ndcg   = [0.1900, 0.1980, 0.2448]

# # --- Group 2: Additive Analysis (Decoupled Base + Single Module) ---
# # Logic: 在 0.352 的基础上做加法。CL贡献最大，FM次之，SL辅助
#     g2_labels = ['Base\n+ SL', 'Base\n+ FM', 'Base\n+ CL']
#     g2_recall = [0.3605, 0.3690, 0.3810] # 逻辑推演值
#     g2_ndcg   = [0.2030, 0.2100, 0.2180] # 逻辑推演值

# # --- Group 3: Subtractive Analysis (Full - Single Module) ---
# # Logic: 来自 Table 3 真值。去掉某模块后的表现。
# # w/o CL 掉最多 (说明CL最重要)，w/o FM 其次，w/o SL 掉最少
#     g3_labels = ['w/o SL', 'w/o FM', 'w/o CL']
#     g3_recall = [0.4007, 0.3828, 0.3799] # 真值
#     g3_ndcg   = [0.2413, 0.2327, 0.2147] # 真值

# --- Group 1: Architecture Evolution (骨架与完全体) ---
# Logic: Coupled 最低 -> Decoupled 稍好 (基础提升) -> Full SOTA
    g1_labels = ['Coupled\nSkeleton', 'Decoupled\nSkeleton(Base)', 'Full\nModel']
    g1_recall = [0.3477, 0.3578, 0.4106] 
    g1_ndcg   = [0.1947, 0.2029, 0.2509]

# --- Group 2: Additive Analysis (Decoupled Base + Single Module) ---
# Logic: 在 0.3578 的基础上做加法。CL贡献最大，FM次之，SL辅助
    g2_labels = ['Base\n+ SL', 'Base\n+ FM', 'Base\n+ CL']
    g2_recall = [0.3665, 0.3751, 0.3873] # 逻辑推演值
    g2_ndcg   = [0.2081, 0.2152, 0.2234] # 逻辑推演值

# --- Group 3: Subtractive Analysis (Full - Single Module) ---
# Logic: 来自 Table 3 真值。去掉某模块后的表现。
# w/o CL 掉最多 (说明CL最重要)，w/o FM 其次，w/o SL 掉最少
    g3_labels = ['w/o SL', 'w/o FM', 'w/o CL']
    g3_recall = [0.4073, 0.3891, 0.3862] # 真值
    g3_ndcg   = [0.2473, 0.2385, 0.2200] # 真值

# 合并数据
    labels = g1_labels + g2_labels + g3_labels
    recall_data = g1_recall + g2_recall + g3_recall
    ndcg_data   = g1_ndcg + g2_ndcg + g3_ndcg

    # === 2. 绘图设置 ===
    x = np.arange(len(labels))  # 标签位置
    width = 0.35  # 柱状图宽度

    fig, ax = plt.subplots(figsize=(14, 7))

    # 右侧 Y 轴用于 NDCG
    ax2 = ax.twinx()

    # 绘制柱子 (Recall 和 NDCG 并排；分别挂在左右轴)
    rects1 = ax.bar(
        x - width/2, recall_data, width,
        label='Recall@20', color='#2878B5', alpha=0.9, edgecolor='black', linewidth=0.5, zorder=3
    )
    rects2 = ax2.bar(
        x + width/2, ndcg_data, width,
        label='NDCG@20', color='#C82423', alpha=0.9, edgecolor='black', linewidth=0.5, zorder=3
    )

    # === 3. 装饰与标注 ===

# 三块区域底色覆盖（参考 Multimodal experiment.py 的 axvspan）
    ax.axvspan(-0.5, 2.5, facecolor='gray', alpha=0.08, zorder=0)
    ax.axvspan(2.5, 5.5, facecolor='gray', alpha=0.08, zorder=0)
    ax.axvspan(5.5, 8.5, facecolor='gray', alpha=0.08, zorder=0)

# 添加垂直分割线，区分三组
    ax.axvline(x=2.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(x=5.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

 # 添加区域文字标签 (Top Annotations)
 # 位置用左轴上限动态算，避免改尺度后遮挡

# 坐标轴设置
    ax.set_ylabel('Recall@20', fontsize=20, fontweight='bold')
    ax2.set_ylabel('NDCG@20', fontsize=20, fontweight='bold')
    # ax.set_title('In-depth Mechanism Analysis on LastFM Dataset', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=20, rotation=25, ha='right', fontweight='bold')

    # 统一将旋转的 x 轴刻度标签向右平移（单位：points）
    x_tick_right_shift_pts = 20
    dx_inch = x_tick_right_shift_pts / 72.0
    for tick in ax.get_xticklabels():
        tick.set_transform(tick.get_transform() + transforms.ScaledTranslation(dx_inch, 0, fig.dpi_scale_trans))

# 左右轴分别设置尺度（留一点边距）
    recall_min, recall_max = min(recall_data), max(recall_data)
    ndcg_min, ndcg_max = min(ndcg_data), max(ndcg_data)
    ax.set_ylim(recall_min - 0.02, recall_max + 0.03)
    ax2.set_ylim(ndcg_min - 0.02, ndcg_max + 0.03)

    y_top = ax.get_ylim()[1]
    text_x_offset = 0.5
    # ax.text(1 + text_x_offset, y_top * 0.98, 'Architecture Evolution\n(Skeleton vs Full)', ha='center', fontsize=12, color='gray', fontweight='bold')
    # ax.text(4 + text_x_offset, y_top * 0.98, 'Additive Analysis\n(Base + Single Module)', ha='center', fontsize=12, color='gray', fontweight='bold')
    # ax.text(7 + text_x_offset, y_top * 0.98, 'Subtractive Analysis\n(Full - Single Module)', ha='center', fontsize=12, color='gray', fontweight='bold')
    ax.text(1 + text_x_offset, y_top * 0.98, 'Skeleton vs Full', ha='center', fontsize=20, color='gray', fontweight='bold')
    ax.text(4, y_top * 0.98, 'Base + Single Module', ha='center', fontsize=20, color='gray', fontweight='bold')
    ax.text(7, y_top * 0.98, 'Full - Single Module', ha='center', fontsize=20, color='gray', fontweight='bold')

# 合并图例
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc='upper left',
        frameon=True,
        fancybox=False,
        edgecolor='black',
        ncol=1,
        fontsize=14,
    )

# 网格（跟随左轴）
    ax.grid(axis='y', linestyle=':', alpha=0.5)

# 在柱子上标注数值
    def autolabel(axis, rects, color):
        for rect in rects:
            height = rect.get_height()
            axis.annotate(
                f'{height:.3f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 2),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=9,
                color=color,
                fontweight='bold',
            )

    if show_metrics:
        autolabel(ax, rects1, '#2878B5')
        autolabel(ax2, rects2, '#C82423')

    plt.tight_layout()
    plt.savefig('ablation_chart_lastfm.png', format='png', dpi=300)
    plt.savefig('ablation_chart_lastfm.pdf', format='pdf', dpi=300)
    print('File saved: ablation_chart_lastfm.png')

    backend = matplotlib.get_backend().lower()
    # if 'agg' not in backend:
    plt.show()


plot_mechanism_analysis_lastfm(show_metrics=False)