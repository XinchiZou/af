import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def _load_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}

def main() -> None:
    parser = argparse.ArgumentParser(description='绘制散点图：无坐标刻度，无总体图注。')
    parser.add_argument('--npz', type=str, default='latest_scatter.npz', help='npz 文件路径')
    parser.add_argument('--out_file', type=str, default='scatter.png', help='输出图片路径')

    parser.add_argument('--font', type=str, default='Times New Roman', help='字体')
    parser.add_argument('--font_size', type=float, default=20.0, help='基础字号')
    parser.add_argument('--title_size', type=float, default=30.0, help='子图标题字号')
    parser.add_argument('--legend_size', type=float, default=20.0, help='图例字号')

    parser.add_argument('--point_size', type=float, default=50.0)
    parser.add_argument('--alpha', type=float, default=0.75)
    parser.add_argument('--dpi', type=int, default=300)

    args = parser.parse_args()

    # 1. 加载数据
    d = _load_npz(args.npz)
    enc_2d = d['encoder_2d']
    flow_2d = d['flow_2d']
    labels = d['labels'].astype(str)
    # breakpoint()
    categories = d['categories'].astype(str).tolist()

    # 图例显示名（手动固定，不改 labels 的原始取值）
    legend_name_map = {
        'sports': 'Sports',
        'autos': 'Autos',
        'foodanddrink': 'Food & Drink',
        'food_and_drink': 'Food & Drink',
        'food and drink': 'Food & Drink',
    }

    # 若 npz 里的类别刚好是这三类，则固定显示顺序
    preferred_order = ['sports', 'autos', 'foodanddrink']
    if all(c in categories for c in preferred_order):
        categories = preferred_order
    # breakpoint()
    out_dir = os.path.dirname(os.path.abspath(args.out_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 2. 样式设置
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': [args.font, 'Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': args.font_size,
        'axes.labelsize': args.font_size,
        'legend.fontsize': args.legend_size,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    cmap = plt.get_cmap('tab20')
    cat_colors = {cat: cmap(i % cmap.N) for i, cat in enumerate(categories)}

    # 3. 绘图核心函数
    def _plot_on_ax(ax, Y: np.ndarray, sub_title: str) -> None:
        for cat in categories:
            idx = np.where(labels == cat)[0]
            pts = Y[idx]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=args.point_size,
                alpha=args.alpha,
                color=cat_colors[cat],
                label=legend_name_map.get(cat, cat),
                edgecolors='none',
            )
        
        # --- 修改1：移除所有标签、刻度和数字 ---
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([]) # 移除X轴刻度和数字
        ax.set_yticks([]) # 移除Y轴刻度和数字

        # --- 子图标题位置 ---
        # y=-0.03: 紧贴边框下方 (因为没有刻度数字了，可以靠得很近)
        ax.text(0.5, -0.03, sub_title, transform=ax.transAxes, 
                ha='center', va='top', fontsize=args.title_size, fontweight='bold')

        # --- 图例放在左上角 ---
        ax.legend(loc='upper left', frameon=True, framealpha=0.8, borderaxespad=0.5)

    # 4. 创建画布
    # 修改：高度减小，因为不需要放底部长文字了
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 子图标题 (a) 和 (b)
    title_a = "(a) Initial Semantic Embeddings"
    title_b = "(b) Flow-Refined Embeddings"

    # 绘制两个子图
    _plot_on_ax(axes[0], enc_2d, title_a)
    _plot_on_ax(axes[1], flow_2d, title_b)

    # --- 修改2：布局调整 ---
    # bottom=0.08: 只需要留一点点空间给 (a)/(b) 标题即可，不需要留大片空白
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08) 

    # 注意：这里不再调用 fig.text 添加总体题注

    print(f'Saving clean plot to: {args.out_file}')
    plt.savefig(args.out_file, dpi=args.dpi, bbox_inches='tight')
    plt.savefig('scatter.pdf', dpi=args.dpi, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()