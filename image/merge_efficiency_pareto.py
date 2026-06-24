import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
    # 重要：先生成四张子图 png，再运行本脚本进行合并
    # 默认子图文件名（来自四个子图脚本的 plt.savefig）
    parts = [
        "efficiency_pareto_mind.png",
        "efficiency_pareto_ml1m.png",
        "efficiency_pareto_lastfm.png",
        "efficiency_pareto_yelp.png",
    ]

    here = os.path.dirname(os.path.abspath(__file__))
    part_paths = [os.path.join(here, p) for p in parts]

    missing = [p for p in part_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "缺少子图文件，请先运行四个子图脚本生成 png：\n" + "\n".join(missing)
        )

    # 读取图片并拼接为 1x4
    images = [mpimg.imread(p) for p in part_paths]

    fig, axes = plt.subplots(1, 4, figsize=(30, 6.5))
    for ax, im in zip(axes, images):
        ax.imshow(im)
        ax.axis("off")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.02)

    out_png = os.path.join(here, "efficiency_pareto.png")
    out_pdf = os.path.join(here, "efficiency_pareto.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
