from __future__ import annotations

from pathlib import Path

from typing import Iterable


def _require_pymupdf():
    try:
        import fitz  # type: ignore

        return fitz
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: pymupdf. Install via `pip install pymupdf` and rerun."
        ) from exc


def _pairs() -> list[list[str]]:
    return [
        [
            "hyperparameter_keeprate_recall_ndcg_mind.pdf",
            "hyperparameter_keeprate_recall_ndcg_ml1m.pdf",
            "hyperparameter_steps_recall_ndcg_mind.pdf",
            "hyperparameter_steps_recall_ndcg_ml1m.pdf",
            "hyperparameter_temp_recall_ndcg_mind.pdf",
            "hyperparameter_temp_recall_ndcg_ml1m.pdf",
        ],
        [
            "hyperparameter_keeprate_recall_ndcg_lastfm.pdf",
            "hyperparameter_keeprate_recall_ndcg_yelp.pdf",
            "hyperparameter_steps_recall_ndcg_lastfm.pdf",
            "hyperparameter_steps_recall_ndcg_yelp.pdf",
            "hyperparameter_temp_recall_ndcg_lastfm.pdf",
            "hyperparameter_temp_recall_ndcg_yelp.pdf",
        ],
    ]


def _resolve_existing(img_dir: Path, name: str) -> Path:
    """Prefer PDF; fallback to PNG with the same stem."""

    pdf_path = img_dir / name
    if pdf_path.exists():
        return pdf_path

    if pdf_path.suffix.lower() == ".pdf":
        png_path = pdf_path.with_suffix(".png")
        if png_path.exists():
            return png_path

    raise FileNotFoundError(f"Missing image: {pdf_path} (or {pdf_path.with_suffix('.png')})")


def _iter_grid_paths(img_dir: Path) -> list[list[Path]]:
    pairs = _pairs()
    return [[_resolve_existing(img_dir, name) for name in row] for row in pairs]


def _caption_specs() -> list[tuple[tuple[int, int], str]]:
    return [
        ((0, 1), "(a) The impact of retention rate hyperparameters"),
        ((2, 3), "(b) The impact of integration steps hyperparameters"),
        ((4, 5), "(c) The impact of temperature hyperparameters"),
    ]


def _as_points(inches: float) -> float:
    return inches * 72.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _auto_clip_rect_for(
    fitz,
    src_page,
    *,
    dpi: int = 120,
    threshold: int = 250,
    pad_pt: float = 2.0,
):
    """Estimate a tight clip rect by rasterizing and trimming near-white borders.

    This avoids accidentally cutting off titles / tick labels which are often
    very close to the PDF page boundary when the source was saved with tight
    bounding boxes.
    """

    rect = src_page.rect
    scale = dpi / 72.0
    pix = src_page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)

    w, h, n = pix.width, pix.height, pix.n
    samples = pix.samples

    def _row_has_ink(y: int) -> bool:
        base = y * w * n
        row = samples[base : base + w * n]
        for i in range(0, len(row), n):
            # background if all channels are near-white
            if not all(ch >= threshold for ch in row[i : i + n]):
                return True
        return False

    def _col_has_ink(x: int, y0: int, y1: int) -> bool:
        for y in range(y0, y1 + 1):
            i = (y * w + x) * n
            if not all(ch >= threshold for ch in samples[i : i + n]):
                return True
        return False

    top = 0
    while top < h and not _row_has_ink(top):
        top += 1

    bottom = h - 1
    while bottom >= 0 and not _row_has_ink(bottom):
        bottom -= 1

    # All white: fallback to full page.
    if top >= bottom:
        return rect

    left = 0
    while left < w and not _col_has_ink(left, top, bottom):
        left += 1

    right = w - 1
    while right >= 0 and not _col_has_ink(right, top, bottom):
        right -= 1

    if left >= right:
        return rect

    sx = rect.width / w
    sy = rect.height / h

    l = rect.x0 + left * sx
    r = rect.x0 + (right + 1) * sx
    t = rect.y0 + top * sy
    b = rect.y0 + (bottom + 1) * sy

    # Safety padding (points). Increase this if titles/ticks get cut.
    l = max(rect.x0, l - pad_pt)
    r = min(rect.x1, r + pad_pt)
    t = max(rect.y0, t - pad_pt)
    b = min(rect.y1, b + pad_pt)

    return fitz.Rect(l, t, r, b)


def compose_hyperparameter_grid_pdf(
    *,
    out_name: str = "hyperparameter_grid.pdf",
    cols: int = 6,
    rows: int = 2,
    margin_in: float = 0.10,
    col_gap_in: float = 0.05,
    group_cols: int = 2,
    group_gap_in: float = 0.15,
    row_gap_in: float = 0.02,
    caption_gap_in: float = 0.06,
    caption_fontsize: float = 12.0,
    trim_left: float = 0.00,
    trim_right: float = 0.00,
    trim_top: float = 0.00,
    trim_bottom: float = 0.00,
    auto_trim: bool = True,
    auto_trim_dpi: int = 120,
    auto_trim_threshold: int = 250,
    auto_trim_pad_pt: float = 4.0,
    preview_png: bool = False,
    preview_dpi: int = 200,
) -> None:
    """Compose 12 hyperparameter plots into one big PDF by tiling source PDFs.

    If a source PDF is missing, it will fallback to the PNG with the same stem.

    Layout: 2 rows × 6 cols
    - Columns 0-1: keeprate (MIND, ML-1M / LastFM, Yelp2018)
    - Columns 2-3: steps    (MIND, ML-1M / LastFM, Yelp2018)
    - Columns 4-5: temp     (MIND, ML-1M / LastFM, Yelp2018)

        Spacing (all *_in are inches):
        - margin_in: page margin around the whole grid
        - col_gap_in: gap between adjacent columns inside a group
        - group_gap_in: extra gap between groups (each group has group_cols columns)
        - row_gap_in: gap between the two rows

        Cropping:
        - If auto_trim=True (default), we estimate a tight clip box by rasterizing each
            source page and trimming near-white borders (safer; avoids cutting labels).
        - Otherwise, trim_* crops away whitespace from each tile (fractions of width/height).
    """

    fitz = _require_pymupdf()
    img_dir = Path(__file__).resolve().parent
    grid = _iter_grid_paths(img_dir)

    # Open one source (prefer first PDF tile) to infer a reasonable tile aspect.
    first_path = next((p for p in (grid[0][0],) if p.exists()), grid[0][0])
    if first_path.suffix.lower() == ".pdf":
        src0 = fitz.open(first_path)
        src0_page = src0[0]
        src_rect = src0_page.rect
        src0.close()
        tile_aspect = src_rect.height / src_rect.width
    else:
        tile_aspect = 0.42  # heuristic; PNG fallback should be rare

    # Convert inches -> PDF points (1 in = 72 pt)
    margin = _as_points(margin_in)
    col_gap = _as_points(col_gap_in)
    group_gap = _as_points(group_gap_in)
    row_gap = _as_points(row_gap_in)
    caption_gap = _as_points(caption_gap_in)

    if cols <= 0 or rows <= 0:
        raise ValueError("cols and rows must be positive")
    if group_cols <= 0:
        raise ValueError("group_cols must be positive")
    if cols % group_cols != 0:
        raise ValueError("cols must be divisible by group_cols")

    # Add extra spacing after the last column of each group.
    # Example: cols=6, group_cols=2 => groups are [0,1],[2,3],[4,5], so boundaries are {1,3}.
    group_boundaries_after: set[int] = {
        k * group_cols - 1 for k in range(1, cols // group_cols)
    }

    def _x0_for_col(c: int) -> float:
        x = margin
        for i in range(c):
            x += tile_w
            # Normal gap between every pair of adjacent columns.
            x += col_gap
            if i in group_boundaries_after:
                # Extra gap between groups (e.g., between col 1 and 2, between col 3 and 4).
                x += group_gap
        return x

    # Choose a compact wide page (good for LaTeX \linewidth figures).
    # Default target width ~= 12 inches.
    page_w = _as_points(12.0)
    n_group_gaps = max(0, (cols // group_cols) - 1)
    # Total horizontal gaps = normal column gaps + extra group gaps.
    total_x_gaps = (cols - 1) * col_gap + n_group_gaps * group_gap
    tile_w = (page_w - 2 * margin - total_x_gaps) / cols
    tile_h = tile_w * tile_aspect
    captions_h = _as_points(0.27)
    page_h = 2 * margin + rows * tile_h + (rows - 1) * row_gap + caption_gap + captions_h

    out_doc = fitz.open()
    out_page = out_doc.new_page(width=page_w, height=page_h)

    def _tile_rect(r: int, c: int):
        x0 = _x0_for_col(c)
        y0 = margin + r * (tile_h + row_gap)
        return fitz.Rect(x0, y0, x0 + tile_w, y0 + tile_h)

    def _clip_rect_for(src_page):
        rect = src_page.rect
        if auto_trim:
            return _auto_clip_rect_for(
                fitz,
                src_page,
                dpi=auto_trim_dpi,
                threshold=auto_trim_threshold,
                pad_pt=auto_trim_pad_pt,
            )
        l = rect.x0 + rect.width * _clamp(trim_left, 0.0, 0.45)
        r = rect.x1 - rect.width * _clamp(trim_right, 0.0, 0.45)
        t = rect.y0 + rect.height * _clamp(trim_top, 0.0, 0.45)
        b = rect.y1 - rect.height * _clamp(trim_bottom, 0.0, 0.45)
        return fitz.Rect(l, t, r, b)

    # Place tiles.
    for r in range(rows):
        for c in range(cols):
            src_path = grid[r][c]
            target = _tile_rect(r, c)

            if src_path.suffix.lower() == ".pdf":
                src_doc = fitz.open(src_path)
                src_page = src_doc[0]
                clip = _clip_rect_for(src_page)
                out_page.show_pdf_page(target, src_doc, 0, clip=clip)
                src_doc.close()
            else:
                # PNG fallback: embed as an image
                out_page.insert_image(target, filename=str(src_path))

    # Group captions: centered under each 2-column block.
    cap_y0 = margin + rows * tile_h + (rows - 1) * row_gap + caption_gap
    cap_y1 = cap_y0 + captions_h
    for (c0, c1), text in _caption_specs():
        x0 = _x0_for_col(c0)
        x1 = _x0_for_col(c1) + tile_w
        rect = fitz.Rect(x0, cap_y0, x1, cap_y1)
        out_page.insert_textbox(
            rect,
            text,
            fontsize=caption_fontsize,
            fontname="Times-Roman",
            align=1,  # center
        )

    out_path = img_dir / out_name
    out_doc.save(out_path)
    out_doc.close()

    if preview_png:
        doc = fitz.open(out_path)
        page = doc[0]
        mat = fitz.Matrix(preview_dpi / 72.0, preview_dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_path = out_path.with_suffix(".png")
        pix.save(png_path)
        doc.close()
        print(f"Saved: {out_path.name} / {png_path.name}")
    else:
        print(f"Saved: {out_path.name}")


if __name__ == "__main__":
    compose_hyperparameter_grid_pdf(preview_png=True)
