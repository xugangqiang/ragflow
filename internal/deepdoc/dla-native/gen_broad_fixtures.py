"""Generate broad-coverage det fixtures: multi-language / multi-layout pages
rendered from real_pdfs, plus synthesized degenerate samples.

Companion to generate_crops.py (which builds page0/table0/line0). This script
broadens the regression surface so rare paths (non-English scripts, table/figure
heavy layouts, single-line / noise / solid / gradient / low-contrast / skewed
inputs) are exercised, not just the single page0.jpg page.

Run with the workspace venv (deepdoc + pypdfium2 resolvable):
  cd /home/shenyushi/workspace/ragflow
  /home/shenyushi/workspace/ragflow/.venv/bin/python \\
    /home/shenyushi/codex-workspace/ragflow/internal/deepdoc/dla-native/gen_broad_fixtures.py

Writes *.jpg into the Go module's testdata/. Golden JSON is produced separately
by ref_det.py (see HANDOFF §5 / the make-golden loop in CI), so this script
only owns the *inputs*.
"""
import os
import sys
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pypdfium2 as pdfium

SCALE = 3.0
REAL_PDFS = "/home/shenyushi/codex-workspace/ragflow/internal/deepdoc/parser/pdf/testdata/real_pdfs"
OUTDIR = "/home/shenyushi/codex-workspace/ragflow/internal/deepdoc/dla-native/testdata"
os.makedirs(OUTDIR, exist_ok=True)

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def render_pdf_page(pdf_name, page, stem):
    path = os.path.join(REAL_PDFS, pdf_name)
    doc = pdfium.PdfDocument(path)
    n = len(doc)
    if page >= n:
        page = 0
    pil = doc[page].render(scale=SCALE).to_pil().convert("RGB")
    doc.close()
    out = os.path.join(OUTDIR, stem + ".jpg")
    pil.save(out)
    print(f"{stem}.jpg  ({pil.size[0]}x{pil.size[1]})  from {pdf_name}[{page}]/{n}")
    return pil


def draw_text(draw, xy, text, font, fill):
    draw.text(xy, text, font=font, fill=fill)


def synth_degenerate():
    W, H = 900, 700

    # single large char on white — expects ~1 box.
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, 360)
    draw_text(d, (W // 2 - 120, H // 2 - 200), "A", f, (0, 0, 0))
    img.save(os.path.join(OUTDIR, "deg_single_char.jpg"))
    print("deg_single_char.jpg")

    # single line of text — expects ~1 box.
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, 48)
    draw_text(d, (60, H // 2 - 30), "The quick brown fox jumps", f, (0, 0, 0))
    img.save(os.path.join(OUTDIR, "deg_single_line.jpg"))
    print("deg_single_line.jpg")

    # same line, rotated 20deg — expects ~1 box (rotation path).
    rot = img.rotate(20, expand=True, fillcolor=(255, 255, 255))
    rc = Image.new("RGB", (W, H), (255, 255, 255))
    rc.paste(rot, ((W - rot.size[0]) // 2, (H - rot.size[1]) // 2))
    rc.save(os.path.join(OUTDIR, "deg_skewed.jpg"))
    print("deg_skewed.jpg")

    # salt & pepper noise on white — expects 0 boxes.
    arr = np.full((H, W, 3), 255, dtype=np.uint8)
    rng = np.random.default_rng(7)
    mask = rng.random((H, W)) < 0.01
    arr[mask] = 0
    mask2 = rng.random((H, W)) < 0.01
    arr[mask2] = 255
    Image.fromarray(arr).save(os.path.join(OUTDIR, "deg_noise.jpg"))
    print("deg_noise.jpg")

    # solid mid-gray — expects 0 boxes.
    Image.new("RGB", (W, H), (128, 128, 128)).save(
        os.path.join(OUTDIR, "deg_solid.jpg"))
    print("deg_solid.jpg")

    # horizontal gradient 0..255 — expects 0 boxes.
    grad = np.linspace(0, 255, W, dtype=np.uint8)
    grad = np.repeat(grad[np.newaxis, :], H, axis=0)
    Image.fromarray(np.stack([grad, grad, grad], axis=2)).save(
        os.path.join(OUTDIR, "deg_gradient.jpg"))
    print("deg_gradient.jpg")

    # light-gray text on white, near threshold — expects ~0 boxes.
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, 48)
    draw_text(d, (60, H // 2 - 30), "faint sample text", f, (205, 205, 205))
    img.save(os.path.join(OUTDIR, "deg_low_contrast.jpg"))
    print("deg_low_contrast.jpg")


def main():
    # --- multi-language / multi-layout real pages (page 0 each) ---
    render_pdf_page(
        "JaColBERTv2.5-Optimising Multi-Vector Retrievers to Create "
        "State-of-the-Art Japanese Retrievers with Constrained Resources.pdf",
        0, "mp_jp_p0")
    render_pdf_page(
        "asset-recovery-services-sd-zh-tw.pdf", 0, "mp_zhtw_p0")
    render_pdf_page(
        "GB 51249-2017 建筑钢结构防火技术规范.pdf", 0, "mp_cn_std_p0")
    render_pdf_page(
        "20240815-华福证券-海光信息-688041.SH-中报略超预告中值_新增适配AI大模型通义千问_4页_467kb.pdf",
        0, "mp_sec_p0")
    render_pdf_page(
        "2510.07233v1.pdf", 0, "mp_en_dense_p0")

    # --- synthesized degenerate samples ---
    synth_degenerate()

    print("done; run ref_det.py on each new .jpg to regenerate goldens")


if __name__ == "__main__":
    main()
