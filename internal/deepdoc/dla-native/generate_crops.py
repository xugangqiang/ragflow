"""Generate realistic TSR / OCR-rec test crops from a real PDF page.

Run with the workspace venv (where deepdoc can resolve its model dir):
  cd /home/shenyushi/workspace/ragflow
  /home/shenyushi/workspace/ragflow/.venv/bin/python \
    /home/shenyushi/codex-workspace/ragflow/internal/deepdoc/dla-native/generate_crops.py

Writes into the Go module's testdata/:
  page0.jpg   full rendered page  (DLA + whole-page TSR input)
  table0.jpg  cropped table region (TSR input)
  line0.jpg   cropped single text line (OCR-rec input)
"""
import os
import sys
import numpy as np
from PIL import Image
import pypdfium2 as pdfium

from deepdoc.vision import LayoutRecognizer, OCR

SCALE = 3.0
PDF = "/home/shenyushi/codex-workspace/ragflow/test/benchmark/test_docs/Doc1.pdf"
OUTDIR = "/home/shenyushi/codex-workspace/ragflow/internal/deepdoc/dla-native/testdata"
os.makedirs(OUTDIR, exist_ok=True)


def render_page():
    doc = pdfium.PdfDocument(PDF)
    pil = doc[0].render(scale=SCALE).to_pil().convert("RGB")
    doc.close()
    return pil


def main():
    pil = render_page()
    W, H = pil.size
    pil.save(os.path.join(OUTDIR, "page0.jpg"))
    print("page0.jpg", (W, H))

    # --- Table crop via DLA layout ---
    detr = LayoutRecognizer("layout")
    lyt = detr.forward([pil], thr=0.8)[0]
    tables = [b for b in lyt if b["type"] == "table"]
    if tables:
        bb = tables[0]["bbox"]  # [x0, y0, x1, y1]
        x0, y0, x1, y1 = [int(round(v)) for v in bb]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        crop = pil.crop((x0, y0, x1, y1))
        crop.save(os.path.join(OUTDIR, "table0.jpg"))
        print("table0.jpg", crop.size)
    else:
        print("no table detected on page 0")

    # --- Single text-line crop via OCR detection ---
    ocr = OCR()
    boxes = ocr(np.array(pil))
    if boxes:
        quad = np.array(boxes[0][0]).astype(int)
        xs, ys = quad[:, 0], quad[:, 1]
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        crop = pil.crop((x0, y0, x1, y1))
        crop.save(os.path.join(OUTDIR, "line0.jpg"))
        print("line0.jpg", crop.size)
    else:
        print("no text line detected on page 0")


if __name__ == "__main__":
    main()
