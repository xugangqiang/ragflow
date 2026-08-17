"""Reference raster-path oracle (Python) — validates the Go prototype through
the *production* render path, not just against frozen PNG goldens.

The dla-native proof establishes "given the same raster image bytes, Go == Python"
(EQUIVALENCE.md Scope/§Boundary). But the production pipeline does NOT feed
pre-rendered PNGs to the recognizers: the Go server rasterizes PDF pages with
pdfium (RenderPage @ 216 DPI) and the Python deepdoc pipeline rasterizes with
pdfplumber (page.to_image(resolution=72*zoomin, antialias=True) with zoomin=3 =>.
216 DPI). This script closes the remaining gap by reproducing deepdoc's OWN
rasterization (216 DPI, antialias) for a given PDF page, then running the real
deepdoc recognizers (DLA / Det / TSR) — exactly what the live deepdoc_server
would do over that page. The Go side rasterizes the same PDF page with pdfium at
the same DPI; the two box sets are then compared directly in source-pixel
coordinates. If they match within the documented floors, the "same-bytes-in
assumption" is no longer an assumption — it is measured.

This is the empirical backing for:
  - Reviewer gap #1: end-to-end raster alignment (DLA / Det / TSR).
  - Reviewer gap #2: TSR floor on full-page real tables (large/complex tables).

Usage (one task per invocation, JSON to stdout):
  uv run python3 ref_raster.py <pdf_path> <page_idx> <task> [model_dir]
    task in {dla, det, tsr}
  page_idx is 0-based (pdfium convention) so it matches the Go side's pageIdx.

Wire formats match the Go DocAnalyzer / dla-native oracles:
  dla -> {"bboxes": [[x0,y0,x1,y1,score,class_id], ...]}
  tsr -> {"bboxes": [[x0,top,x1,bot,score,class_id], ...]}
  det -> {"output": [[ [ [x,y]*4, ... ] ]]}
"""
import sys
import json
import os

import pdfplumber
from PIL import Image

# Keep imports lazy so a missing optional recognizer does not break the others.
MODEL_DIR = os.environ.get(
    "MODEL_DIR", "/home/shenyushi/workspace/ragflow/rag/res/deepdoc"
)

# deepdoc default zoomin=3 => 72*3 = 216 DPI, matching pdfium's dpi=216 path.
ZOOMIN = 3
DPI = 72 * ZOOMIN

DLA_CLASS_MAP = {"title": 0, "text": 1, "reference": 2, "figure": 3,
                 "figure caption": 4, "table": 5, "table caption": 6,
                 "equation": 8}
TSR_CLASS_MAP = {"table": 0, "table column": 1, "table row": 2,
                 "table column header": 3, "table projected row header": 4,
                 "table spanning cell": 5}


def render_page(pdf_path, page_idx):
    """Render PDF page at 216 DPI using deepdoc's own pdfplumber path."""
    with pdfplumber.open(pdf_path) as pdf:
        if page_idx < 0 or page_idx >= len(pdf.pages):
            raise IndexError(
                f"page_idx {page_idx} out of range (doc has {len(pdf.pages)} pages)"
            )
        page = pdf.pages[page_idx]
        # deepdoc's PdfParser.__images__ uses page.to_image(resolution=72*zoomin,
        # antialias=True). We use .original (no OCR annotation overlay) so the
        # rendered pixels match what the recognizers actually consume.
        img = page.to_image(resolution=DPI, antialias=True).original.convert("RGB")
    return img


def run_dla(img):
    from deepdoc.vision.layout_recognizer import LayoutRecognizer4YOLOv10

    W, H = img.size
    dla = LayoutRecognizer4YOLOv10("layout")
    raw = dla.forward([img], thr=0.2)[0]
    result = []
    for b in raw:
        cls = DLA_CLASS_MAP.get(b["type"].lower())
        if cls is None:
            continue
        x0, y0, x1, y1 = b["bbox"]
        result.append([
            max(0.0, min(float(x0), W)), max(0.0, min(float(y0), H)),
            max(0.0, min(float(x1), W)), max(0.0, min(float(y1), H)),
            float(b["score"]), float(cls),
        ])
    return {"bboxes": result}


def run_det(img):
    from deepdoc.vision.ocr import TextDetector

    det = TextDetector(MODEL_DIR)
    dt_boxes, _ = det(np_array(img))
    quads = [b.tolist() for b in dt_boxes]
    return {"output": [[quads]]}


def run_tsr(img):
    from deepdoc.vision.table_structure_recognizer import TableStructureRecognizer

    W, H = img.size
    tsr = TableStructureRecognizer()
    tables = tsr([img], thr=0.2)
    result = []
    for elem in tables[0]:
        cls = TSR_CLASS_MAP.get(elem["label"])
        if cls is None:
            continue
        x0 = max(0.0, min(float(elem["x0"]), W))
        x1 = max(0.0, min(float(elem["x1"]), W))
        top = max(0.0, min(float(elem["top"]), H))
        bot = max(0.0, min(float(elem["bottom"]), H))
        result.append([x0, top, x1, bot, float(elem["score"]), float(cls)])
    return {"bboxes": result}


def np_array(img):
    import numpy as np

    return np.array(img)


def main():
    if len(sys.argv) < 4:
        sys.stderr.write(
            "usage: ref_raster.py <pdf_path> <page_idx> <dla|det|tsr> [model_dir]\n"
        )
        sys.exit(2)
    pdf_path = sys.argv[1]
    page_idx = int(sys.argv[2])
    task = sys.argv[3]
    if len(sys.argv) > 4:
        global MODEL_DIR
        MODEL_DIR = sys.argv[4]

    img = render_page(pdf_path, page_idx)
    if task == "dla":
        out = run_dla(img)
    elif task == "det":
        out = run_det(img)
    elif task == "tsr":
        out = run_tsr(img)
    else:
        sys.stderr.write(f"unknown task {task}\n")
        sys.exit(2)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
