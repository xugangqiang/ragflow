"""Reference TSR implementation (Python) — calls the real deepdoc model.

Used only to validate the Go prototype against a ground-truth reference. This
oracle invokes deepdoc's own TableStructureRecognizer (preprocess -> onnx
inference -> postprocess -> alignTSR) and re-maps the labels to the Go TSR
class ids, so the pinned golden is *production deepdoc output*, not a
hand-written mirror. Wire format matches the Go DocAnalyzer:
{"bboxes": [[x0,y0,x1,y1,score,class_id], ...]}.

Usage:
  python3 ref_tsr.py <image> [model_dir]
"""
import sys
import json
import os

from PIL import Image
from deepdoc.vision.table_structure_recognizer import TableStructureRecognizer

MODEL_DIR = os.environ.get("MODEL_DIR", "/home/shenyushi/workspace/ragflow/rag/res/deepdoc")

# OSS model label -> Go tsrLabels index (mirrors tsr_adapter.TSR_CLASS_MAP and
# tsr.go tsrClassMap).
TSR_CLASS_MAP = {"table": 0, "table column": 1, "table row": 2,
                 "table column header": 3, "table projected row header": 4,
                 "table spanning cell": 5}


def main():
    img_path = sys.argv[1]
    model_dir = sys.argv[2] if len(sys.argv) > 2 else MODEL_DIR

    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    _ = model_dir
    tsr = TableStructureRecognizer()
    # __call__ runs the full pipeline including alignTSR (mean/median when more
    # than 4 rows/columns, else min/max). Returns a list per image of
    # {"label", "x0", "x1", "top", "bottom", "score"} in source pixels.
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
    # Match the Go DocAnalyzer TSR wire: {"bboxes": [[x0,top,x1,bot,score,class_id], ...]}.
    print(json.dumps({"bboxes": result}))


if __name__ == "__main__":
    main()
