"""Reference DLA implementation (Python) — calls the real deepdoc model.

Used only to validate the Go prototype against a ground-truth reference. This
oracle invokes deepdoc's own LayoutRecognizer4YOLOv10 (preprocess -> onnx
inference -> postprocess) and re-maps the YOLOv10 labels to the Go DLA class
ids, so the pinned golden is *production deepdoc output*, not a hand-written
mirror. Wire format matches the Go DocAnalyzer:
{"bboxes": [[x0,y0,x1,y1,score,class_id], ...]}.

Usage:
  python3 ref_dla.py <image> [model_dir]
"""
import sys
import json
import os

from PIL import Image
from deepdoc.vision.layout_recognizer import LayoutRecognizer4YOLOv10

MODEL_DIR = os.environ.get("MODEL_DIR", "/home/shenyushi/workspace/ragflow/rag/res/deepdoc")

# OSS model label -> Go dlaClass index (mirrors dla_adapter.DLA_CLASS_MAP and
# dla.go dlaClassMap).
DLA_CLASS_MAP = {"title": 0, "text": 1, "reference": 2, "figure": 3, "figure caption": 4,
                 "table": 5, "table caption": 6, "equation": 8}


def main():
    img_path = sys.argv[1]
    model_dir = sys.argv[2] if len(sys.argv) > 2 else MODEL_DIR

    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    # LayoutRecognizer4YOLOv10 loads layout.onnx from its default model dir
    # (rag/res/deepdoc); the optional model_dir arg is accepted for parity with
    # the other oracles but the model location is fixed by deepdoc.
    _ = model_dir
    dla = LayoutRecognizer4YOLOv10("layout")
    # forward() returns raw Recognizer output (no OCR integration): a list
    # per image of {"type", "bbox": [x0,y0,x1,y1], "score"} in source pixels.
    raw = dla.forward([img], thr=0.2)[0]

    result = []
    for b in raw:
        label = b["type"].lower()
        cls = DLA_CLASS_MAP.get(label)
        if cls is None:
            continue
        x0, y0, x1, y1 = b["bbox"]
        result.append([
            max(0.0, min(float(x0), W)),
            max(0.0, min(float(y0), H)),
            max(0.0, min(float(x1), W)),
            max(0.0, min(float(y1), H)),
            float(b["score"]),
            float(cls),
        ])
    # Match the Go DocAnalyzer DLA wire: {"bboxes": [[x0,y0,x1,y1,score,class_id], ...]}.
    print(json.dumps({"bboxes": result}))


if __name__ == "__main__":
    main()
