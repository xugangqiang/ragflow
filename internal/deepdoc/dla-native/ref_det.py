"""Reference OCR-detection implementation (Python) — the oracle for the Go port.

Uses deepdoc's TextDetector directly so the output is byte-for-byte what the
production pipeline produces (DetResizeForTest -> NormalizeImage -> det.onnx ->
DBPostProcess(thresh=0.3, box_thresh=0.5, unclip_ratio=1.5) ->
filter_tag_det_res). Wire format matches deepdoc/server/adapters/ocr_adapter.py
detect mode: {"output": [[ [ [x,y]*4, ... ] ]]} (5-level: batch/page/quad/point/coord).

Usage:
  python3 ref_det.py <image> [model_dir]
"""
import sys
import json
import os
import numpy as np
from PIL import Image
from deepdoc.vision.ocr import TextDetector

MODEL_DIR = os.environ.get("MODEL_DIR", "/home/shenyushi/workspace/ragflow/rag/res/deepdoc")


def main():
    img_path = sys.argv[1]
    model_dir = sys.argv[2] if len(sys.argv) > 2 else MODEL_DIR

    img = Image.open(img_path).convert("RGB")
    det = TextDetector(model_dir)
    dt_boxes, _ = det(np.array(img))  # [N,4,2] after filter_tag_det_res (clockwise)
    quads = [b.tolist() for b in dt_boxes]
    # 5-level nesting matching Go [][][][][]float64
    print(json.dumps({"output": [[quads]]}))


if __name__ == "__main__":
    main()
