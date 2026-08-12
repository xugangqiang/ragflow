"""Reference OCR-rec implementation (Python) — calls the real deepdoc model.

Used only to validate the Go prototype against a ground-truth reference. This
oracle invokes deepdoc's own TextRecognizer (the production recognition path,
including its dynamic-width resize_norm_img), so the pinned golden is
*production deepdoc output*, not a hand-written mirror. The recognized text
comes from real deepdoc; the wire score is fixed at 1.0 to match the Go
DocAnalyzer OCR-rec wire (internal/parser/deepdoc.go hard-codes 1.0).

Wire format: {"output": [[[[text, 1.0]]]]} (Go [][][][]any).

Usage:
  python3 ref_ocr_rec.py <image> [model_dir]
"""
import sys
import json
import os

import cv2
import numpy as np
from PIL import Image
from deepdoc.vision.ocr import TextRecognizer

MODEL_DIR = os.environ.get("MODEL_DIR", "/home/shenyushi/workspace/ragflow/rag/res/deepdoc")


def main():
    # All args but the last are image paths; the last is the model dir.
    args = sys.argv[1:]
    model_dir = args[-1]
    img_paths = args[:-1]

    rec = TextRecognizer(model_dir)
    bgrs = []
    for p in img_paths:
        img = Image.open(p).convert("RGB")
        bgrs.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    # One call with the whole list: TextRecognizer resizes every line to the
    # batch-wide max wh_ratio (floored at 320/48), so a narrow line inside a
    # wide batch is widened — exactly the production batch semantics Go's
    # RunOCRRecBatch must reproduce.
    res, _ = rec(bgrs)
    out = []
    for r in res:
        text = r[0]
        # Match the Go DocAnalyzer OCR-rec wire (score fixed at 1.0).
        out.append({"output": [[[[text, 1.0]]]]})
    # A single image keeps the standalone wire (matching the frozen per-line
    # golden); multiple images emit an ordered array for line-by-line batch
    # comparison.
    if len(out) == 1:
        print(json.dumps(out[0]))
    else:
        print(json.dumps(out))


if __name__ == "__main__":
    main()
