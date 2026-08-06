"""Reference OCR-rec implementation (Python) — replicates the Go ocr_rec.go pipeline.

Used only to validate the Go prototype against a ground-truth reference.
Mirrors deepdoc/vision/ocr.py TextRecognizer.resize_norm_img and
deepdoc/vision/postprocess.py CTCLabelDecode, plus the wire mapping in
deepdoc/server/adapters/ocr_adapter.py (recognize mode).

Usage:
  python3 ref_ocr_rec.py <image> [model_dir]
"""
import sys
import json
import os
import math
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort

MODEL_DIR = os.environ.get("MODEL_DIR", "/home/shenyushi/workspace/ragflow/rag/res/deepdoc")
REC_H, REC_W = 48, 320


def load_char_dict(model_dir):
    with open(os.path.join(model_dir, "ocr.res"), encoding="utf-8") as f:
        lines = f.read().split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    chars = ["blank"] + lines + [" "]
    return chars


def resize_norm_img(img, max_wh_ratio):
    h, w = img.shape[:2]
    ratio = w / float(h)
    resized_w = int(math.ceil(REC_H * ratio))
    if resized_w > REC_W:
        resized_w = REC_W
    resized = cv2.resize(img, (resized_w, REC_H)).astype(np.float32)
    resized = resized.transpose(2, 0, 1) / 255.0
    resized = (resized - 0.5) / 0.5
    pad = np.zeros((3, REC_H, REC_W), dtype=np.float32)
    pad[:, :, :resized_w] = resized
    return pad


def ctc_decode(preds, chars):
    idx = preds.argmax(axis=1)
    prob = preds.max(axis=1)
    text = []
    probs = []
    prev = -1
    for t in range(len(idx)):
        if idx[t] == 0:
            prev = 0
            continue
        if idx[t] != prev:
            if idx[t] < len(chars):
                text.append(chars[idx[t]])
                probs.append(float(prob[t]))
        prev = idx[t]
    score = 1.0 if not probs else float(np.mean(probs))
    return "".join(text), round(score, 4)


def main():
    img_path = sys.argv[1]
    model_dir = sys.argv[2] if len(sys.argv) > 2 else MODEL_DIR

    img = Image.open(img_path).convert("RGB")
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    ratio = w / float(h)
    max_wh_ratio = ratio

    blob = resize_norm_img(bgr, max_wh_ratio)[np.newaxis, :, :, :].astype(np.float32)
    sess = ort.InferenceSession(os.path.join(model_dir, "rec.onnx"),
                                providers=["CPUExecutionProvider"])
    out = sess.run(["softmax_11.tmp_0"], {"x": blob})[0]
    out = np.squeeze(out)  # [seq_len, vocab]

    chars = load_char_dict(model_dir)
    text, score = ctc_decode(out, chars)
    # 4-level nesting matching Go [][][][]any: batch -> page -> items -> pair
    print(json.dumps({"output": [[[[text, 1.0]]]]}))


if __name__ == "__main__":
    main()
