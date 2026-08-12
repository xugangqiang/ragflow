"""Validate find_contours_proto by monkeypatching cv2.findContours inside
deepdoc's TextDetector, then comparing the resulting final boxes to the
real cv2-oracle final boxes (py_final.json) with the same IoU logic as
final_compare.py.

If the orphan count drops to ~0, the algorithm is validated for porting to Go.
"""
import sys
import os
import json
import numpy as np
from PIL import Image

import cv2
import find_contours_proto as fc_mod

# Monkeypatch cv2.findContours to use our tracer.
_orig = cv2.findContours


def _patched(bitmap, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE):
    arr = (np.asarray(bitmap) > 0).astype(np.uint8)
    contours = fc_mod.find_contours(arr)
    # return format OpenCV 4.x: (contours, hierarchy)
    return contours, np.zeros((len(contours), 4), np.int32)


cv2.findContours = _patched

from deepdoc.vision.ocr import TextDetector

MODEL_DIR = os.environ.get("MODEL_DIR", "/home/shenyushi/codex-workspace/ragflow/rag/res/deepdoc")
img_path = sys.argv[1]


def iou(a, b):
    def area(p):
        p = np.array(p, float)
        return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) -
                         np.dot(p[:, 1], np.roll(p[:, 0], -1)))

    def inter(p, q):
        cp = [tuple(x) for x in p]
        for i in range(len(q)):
            a1, b1 = np.array(q[i]), np.array(q[(i + 1) % len(q)])
            e = (b1[1] - a1[1], a1[0] - b1[0], b1[0] * a1[1] - a1[0] * b1[1])
            ncp = []
            for j in range(len(cp)):
                cur = np.array(cp[j]); nxt = np.array(cp[(j + 1) % len(cp)])
                d1 = e[0] * cur[0] + e[1] * cur[1] + e[2]
                d2 = e[0] * nxt[0] + e[1] * nxt[1] + e[2]
                if d1 <= 0:
                    ncp.append(cp[j])
                if d1 * d2 < 0:
                    t = d1 / (d1 - d2)
                    ncp.append(tuple(cur + t * (nxt - cur)))
            cp = ncp
        return area(cp) if cp else 0.0

    ia = inter(a, b)
    return ia / (area(a) + area(b) - ia) if (area(a) + area(b) - ia) > 0 else 0.0


img = Image.open(img_path).convert("RGB")
det = TextDetector(MODEL_DIR)
dt_boxes, _ = det(np.array(img))
mine = [np.array(b, float) for b in dt_boxes]
print(f"MY-contours final boxes: {len(mine)}")

pyf = json.load(open("/tmp/py_final.json"))["boxes"]
py = [np.array(b, float) for b in pyf]
print(f"cv2 oracle final boxes:  {len(py)}")


def ctr(q):
    return q.mean(axis=0)


mc = np.array([ctr(q) for q in mine])
pc = np.array([ctr(q) for q in py])
matched = set()
orphans = 0
for j in range(len(py)):
    best_i, best_v = -1, 0.0
    for i in range(len(mine)):
        v = iou(py[j], mine[i])
        if v > best_v:
            best_i, best_v = i, v
    if best_v >= 0.5:
        matched.add(best_i)
    else:
        orphans += 1
print(f"golden-orphans (mine vs cv2 oracle): {orphans}")
print(f"  matched: {len(matched)}")
PYTHONPATH_HINT = None
