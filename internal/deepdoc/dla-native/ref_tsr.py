"""Reference TSR implementation (Python) — replicates the Go tsr.go pipeline.

Used only to validate the Go prototype against a ground-truth reference.
Mirrors deepdoc/vision/recognizer.py postprocess (scale_factor, xywh->xyxy,
per-class NMS 0.2) and deepdoc/vision/table_structure_recognizer.py alignTSR,
plus the wire mapping in deepdoc/server/adapters/tsr_adapter.py.

Usage:
  python3 ref_tsr.py <image> [model_dir]
"""
import sys
import json
import os
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort

MODEL_DIR = os.environ.get("MODEL_DIR", "/home/shenyushi/workspace/ragflow/rag/res/deepdoc")
TSR_LABELS = ["table", "table column", "table row",
              "table column header", "table projected row header", "table spanning cell"]
TSR_CLASS_MAP = {l: i for i, l in enumerate(TSR_LABELS)}


def nms(bboxes, scores, iou_thr):
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = (y2 - y1) * (x2 - x1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[order[1:]] - overlaps)
        idx = np.where(ious <= iou_thr)[0]
        order = order[idx + 1]
    return keep


def main():
    img_path = sys.argv[1]
    model_dir = sys.argv[2] if len(sys.argv) > 2 else MODEL_DIR

    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    arr = np.array(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR).astype(np.float32)

    resized = cv2.resize(bgr, (640, 640), interpolation=cv2.INTER_LINEAR)
    blob = resized.transpose(2, 0, 1) / 255.0
    blob = blob[np.newaxis, :, :, :].astype(np.float32)
    sf = np.array([W / 640.0, H / 640.0], dtype=np.float32)

    sess = ort.InferenceSession(os.path.join(model_dir, "tsr.onnx"),
                                providers=["CPUExecutionProvider"])
    out = sess.run(["output0"], {"images": blob})[0]
    out = np.squeeze(out).T  # [8400, 11]

    score_thr = 0.2
    cands = []
    for a in range(out.shape[0]):
        cls_scores = out[a, 4:11]
        best = float(cls_scores.max())
        best_cls = int(cls_scores.argmax())
        if best <= score_thr or best_cls >= len(TSR_LABELS):
            continue
        cx = out[a, 0] * sf[0]
        cy = out[a, 1] * sf[1]
        hw = out[a, 2] * sf[0] * 0.5
        hh = out[a, 3] * sf[1] * 0.5
        cands.append({
            "x0": float(cx - hw), "y0": float(cy - hh),
            "x1": float(cx + hw), "y1": float(cy + hh),
            "score": best, "cls": best_cls,
        })

    boxes = []
    by_class = {}
    for i, c in enumerate(cands):
        by_class.setdefault(c["cls"], []).append(i)
    for cls, idxs in by_class.items():
        sub = np.array([[cands[i]["x0"], cands[i]["y0"], cands[i]["x1"], cands[i]["y1"]]
                        for i in idxs], dtype=np.float32)
        sc = np.array([cands[i]["score"] for i in idxs], dtype=np.float32)
        for k in nms(sub, sc, 0.2):
            i = idxs[k]
            boxes.append({
                "label": TSR_LABELS[cls],
                "score": round(cands[i]["score"], 4),
                "x0": round(cands[i]["x0"], 2), "x1": round(cands[i]["x1"], 2),
                "top": round(cands[i]["y0"], 2), "bottom": round(cands[i]["y1"], 2),
            })

    # alignTSR
    left_vals = [b["x0"] for b in boxes if ("row" in b["label"] or "header" in b["label"])]
    right_vals = [b["x1"] for b in boxes if ("row" in b["label"] or "header" in b["label"])]
    top_vals = [b["top"] for b in boxes if b["label"] == "table column"]
    bot_vals = [b["bottom"] for b in boxes if b["label"] == "table column"]
    if left_vals:
        left, right = min(left_vals), max(right_vals)
        for b in boxes:
            if "row" in b["label"] or "header" in b["label"]:
                if b["x0"] > left:
                    b["x0"] = left
                if b["x1"] < right:
                    b["x1"] = right
    if top_vals:
        top, bot = min(top_vals), max(bot_vals)
        for b in boxes:
            if b["label"] == "table column":
                if b["top"] > top:
                    b["top"] = top
                if b["bottom"] < bot:
                    b["bottom"] = bot

    result = []
    for b in boxes:
        cls = TSR_CLASS_MAP.get(b["label"])
        if cls is None:
            continue
        # Clamp into image bounds (mirrors tsr_adapter.py).
        x0 = min(max(b["x0"], 0.0), float(W))
        x1 = min(max(b["x1"], 0.0), float(W))
        top = min(max(b["top"], 0.0), float(H))
        bot = min(max(b["bottom"], 0.0), float(H))
        result.append([x0, top, x1, bot, b["score"], float(cls)])
    print(json.dumps(result))


if __name__ == "__main__":
    main()
