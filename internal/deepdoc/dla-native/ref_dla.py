"""Reference DLA implementation (Python) — replicates deepdoc YOLOv10 forward + DLA adapter.

Used only to validate the Go prototype against a ground-truth reference.
Mirrors deepdoc/vision/layout_recognizer.py LayoutRecognizer4YOLOv10.preprocess/postprocess
and deepdoc/server/adapters/dla_adapter.py.
"""
import sys, json, os
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort

MODEL = os.path.join(
    os.environ.get("MODEL_DIR", "/home/shenyushi/workspace/ragflow/rag/res/deepdoc"),
    "layout.onnx")

YOLO_LABELS = ["title", "Text", "Reference", "Figure", "Figure caption",
               "Table", "Table caption", "Table caption", "Equation", "Figure caption"]
DLA_CLASS_MAP = {"title": 0, "text": 1, "reference": 2, "figure": 3, "figure caption": 4,
                 "table": 5, "table caption": 6, "equation": 8}


def nms(bboxes, scores, iou_thresh):
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = (y2 - y1) * (x2 - x1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[order[1:]] - overlaps)
        idx = np.where(ious <= iou_thresh)[0]
        order = order[idx + 1]
    return keep


def main():
    img_path = sys.argv[1]
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    arr = np.array(img)
    arr = np.array(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)).astype(np.float32)

    input_shape = (1024, 1024)
    shape = arr.shape[:2]
    r = min(input_shape[0] / shape[0], input_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = (input_shape[1] - new_unpad[0]) / 2.0, (input_shape[0] - new_unpad[1]) / 2.0
    arr = cv2.resize(arr, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    arr = cv2.copyMakeBorder(arr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    arr /= 255.0
    arr = arr.transpose(2, 0, 1)
    blob = arr[np.newaxis, :, :, :].astype(np.float32)
    scale_factor = np.array([shape[1] / new_unpad[0], shape[0] / new_unpad[1], dw, dh], dtype=np.float32)

    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    out = sess.run(["output0"], {"images": blob})[0]
    out = np.squeeze(out)

    thr = 0.08
    scores = out[:, 4]
    m = scores > thr
    out = out[m]
    scores = scores[m]
    class_ids = out[:, -1].astype(int)
    boxes = out[:, :4].astype(np.float32)
    boxes[:, [0, 2]] -= scale_factor[2]
    boxes[:, [1, 3]] -= scale_factor[3]
    boxes = boxes * np.array([scale_factor[0], scale_factor[1], scale_factor[0], scale_factor[1]], dtype=np.float32)

    result = []
    for cid in np.unique(class_ids):
        idx = np.where(class_ids == cid)[0]
        keep = nms(boxes[idx], scores[idx], 0.45)
        for k in keep:
            i = idx[k]
            label = YOLO_LABELS[int(class_ids[i])].lower()
            cls = DLA_CLASS_MAP.get(label)
            if cls is None:
                continue
            x0, y0, x1, y1 = [float(v) for v in boxes[i].tolist()]
            result.append([max(0.0, min(x0, W)), max(0.0, min(y0, H)),
                           max(0.0, min(x1, W)), max(0.0, min(y1, H)),
                           round(float(scores[i]), 4), float(cls)])
    print(json.dumps(result))


if __name__ == "__main__":
    main()
