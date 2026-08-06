"""Oracle: run deepdoc's DBPostProcess.boxes_from_bitmap on a precomputed pred
(the same bit-exact pred the Go gocv path uses) and dump the intermediate
pre-unclip and post-unclip quads (in RESIZED coordinates, before scaling) plus
the final boxes (scaled to source), so they can be diffed box-for-box against
the Go dumps (/tmp/go_quads_pre.json, /tmp/go_quads_post.json).

This isolates exactly which stage introduces any residual DET error:
  pre-unclip  : findContours -> minAreaRect(contour)
  post-unclip : unclip(minAreaRect) -> get_mini_boxes (re-rect) -> scale

Uses cv2 4.10 (the deepdoc venv) so the geometry is the true oracle.
"""
import json
import os
import sys
import numpy as np
import cv2


class DBPostProcess:
    def __init__(self, thresh=0.3, box_thresh=0.5, max_candidates=1000,
                 unclip_ratio=1.5, min_size=3):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = min_size

    def boxes_from_bitmap(self, pred, mask, dest_w, dest_h, capture):
        h, w = mask.shape
        contours, _ = cv2.findContours((mask * 255).astype(np.uint8),
                                        cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        n = min(len(contours), self.max_candidates)
        boxes, scores = [], []
        for i in range(n):
            contour = contours[i]
            pts, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            capture["pre"].append([[float(p[0]), float(p[1])] for p in pts])
            pts = np.array(pts)
            score = self.box_score_fast(pred, pts.reshape(-1, 2))
            if self.box_thresh > score:
                continue
            box = self.unclip(pts, self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            capture["post"].append([[float(p[0]), float(p[1])] for p in box])
            box = np.array(box)
            box[:, 0] = np.clip(np.round(box[:, 0] / w * dest_w), 0, dest_w)
            box[:, 1] = np.clip(np.round(box[:, 1] / h * dest_h), 0, dest_h)
            boxes.append(box.astype("int32").tolist())
            scores.append(float(score))
        return boxes, scores

    def unclip(self, box, r):
        from shapely.geometry import Polygon
        import pyclipper
        poly = Polygon(box)
        d = poly.area * r / poly.length
        off = pyclipper.PyclipperOffset()
        off.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        return np.array(off.Execute(d))

    def get_mini_boxes(self, contour):
        bb = cv2.minAreaRect(contour)
        pts = sorted(list(cv2.boxPoints(bb)), key=lambda x: x[0])
        i1, i2, i3, i4 = 0, 1, 2, 3
        if pts[1][1] > pts[0][1]:
            i1, i4 = 0, 1
        else:
            i1, i4 = 1, 0
        if pts[3][1] > pts[2][1]:
            i2, i3 = 2, 3
        else:
            i2, i3 = 3, 2
        return [pts[i1], pts[i2], pts[i3], pts[i4]], min(bb[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = int(np.clip(np.floor(box[:, 0].min()), 0, w - 1))
        xmax = int(np.clip(np.ceil(box[:, 0].max()), 0, w - 1))
        ymin = int(np.clip(np.floor(box[:, 1].min()), 0, h - 1))
        ymax = int(np.clip(np.ceil(box[:, 1].max()), 0, h - 1))
        m = np.zeros((ymax - ymin + 1, xmax - xmin + 1), np.uint8)
        box[:, 0] -= xmin
        box[:, 1] -= ymin
        cv2.fillPoly(m, box.reshape(1, -1, 2).astype("int32"), 1)
        return float(cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], m)[0])


def main():
    pred_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ref_pred.bin"
    dims_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/ref_pred_dims.txt"
    rh, rw, sh, sw = [int(x) for x in open(dims_path).read().split()]
    pred = np.fromfile(pred_path, dtype="<f4").reshape(rh, rw)

    cap = {"pre": [], "post": []}
    p = DBPostProcess()
    mask = (pred > 0.3).astype(np.uint8)
    boxes, scores = p.boxes_from_bitmap(pred, mask, sw, sh, cap)

    out = {
        "pre": cap["pre"],
        "post": cap["post"],
        "final": boxes,
        "scores": scores,
        "dims": [rh, rw, sh, sw],
    }
    with open("/tmp/ref_det_quads.json", "w") as f:
        json.dump(out, f)
    print("contours(pre)=%d post=%d final=%d  src=%dx%d resize=%dx%d"
          % (len(cap["pre"]), len(cap["post"]), len(boxes), sw, sh, rw, rh))


if __name__ == "__main__":
    main()
