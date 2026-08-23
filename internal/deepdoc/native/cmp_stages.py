"""Stage-by-stage det oracle for the Go port (cmp_stages.py).

Subclasses deepdoc's DBPostProcess so the SAME deepdoc code paths
(get_mini_boxes / box_score_fast / unclip — inherited verbatim) are used, but
boxes_from_bitmap is overridden to dump intermediates that mirror what the Go
pipeline writes under DLA_DUMP_*:

  /tmp/py_pred.json        — raw pred map (post-sigmoid, pre-threshold)
  /tmp/py_candidates.json  — {cands:[{quad, preQuad, score}]}, the post-geometry
                             pre-score-filter set, matching Go's
                             /tmp/go_candidates.json exactly.

Usage (after `go test -tags integration -run TestDumpStages`):
  PYTHONPATH=<ragflow根> MODEL_DIR=<...> python cmp_stages.py <img.png> [model_dir]
Then:
  python diff_stages.py
"""
import sys
import os
import json

import numpy as np
import cv2
from PIL import Image
from deepdoc.vision.ocr import TextDetector
from deepdoc.vision import postprocess as pp

MODEL_DIR = os.environ.get("MODEL_DIR", "/home/shenyushi/workspace/ragflow/rag/res/deepdoc")


class StageDBPostProcess(pp.DBPostProcess):
    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        bitmap = _bitmap
        height, width = bitmap.shape

        # Dump the segmentation mask (pred > thresh) for a direct Go/py seg
        # comparison — isolates whether the grouping divergence originates in
        # the thresholded map (i.e. the pred micro-difference) or in the
        # contour/component extraction itself.
        if not getattr(self, "_seg_dumped", False):
            self._seg_dumped = True
            with open("/tmp/py_seg.json", "w") as f:
                json.dump(
                    {
                        "h": int(height),
                        "w": int(width),
                        "seg": (bitmap.astype(np.uint8)).tolist(),
                    },
                    f,
                )

        outs = cv2.findContours(
            (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(outs) == 3:
            contours = outs[1]
        else:
            contours = outs[0]

        # Dump the raw pred map once (S0–S2 equivalence check on the Go side).
        if not getattr(self, "_pred_dumped", False):
            self._pred_dumped = True
            with open("/tmp/py_pred.json", "w") as f:
                json.dump(
                    {
                        "rh": int(height),
                        "rw": int(width),
                        "sh": int(dest_height),
                        "sw": int(dest_width),
                        "pred": pred.astype(np.float32).reshape(-1).tolist(),
                    },
                    f,
                )

        cands = []
        n = min(len(contours), self.max_candidates)
        for i in range(n):
            contour = contours[i]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            score = self.box_score_fast(pred, np.array(points).reshape(-1, 2))
            box = self.unclip(np.array(points), self.unclip_ratio).reshape(-1, 1, 2)
            box, sside2 = self.get_mini_boxes(box)
            if sside2 < self.min_size + 2:
                continue
            box = np.array(box)
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            cands.append(
                {
                    "quad": box.tolist(),
                    "preQuad": np.array(points).tolist(),
                    "score": float(score),
                }
            )

        with open("/tmp/py_candidates.json", "w") as f:
            json.dump({"cands": cands}, f)

        # Dump each findContours contour's BOUNDARY pixel set (resized coords,
        # +0.5 center offset to mirror Go's component points) for the same
        # cv2.minAreaRect localization test described in det.go.
        csets = []
        for contour in contours[:n]:
            s = (contour.reshape(-1, 2).astype(np.float64) + 0.5).tolist()
            csets.append(s)
        with open("/tmp/py_comps.json", "w") as f:
            json.dump({"w": int(width), "h": int(height), "comps": csets}, f)

        return np.array([c["quad"] for c in cands], dtype="int32"), [
            c["score"] for c in cands
        ]


def main():
    img_path = sys.argv[1]
    model_dir = sys.argv[2] if len(sys.argv) > 2 else MODEL_DIR

    img = Image.open(img_path).convert("RGB")
    det = TextDetector(model_dir)
    # Clone the production post-process params into the stage-dumping subclass.
    op = det.postprocess_op
    sp = StageDBPostProcess(
        thresh=op.thresh,
        box_thresh=op.box_thresh,
        max_candidates=op.max_candidates,
        unclip_ratio=op.unclip_ratio,
        use_dilation=getattr(op, "use_dilation", False),
        score_mode=op.score_mode,
        box_type=op.box_type,
    )
    det.postprocess_op = sp

    det(np.array(img))  # triggers boxes_from_bitmap -> dumps
    print("wrote /tmp/py_pred.json, /tmp/py_candidates.json for", img_path)


if __name__ == "__main__":
    main()
