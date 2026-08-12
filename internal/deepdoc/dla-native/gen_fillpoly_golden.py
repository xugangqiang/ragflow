"""Generate a pixel-exact cv2.fillPoly golden for the pure-Go fillPoly.

For each det candidate's PRE-unclip quad (resized coords), replicate
DBPostProcess.box_score_fast's mask frame and let cv2.fillPoly rasterize it.
We only need the MASK (score = mean over mask, so mask-align => score-align
for ANY pred). Dump {mw,mh,quad(local),mask(local)} per case so a pure-Go
unit test can diff its fillPoly against cv2 bit-for-bit.

Usage: python3 gen_fillpoly_golden.py <stem> [model_dir]
Writes testdata/<stem>.fillpoly.golden.json
"""
import sys
import json
import os
import numpy as np
import cv2
from deepdoc.vision.ocr import TextDetector

MODEL_DIR = os.environ.get("MODEL_DIR", "/home/shenyushi/workspace/ragflow/rag/res/deepdoc")


def main():
    stem = sys.argv[1]
    model_dir = sys.argv[2] if len(sys.argv) > 2 else MODEL_DIR
    from PIL import Image
    img = np.array(Image.open(stem + ".png").convert("RGB"))
    det = TextDetector(model_dir)
    orig_post = det.postprocess_op
    captured = {}

    def fake_post(out_dict, shape_list):
        captured["maps"] = out_dict["maps"]
        captured["shape"] = shape_list
        return [{"points": []}]

    det.postprocess_op = fake_post
    det(img)
    pred = captured["maps"][0, 0]          # [H,W]
    H, W = pred.shape
    db = orig_post

    bitmap = (pred > db.thresh)
    outs = cv2.findContours((bitmap * 255).astype(np.uint8),
                            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = outs[1] if len(outs) == 3 else outs[0]

    cases = []
    for contour in contours[: db.max_candidates]:
        points, sside = db.get_mini_boxes(contour)
        if sside < db.min_size:
            continue
        quad = np.array(points, dtype=float)   # pre-unclip, resized coords
        xmin = int(np.clip(np.floor(quad[:, 0].min()), 0, W - 1))
        xmax = int(np.clip(np.ceil(quad[:, 0].max()), 0, W - 1))
        ymin = int(np.clip(np.floor(quad[:, 1].min()), 0, H - 1))
        ymax = int(np.clip(np.ceil(quad[:, 1].max()), 0, H - 1))
        mw, mh = xmax - xmin + 1, ymax - ymin + 1
        if mw <= 0 or mh <= 0:
            continue
        box = quad.copy()
        box[:, 0] -= xmin
        box[:, 1] -= ymin
        mask = np.zeros((mh, mw), dtype=np.uint8)
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        cases.append({
            "mw": mw, "mh": mh,
            "quad": box.tolist(),
            "mask": mask.flatten().tolist(),
        })
    out = stem + ".fillpoly.golden.json"
    with open(out, "w") as f:
        json.dump(cases, f)
    print("wrote %s : %d cases" % (out, len(cases)))


if __name__ == "__main__":
    main()
