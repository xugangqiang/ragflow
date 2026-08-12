"""IoU comparison of FINAL det boxes: Go (RunDet) vs Python oracle (ref_det).

Go final boxes are dumped by TestDumpStages to /tmp/go_final.json.
Python final boxes come from deepdoc TextDetector (filter_tag_det_res applied).

For each golden (Python) box we find the best-IoU Go match and classify it:
  - matched (IoU>=0.5): same box detected by both
  - misshapen: a Go box is nearby (<30px center) but IoU<0.5 -> the pre-unclip
    region IS detected by Go but unclip/scale shifts it off the golden box
  - missing-region: no Go FINAL box within 30px -> did Go even form a
    pre-unclip candidate there? We look up Go's nearest pre-unclip candidate
    (go_candidates.quad, source coords) to decide:
        * present, score>=thresh -> detected but pushed away by unclip (geometry)
        * present, score<thresh   -> filtered out by the score threshold
        * absent                  -> grouping/component-extraction divergence

Python's per-box score isn't returned by TextDetector, so we recover it by
matching py_final -> py_candidates (same source-coord quad centers) and compare
with Go's final (== pre-unclip) score for matched boxes.

Usage:
  python final_compare.py <img.png> [model_dir]
"""
import sys
import os
import json

import numpy as np
from PIL import Image
from deepdoc.vision.ocr import TextDetector

MODEL_DIR = os.environ.get("MODEL_DIR", "/home/shenyushi/workspace/ragflow/rag/res/deepdoc")
BOX_THRESH = 0.5  # detBoxThresh


def iou(a, b):
    """IoU of two 4-point quads (shoelace-based polygon intersection)."""
    def area(p):
        p = np.array(p, float)
        return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) -
                         np.dot(p[:, 1], np.roll(p[:, 0], -1)))

    def inter(p, q):
        cp = [tuple(x) for x in p]
        for i in range(len(q)):
            a1, b1 = np.array(q[i]), np.array(q[(i + 1) % len(q)])
            edge = (b1[1] - a1[1], a1[0] - b1[0], b1[0] * a1[1] - a1[0] * b1[1])
            ncp = []
            for j in range(len(cp)):
                cur = np.array(cp[j])
                nxt = np.array(cp[(j + 1) % len(cp)])
                d1 = edge[0] * cur[0] + edge[1] * cur[1] + edge[2]
                d2 = edge[0] * nxt[0] + edge[1] * nxt[1] + edge[2]
                if d1 <= 0:
                    ncp.append(cp[j])
                if d1 * d2 < 0:
                    t = d1 / (d1 - d2)
                    ncp.append(tuple(cur + t * (nxt - cur)))
            cp = ncp
        if not cp:
            return 0.0
        return area(cp)

    ia = inter(a, b)
    return ia / (area(a) + area(b) - ia) if (area(a) + area(b) - ia) > 0 else 0.0


def center(q):
    return np.array(q, float).mean(axis=0)


def nearest(query, centers):
    """Return (idx, dist) of nearest center to query."""
    d = np.linalg.norm(centers - query, axis=1)
    i = int(d.argmin())
    return i, float(d[i])


def main():
    img_path = sys.argv[1]
    model_dir = sys.argv[2] if len(sys.argv) > 2 else MODEL_DIR

    g = json.load(open("/tmp/go_final.json"))
    go = [(np.array(b["pts"], float), float(b.get("score", 0))) for b in g["boxes"]]
    gc = np.array([center(q) for q, _ in go])

    # Python final boxes (deepdoc, post filter_tag_det_res).
    img = Image.open(img_path).convert("RGB")
    det = TextDetector(model_dir)
    dt_boxes, _ = det(np.array(img))
    py = [np.array(b, float) for b in dt_boxes]
    pc = np.array([center(q) for q in py])
    with open("/tmp/py_final.json", "w") as f:
        json.dump({"boxes": [b.tolist() for b in py]}, f)
    print(f"final boxes: go={len(go)} py={len(py)}")

    # Recover Python per-box scores: match py_final -> py_candidates (source
    # coords, post-geometry pre-score-filter set).
    pcand = json.load(open("/tmp/py_candidates.json"))["cands"]
    pcand_quad = np.array([center(c["quad"]) for c in pcand])
    py_score = []
    for j in range(len(py)):
        i, _ = nearest(pc[j], pcand_quad)
        py_score.append(float(pcand[i]["score"]))

    # Go pre-unclip candidates (source-coord quad + pre-unclip score) for the
    # orphan-cause lookup.
    gcand = json.load(open("/tmp/go_candidates.json"))["cands"]
    gcand_quad = np.array([center(c["quad"]) for c in gcand])
    gcand_score = [float(c["score"]) for c in gcand]

    matched = set()
    pmatched = set()
    misshapen, missing = [], []
    score_diffs = []
    for j in range(len(py)):
        best_i, best_v = -1, 0.0
        for i in range(len(go)):
            v = iou(py[j], go[i][0])
            if v > best_v:
                best_i, best_v = i, v
        dist = float(np.linalg.norm(gc[best_i] - pc[j])) if best_i >= 0 else 1e9
        if best_v >= 0.5:
            matched.add(best_i)
            pmatched.add(j)
            score_diffs.append(abs(go[best_i][1] - py_score[j]))
        elif dist < 30:
            misshapen.append((j, best_i, best_v, dist))
        else:
            missing.append((j, best_i, best_v, dist))

    print(f"matched={len(matched)}  golden-orphans={len(py)-len(pmatched)}")
    if score_diffs:
        sd = np.array(score_diffs)
        print(f"  matched score |Δ|: mean={sd.mean():.3f} max={sd.max():.3f} "
              f">0.05:{int((sd>0.05).sum())}  >0.1:{int((sd>0.1).sum())}")
    print(f"  misshapen (nearby Go final, IoU<0.5): {len(misshapen)}")
    print(f"  missing-region (no Go final <30px):   {len(missing)}")

    for tag, lst in (("MISSHAPEN", misshapen), ("MISSING", missing)):
        for j, i, v, d in lst[:25]:
            q = py[j]
            w = float(np.linalg.norm(q[0] - q[1]))
            h = float(np.linalg.norm(q[0] - q[3]))
            # Did Go form a pre-unclip candidate near this region at all?
            gi, gd = nearest(pc[j], gcand_quad)
            gs = gcand_score[gi]
            cause = ("detected+shifted" if gd < 30
                     else ("filtered(score<%.2f)" % BOX_THRESH if gs < BOX_THRESH
                           else "grouping(absent)"))
            gscore = go[i][1] if i >= 0 else -1
            print(f"    [{tag}] py#{j} bestIoU={v:.3f} finalDist={d:.1f}px "
                  f"size=({w:.0f}x{h:.0f}) pyScore={py_score[j]:.3f} "
                  f"goFinalScore(nearest)={gscore:.3f} | "
                  f"goPreCandDist={gd:.1f}px goPreScore={gs:.3f} -> {cause}")


if __name__ == "__main__":
    main()
