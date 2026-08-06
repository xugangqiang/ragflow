"""Generate a fresh pyclipper oracle for the 15 real pre-unclip quads captured
from the gocv Go run (go_quads_pre.json). For each quad:
  - truncate to int64 exactly like Go's clipperOffset (math.Trunc)
  - run pyclipper JT_ROUND offset with delta = area*1.5/perimeter (Shapely)
  - record the integer offset polygon (point set) AND the pre-scale rect that
    deepdoc produces (cv2.minAreaRect + get_mini_boxes + scale-free), so the Go
    test can verify clipperOffset against the true oracle at sub-pixel.

Writes testdata/clipper_quads4.json (the trustworthy oracle).
"""
import json
import math
import sys
import numpy as np
import cv2
import pyclipper
from shapely.geometry import Polygon

pre = json.load(open("/tmp/go_quads_pre.json"))


def get_mini_boxes(contour):
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


def area_perim(box):
    poly = Polygon(box)
    return poly.area, poly.length


out = []
for boxf in pre:
    box = [[float(p[0]), float(p[1])] for p in boxf]
    # Go truncates to int64 before offsetting.
    ibox = [[int(math.trunc(p[0])), int(math.trunc(p[1]))] for p in box]
    a, per = area_perim(ibox)
    d = a * 1.5 / per
    off = pyclipper.PyclipperOffset()
    off.AddPath(ibox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    sol = off.Execute(d)
    poly = [[int(p[0]), int(p[1])] for p in sol[0]]
    # pre-scale rect (deepdoc's get_mini_boxes on the offset polygon)
    rect, _ = get_mini_boxes(np.array(poly, dtype="float32").reshape(-1, 1, 2))
    out.append({
        "box": ibox,
        "distance": d,
        "poly": [poly],
        "pre_scale": [[float(p[0]), float(p[1])] for p in rect],
    })

json.dump({"quads": out}, open("testdata/clipper_quads4.json", "w"), indent=1)
print("wrote %d quads, sample poly size=%d" % (len(out), len(out[0]["poly"][0])))
