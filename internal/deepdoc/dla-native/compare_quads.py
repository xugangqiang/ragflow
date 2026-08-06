"""S1 diagnostic: compare the pipeline's pre-unclip min-area rects (dumped by
dbPostProcess under DLA_DUMP_QUADS -> /tmp/go_quads.json) against deepdoc's
pre-unclip boxes (testdata/contours.json "pre_box"), matched by center
proximity. A near-0 result proves the Go geometry (convexHull+minAreaRect) is
exact and the residual DET error lives in mask/contour extraction; a large
result would point back at minAreaRect.

Usage: python3 compare_quads.py [goprefix]
  goprefix defaults to "go"; use "gocv" after copying /tmp/go_quads.json ->
  /tmp/gocv_quads.json from a gocv run.
"""
import sys
import json
import math
import numpy as np

GO = sys.argv[1] if len(sys.argv) > 1 else "go"
FIX = "native/testdata/contours.json"

go = json.load(open("/tmp/%s_quads.json" % GO))
fix = json.load(open(FIX))
ref = [c["pre_box"] for c in fix["contours"]]


def center(q):
    xs = [p[0] for p in q]
    ys = [p[1] for p in q]
    return (sum(xs) / 4.0, sum(ys) / 4.0)


def diff(a, b):
    return max(max(abs(a[j][0] - b[j][0]), abs(a[j][1] - b[j][1])) for j in range(4))


used = set()
maxd = 0.0
sumd = 0.0
n = 0
for gq in go:
    gc = center(gq)
    best = None
    bd = 1e18
    for i, rq in enumerate(ref):
        if i in used:
            continue
        rc = center(rq)
        d = (gc[0] - rc[0]) ** 2 + (gc[1] - rc[1]) ** 2
        if d < bd:
            bd = d
            best = i
    if best is None:
        print("  unmatched go quad center", gc)
        continue
    used.add(best)
    d = diff(gq, ref[best])
    maxd = max(maxd, d)
    sumd += d
    n += 1

print("Go quads=%d  Ref pre_box=%d  matched=%d" % (len(go), len(ref), n))
print("PRE-UNCLIP diff vs deepdoc: max=%.4f px  mean=%.4f px" % (maxd, sumd / max(n, 1)))
print("(If this is ~0, geometry is exact; the 3px lives in mask/contour extraction.)" if n else "")
