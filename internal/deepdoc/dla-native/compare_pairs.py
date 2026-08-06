"""Generalized det tensor comparator.

Usage:
  python3 compare_pairs.py <goprefix> <refprefix>

Compares /tmp/<goprefix>_blob.bin / _pred.bin / _blob_dims.txt against
/tmp/<refprefix>_blob.bin / _pred.bin / _blob_dims.txt.

localizes the DET difference: blob (preprocessing) vs pred (ONNX output) vs
binary mask (if masks differ, contour extraction diverges downstream).
"""
import sys
import numpy as np


def load(prefix):
    a, h, w, sh, sw = loadraw(prefix)
    return a, h, w, sh, sw


def loadraw(prefix):
    dimpath = "/tmp/%s_blob_dims.txt" % prefix
    rh, rw, sh, sw = (int(x) for x in open(dimpath).read().split())
    a = np.fromfile("/tmp/%s_blob.bin" % prefix, dtype="<f4")
    return a, rh, rw, sh, sw


def chans(a, rh, rw):
    return [a[c * rh * rw:(c + 1) * rh * rw].reshape(rh, rw) for c in range(3)]


def report(name, go, ref, rh, rw):
    go = go.reshape(rh, rw); ref = ref.reshape(rh, rw)
    d = np.abs(go - ref)
    maxd = float(d.max()); meand = float(d.mean())
    frac = float((d > 1e-3).mean())
    print("  %-14s max|d|=%.6f mean|d|=%.6f frac>1e-3=%.4f" % (name, maxd, meand, frac))
    return maxd


go_p, ref_p = sys.argv[1], sys.argv[2]
go_blob, rh, rw, sh, sw = load(go_p)
ref_blob, _, _, _, _ = load(ref_p)
# pred dims match blob dims (rh,rw) for our dumps
_, h2, w2, _, _ = loadraw(ref_p.replace("ref", "ref") if False else ref_p)
gp, _, _, _, _ = loadraw(go_p)
rp, _, _, _, _ = loadraw(ref_p)
# pred files named <prefix>_pred.bin but dims reuse blob dims
gp = np.fromfile("/tmp/%s_pred.bin" % go_p, dtype="<f4")
rp = np.fromfile("/tmp/%s_pred.bin" % ref_p, dtype="<f4")

print("=== BLOB per-channel (Go '%s' vs Ref '%s') ===" % (go_p, ref_p))
gc, rc = chans(go_blob, rh, rw), chans(ref_blob, rh, rw)
for c in range(3):
    report("chan%d" % c, gc[c], rc[c], rh, rw)

print("=== PRED ===")
report("pred", gp, rp, rh, rw)

# binary masks at detThresh=0.3
gm = gp.reshape(rh, rw) > 0.3
rm = rp.reshape(rh, rw) > 0.3
print("  Go fg=%d Ref fg=%d  mask-diff px=%d (%.4f%%)" % (
    int(gm.sum()), int(rm.sum()), int((gm != rm).sum()), (gm != rm).mean() * 100))
