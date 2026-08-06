"""Compare Go (DLA_DUMP_PRED) det dumps vs deepdoc dumps.

Go:   /tmp/go_blob.bin   /tmp/go_pred.bin   /tmp/go_blob_dims.txt
Ref:  /tmp/ref_blob.bin  /tmp/ref_pred.bin  /tmp/ref_blob_dims.txt

Localizes the residual DET difference to one of:
  (a) preprocessing blob mismatch (resize / normalize / channel order)
  (b) ONNX output (pred) mismatch given identical input
  (c) later contour extraction (if preds match but boxes differ)
"""
import numpy as np


def load(path, dims_path):
    rh, rw, sh, sw = (int(x) for x in open(dims_path).read().split())
    a = np.fromfile(path, dtype="<f4")
    return a, rh, rw, sh, sw


def chans(a, rh, rw):
    # blob flattened [1,3,rh,rw] -> 3 channels of [rh,rw]
    return [a[c * rh * rw:(c + 1) * rh * rw].reshape(rh, rw) for c in range(3)]


def report(name, go, ref, rh, rw):
    go = go.reshape(rh, rw)
    ref = ref.reshape(rh, rw)
    d = np.abs(go - ref)
    maxd = float(d.max())
    meand = float(d.mean())
    # fraction of pixels differing by more than a tiny epsilon
    frac = float((d > 1e-3).mean())
    # argmax of each
    gi = int(np.argmax(go)); ri = int(np.argmax(ref))
    gxy = (gi % rw, gi // rw); rxy = (ri % rw, ri // rw)
    print("  %-14s max|d|=%.6f  mean|d|=%.6f  frac>1e-3=%.4f  argmax go=%s ref=%s"
          % (name, maxd, meand, frac, gxy, rxy))
    return maxd, meand, frac


print("=== dims ===")
go_dims = open("/tmp/go_blob_dims.txt").read().split()
ref_dims = open("/tmp/ref_blob_dims.txt").read().split()
print("  go :", go_dims)
print("  ref:", ref_dims)
assert go_dims == ref_dims, "dims mismatch!"

rh = rw = None


def loadp(prefix):
    a, h, w, sh, sw = load("/tmp/%s_blob.bin" % prefix, "/tmp/%s_blob_dims.txt" % prefix)
    return a, h, w, sh, sw


go_blob, rh, rw, sh, sw = loadp("go")
ref_blob, _, _, _, _ = loadp("ref")
go_pred, h2, w2, _, _ = load("/tmp/go_pred.bin", "/tmp/go_pred_dims.txt")
ref_pred, _, _, _, _ = load("/tmp/ref_pred.bin", "/tmp/ref_pred_dims.txt")

print("\n=== BLOB: per-channel (0,1,2) Go vs Ref ===")
gc = chans(go_blob, rh, rw)
rc = chans(ref_blob, rh, rw)
for c in range(3):
    report("chan%d" % c, gc[c], rc[c], rh, rw)

print("\n=== BLOB: cross-channel check (is Go channel c == Ref channel p?) ===")
# Tests for a BGR<->RGB swap between the two pipelines.
for go_c in range(3):
    for ref_c in range(3):
        d = np.abs(gc[go_c] - rc[ref_c])
        if d.max() < 1e-3:
            print("  MATCH: Go chan%d == Ref chan%d (max|d|=%.6f)" % (go_c, ref_c, d.max()))

print("\n=== PRED: overall ===")
report("pred", go_pred, ref_pred, rh, rw)

# Where do the biggest pred differences live?
d = np.abs(go_pred.reshape(rh, rw) - ref_pred.reshape(rh, rw))
ys, xs = np.where(d > d.max() * 0.5)
if len(xs):
    print("  top-diff region x∈[%d,%d] y∈[%d,%d] (n=%d px)" %
          (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()), len(xs)))
