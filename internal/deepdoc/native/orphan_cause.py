"""Definitive per-orphan cause analysis for the det gap.

Maps ALL Go/py intermediates into SOURCE coordinates (via the resize scale from
py_pred.json: sw/rw, sh/rh) and, for each golden (py_final) box with no Go
final match (IoU<0.5), decides the cause:

  go_comp  : does Go's connectedComponents even form a region there?
             (nearest go_comp center, source)
  go_cand  : does a Go pre-unclip candidate survive there, and does it pass
             the 0.5 score threshold? (nearest go_candidates.quad, source)
  go_final : is there a Go final box near it at all?

Classification:
  FILTERED   - go_cand present & score<0.5  (boxScoreFast divergence)
  SHIFTED    - go_cand present & score>=0.5 but go_final far (unclip/post-minAreaRect)
  NOCOMP     - no go_comp within 30px (grouping/component-extraction divergence)
  ELSE       - ambiguous

Usage:
  python orphan_cause.py
(needs the dumps from TestDumpStages + cmp_stages.py + final_compare.py)
"""
import json
import numpy as np


def ctr(q):
    return np.array(q, float).mean(axis=0)


def nearest(query, centers):
    d = np.linalg.norm(centers - query, axis=1)
    return int(d.argmin()), float(d.min())


def main():
    pred = json.load(open("/tmp/py_pred.json"))
    rw, rh, sw, sh = pred["rw"], pred["rh"], pred["sw"], pred["sh"]
    sx, sy = sw / rw, sh / rh

    go = json.load(open("/tmp/go_final.json"))["boxes"]
    gof = np.array([ctr(b["pts"]) for b in go])
    gcand = json.load(open("/tmp/go_candidates.json"))["cands"]
    gcq = np.array([ctr(c["quad"]) for c in gcand])      # source coords
    gcs = np.array([c["score"] for c in gcand])
    gcomp = json.load(open("/tmp/go_comps.json"))["comps"]
    gcc = np.array([ctr(c) for c in gcomp]) * np.array([sx, sy])  # -> source

    pyf = json.load(open("/tmp/py_final.json"))["boxes"]
    pc = np.array([ctr(b) for b in pyf])

    print(f"scale src/resized = ({sx:.3f}, {sy:.3f})  go_final={len(go)} py_final={len(pyf)}")
    rows = []
    for j in range(len(pyf)):
        # best IoU with go_final
        best = 0.0
        for i in range(len(go)):
            # rough: use center distance as IoU proxy for orphan detection
            pass
        # nearest go_final
        fi, fd = nearest(pc[j], gof) if len(gof) else (-1, 1e9)
        if fd < 30:
            rows.append((j, "matched-near", fd, -1, -1, -1))
            continue
        ki, kd = nearest(pc[j], gcq) if len(gcq) else (-1, 1e9)
        ci, cd = nearest(pc[j], gcc) if len(gcc) else (-1, 1e9)
        gscore = gcs[ki] if ki >= 0 else -1
        if cd >= 30:
            cause = "NOCOMP(grouping)"
        elif gscore < 0.5:
            cause = "FILTERED(score)"
        else:
            cause = "SHIFTED(unclip/post-mar)"
        rows.append((j, cause, fd, cd, kd, gscore))

    orphans = [r for r in rows if r[1] != "matched-near"]
    print(f"golden-orphans = {len(orphans)}")
    from collections import Counter
    c = Counter(r[1] for r in orphans)
    for k, v in c.items():
        print(f"  {k}: {v}")
    print("  detail (py#, cause, finalDist, compDist, candDist, goScore):")
    for r in orphans:
        print(f"    py#{r[0]:>3} {r[1]:28s} fd={r[2]:7.1f} compD={r[3]:7.1f} candD={r[4]:7.1f} goScore={r[5]:.3f}")


if __name__ == "__main__":
    main()
