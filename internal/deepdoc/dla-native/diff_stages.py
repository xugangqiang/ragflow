"""Per-stage diff: Go pipeline intermediates vs the Python oracle.

Reads the dumps produced by `go test -tags integration -run TestDumpStages`
(/tmp/go_*.json) and `python cmp_stages.py` (/tmp/py_*.json), and reports:

  [S2 pred]   max|Δ| of the raw pred map. ~0 ⇒ decode+preprocess+inference
              are identical; the divergence is purely post-processing.
  [S_group]   component/contour count + box-for-box match of the
              post-geometry, pre-score-filter candidates. Reports the max
              preQuad (geometry) Δ, score Δ, and final-quad Δ on matched
              boxes, plus the unmatched (go_only / py_only) sets.

py_only boxes are the golden orphans Go missed; go_only are Go's extra boxes.
"""
import json
import numpy as np


def load_cands(path):
    with open(path) as f:
        return json.load(f)["cands"]


def prequad_center(c):
    return np.array(c["preQuad"], dtype=float).mean(axis=0)


def main():
    # ---- S2: pred map ----
    try:
        gp = json.load(open("/tmp/go_pred.json"))
        pyp = json.load(open("/tmp/py_pred.json"))
        gpred = np.array(gp["pred"], dtype=np.float32).reshape(gp["rh"], gp["rw"])
        pypred = np.array(pyp["pred"], dtype=np.float32).reshape(pyp["rh"], pyp["rw"])
        if gpred.shape != pypred.shape:
            print(f"[S2 pred] SHAPE MISMATCH go={gpred.shape} py={pypred.shape}")
        else:
            d = np.abs(gpred - pypred)
            maxd = float(d.max())
            mean = float(d.mean())
            n01 = int((d > 0.01).sum())
            n05 = int((d > 0.05).sum())
            n10 = int((d > 0.10).sum())
            tot = d.size
            print(f"[S2 pred] shape={gpred.shape} max|Δ|={maxd:.6e} mean|Δ|={mean:.6e}")
            print(f"          outlier pixels: >0.01={n01} ({100*n01/tot:.3f}%)  "
                  f">0.05={n05}  >0.10={n10}")
            if n10 < 50:
                print("          => decode+preprocess+inference ESSENTIALLY identical "
                      "(few edge pixels differ; not the orphan driver)")
            else:
                print("          => DIVERGENCE in decode/preprocess/inference!")
    except Exception as e:  # noqa: BLE001
        print("[S2 pred] compare skipped:", e)

    # ---- S_group/S_geom: candidate sets ----
    go = load_cands("/tmp/go_candidates.json")
    py = load_cands("/tmp/py_candidates.json")
    print(f"\n[S_group/S_geom] candidate count: go={len(go)} py={len(py)}")

    gc = np.array([prequad_center(c) for c in go])
    pc = np.array([prequad_center(c) for c in py])

    matched = set()
    pmatched = set()
    rows = []
    for i in range(len(go)):
        d = np.linalg.norm(pc - gc[i], axis=1)
        j = int(d.argmin())
        if d[j] < 15 and j not in pmatched:
            matched.add(i)
            pmatched.add(j)
            gq = np.array(go[i]["preQuad"], float)
            pq = np.array(py[j]["preQuad"], float)
            pdiff = float(np.abs(gq - pq).max())
            sd = abs(go[i]["score"] - py[j]["score"])
            gf = np.array(go[i]["quad"], float)
            pf = np.array(py[j]["quad"], float)
            fd = float(np.abs(gf - pf).max())
            rows.append((i, j, pdiff, sd, fd, float(d[j])))

    go_only = [i for i in range(len(go)) if i not in matched]
    py_only = [j for j in range(len(py)) if j not in pmatched]
    print(f"  matched={len(matched)} go_only={len(go_only)} py_only={len(py_only)}")

    if rows:
        pdiff_max = max(r[2] for r in rows)
        sd_max = max(r[3] for r in rows)
        fd_max = max(r[4] for r in rows)
        print(
            f"  matched boxes: max preQuad Δ={pdiff_max:.3f}px  "
            f"max score Δ={sd_max:.4f}  max finalQuad Δ={fd_max:.3f}px"
        )

    def side_info(c):
        q = np.array(c["preQuad"], float)
        e = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)
        return float(e.min()), float(e.max())

    def nearest_go_center(j):
        c = pc[j]
        dd = np.linalg.norm(gc - c, axis=1)
        return float(dd.min())

    print("\n  py_only (GOLDEN ORPHANS Go missed) — (minSide,maxSide,score,nearestGoCenter):")
    for j in py_only:
        ms, xs = side_info(py[j])
        print(f"    [{j:3d}] minSide={ms:7.1f} maxSide={xs:8.1f} score={py[j]['score']:.3f} "
              f"nearestGo={nearest_go_center(j):.1f}px")
    print("\n  go_only (Go EXTRA boxes) — (minSide,maxSide,score):")
    for i in go_only:
        ms, xs = side_info(go[i])
        print(f"    [{i:3d}] minSide={ms:7.1f} maxSide={xs:8.1f} score={go[i]['score']:.3f}")

    # Smoking-gun test for contours-vs-components: if py_only are predominantly
    # small boxes (inner-hole contours that findContours emits but connected
    # components merges), that confirms the grouping divergence is the cause.
    if py_only:
        py_small = [j for j in py_only if side_info(py[j])[1] < 20]
        go_small = [i for i in go_only if side_info(go[i])[1] < 20]
        print(f"\n  py_only small(maxSide<20): {len(py_small)}/{len(py_only)}   "
              f"go_only small: {len(go_small)}/{len(go_only)}")


if __name__ == "__main__":
    main()
