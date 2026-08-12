"""Localize the det grouping divergence: component SET vs minAreaRect ALGORITHM.

Both Go (connectedComponents -> convexHull -> Go minAreaRect) and Python
(cv2.findContours -> cv2.minAreaRect) ultimately pick a rotated rectangle for a
foreground region. This script asks two independent questions under ONE
algorithm (cv2.minAreaRect) so the cause is unambiguous:

  Q1 (sets differ?):  cv2.minAreaRect(go_filled_set)  vs
                      cv2.minAreaRect(py_contour_set)
      matched by centroid. Small center distance  => the two extract the SAME
      pixel region; divergence is NOT grouping. Large distance => the regions
      themselves differ.

  Q2 (algorithm differs?):  Go's own pre-unclip quad (from go_candidates.json,
      which is Go minAreaRect of the convex hull)  vs
      cv2.minAreaRect(go_filled_set)  for the SAME Go component.
      Large distance => Go's rotating-calipers minAreaRect disagrees with cv2
      on an identical point set.

Usage (after TestDumpStages + cmp_stages.py have written the dumps):
  python loc_compare.py
"""
import json
import numpy as np
import cv2


def mar_center(pts):
    """cv2.minAreaRect center for a point set (Nx2)."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
    return np.array([cx, cy]), (w, h), ang


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    go = load("/tmp/go_comps.json")
    py = load("/tmp/py_comps.json")
    gc = load("/tmp/go_candidates.json")

    go_comps = go["comps"]
    py_comps = py["comps"]

    # Go's own pre-unclip quad centers (resized coords), in the SAME order as
    # go_comps (both produced from the same comps slice in dbPostProcess).
    go_own = []
    for c in gc["cands"]:
        pq = np.asarray(c["preQuad"]).reshape(-1, 2)
        go_own.append(pq.mean(axis=0))

    # Precompute Go cv2 minAreaRect centers (for ALL components).
    go_cv = np.asarray([mar_center(c)[0] for c in go_comps])

    # Match Go comps -> nearest Python contour by cv2 center distance (Q1).
    py_cv = np.asarray([mar_center(c)[0] for c in py_comps])

    n_match = 0
    set_diffs = []
    algo_diffs = []
    for i, gc_c in enumerate(go_cv):
        d = np.linalg.norm(py_cv - gc_c, axis=1)
        j = int(d.argmin())
        set_dist = d[j]
        if set_dist < 30:
            n_match += 1
            set_diffs.append(set_dist)
        # Q2: Go's OWN pre-unclip quad (surviving candidates only) vs cv2 on
        # the same component. Match by spatial proximity, NOT by index, because
        # go_own holds only size/score-surviving comps while go_cv holds all.
        # Only count a genuine same-region match (nearest go_own within 10px)
        # to avoid attributing a wrong-region match to the algorithm.
        if len(go_own):
            do = np.linalg.norm(np.asarray(go_own) - gc_c, axis=1)
            k = int(do.argmin())
            if do[k] < 10:
                algo_diffs.append(float(do[k]))

    set_diffs = np.array(set_diffs)
    algo_diffs = np.array(algo_diffs)

    print(f"go comps={len(go_comps)}  py contours={len(py_comps)}")
    print(f"matched (set<30px): {n_match}")
    print("--- Q1: do the extracted REGIONS differ? (cv2 on go vs cv2 on py) ---")
    if len(set_diffs):
        print(f"  set center dist: max={set_diffs.max():.1f}  mean={set_diffs.mean():.1f}  "
              f">10px:{int((set_diffs>10).sum())}  >20px:{int((set_diffs>20).sum())}")
    print("--- Q2: does Go's minAreaRect algorithm diverge from cv2 on the SAME set? ---")
    if len(algo_diffs):
        print(f"  go-own vs cv2(go): max={algo_diffs.max():.1f}  mean={algo_diffs.mean():.1f}  "
              f">10px:{int((algo_diffs>10).sum())}  >20px:{int((algo_diffs>20).sum())}")

    verdict = []
    if len(set_diffs) and set_diffs.mean() > 5:
        verdict.append("SETS DIFFER (grouping) — regions extracted are not the same")
    else:
        verdict.append("sets ~equal (grouping OK)")
    if len(algo_diffs) and algo_diffs.mean() > 5:
        verdict.append("Go minAreaRect DIVERGES from cv2 on same set (algorithm bug)")
    else:
        verdict.append("Go minAreaRect ~equals cv2 on same set")
    print("VERDICT:", " | ".join(verdict))


if __name__ == "__main__":
    main()
