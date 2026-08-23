"""Prototype: Moore-neighbor (Suzuki-Abe style) contour tracing matching
cv2.findContours(RETR_LIST) region separation, for validation before porting
to Go. Returns list of contours, each a list of (x,y) boundary points.

Goal: reproduce cv2.findContours' component SETS (the regions it separates),
since downstream uses convexHull+minAreaRect on the boundary points.
"""
import numpy as np


def find_contours(bin_img):
    """bin_img: 0/1 2D array. Returns list of contours (each a list of [x,y])."""
    h, w = bin_img.shape
    # pad with 0 border (OpenCV processes with a 1px bg border)
    m = np.zeros((h + 2, w + 2), dtype=np.int32)
    m[1:h + 1, 1:w + 1] = (np.asarray(bin_img) > 0).astype(np.int32)
    # visited markers: 0 bg, 1 fg unvisited, >=2 visited contour id
    visited = m.copy()
    H, W = m.shape
    contours = []
    nbd = 2

    # 8 neighbors in clockwise order starting from "up"
    NB = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

    def next_clockwise(b, cur):
        # find index of b in NB relative to cur, scan clockwise for first fg
        # b is the backtrack pixel (relative offset from cur)
        bi = None
        for k, d in enumerate(NB):
            if d == (b[0], b[1]):
                bi = k
                break
        if bi is None:
            bi = 7
        for step in range(8):
            k = (bi + 1 + step) % 8
            nr, nc = cur[0] + NB[k][0], cur[1] + NB[k][1]
            if 0 <= nr < H and 0 <= nc < W and m[nr, nc] == 1:
                return (NB[k][0], NB[k][1]), (nr, nc)
        return None, None

    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if visited[r, c] != 1:
                continue
            # outer border start: left is bg (0), current is fg
            is_outer = visited[r, c - 1] == 0
            # hole border start: top is bg and right is bg (and current fg, not outer)
            is_hole = (not is_outer) and visited[r - 1, c] == 0 and visited[r, c + 1] == 0
            if not (is_outer or is_hole):
                continue
            # start tracing
            start = (r, c)
            if is_outer:
                back = (0, -1)  # left (background)
            else:
                back = (-1, 0)  # up (background)
            cur = start
            contour = []
            first = True
            while True:
                bdir, nxt = next_clockwise(back, cur)
                if nxt is None:
                    break
                nr, nc = nxt
                contour.append([nc - 1, nr - 1])  # back to original coords
                # mark visited
                if visited[nr, nc] == 1:
                    visited[nr, nc] = nbd
                # closing test: returned to start with same backtrack
                if (not first) and nr == start[0] and nc == start[1]:
                    # need also that the next clockwise from back is bg
                    b2, _ = next_clockwise(bdir, (nr, nc))
                    if b2 is not None:
                        break
                first = False
                back = (-bdir[0], -bdir[1])  # came from opposite of found dir
                cur = (nr, nc)
                if len(contour) > H * W:
                    break
            if len(contour) >= 3:
                contours.append(np.array(contour, np.int32).reshape(-1, 1, 2))
                nbd += 1
            # mark start visited regardless
            visited[r, c] = nbd
    return contours


if __name__ == "__main__":
    import json
    py = json.load(open("/tmp/py_seg.json"))
    H, W = py["h"], py["w"]
    pseg = np.array(py["seg"], np.uint8).reshape(H, W)
    cnts = find_contours(pseg)
    print("prototype contours:", len(cnts))
    # compare region at py#43
    pyc = json.load(open("/tmp/py_candidates.json"))["cands"][43]["preQuad"]
    cx, cy = int(np.mean([p[0] for p in pyc])), int(np.mean([p[1] for p in pyc]))
    # does any prototype contour contain (cx,cy)?
    hit = -1
    for i, ct in enumerate(cnts):
        pts = np.array(ct)
        if (pts[:, 0] == cx).any() and (pts[:, 1] == cy).any():
            hit = i
            break
        # cheaper: bbox check
        if pts[:, 0].min() <= cx <= pts[:, 0].max() and pts[:, 1].min() <= cy <= pts[:, 1].max():
            # refine: point-in-polygon cheap via min distance
            if np.linalg.norm(pts - np.array([cx, cy]), axis=1).min() < 3:
                hit = i
                break
    print("py#43 contained in prototype contour:", hit)
