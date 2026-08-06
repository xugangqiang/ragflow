package native

// det_helpers.go — small sorting / rasterization helpers for the pure-Go DB
// post-process (det.go).

import (
	"math"
	"sort"
)

// sortPts orders points by x then y (used by the monotone-chain convex hull).
func sortPts(p []pt) {
	sort.Slice(p, func(i, j int) bool {
		if p[i].X != p[j].X {
			return p[i].X < p[j].X
		}
		return p[i].Y < p[j].Y
	})
}

// sortPtsByX is a STABLE sort by x (mirrors Python sorted(..., key=lambda x: x[0]),
// which getMiniBoxes relies on for tie-breaking).
func sortPtsByX(p []pt) {
	sort.SliceStable(p, func(i, j int) bool {
		return p[i].X < p[j].X
	})
}

// sortIntsByCount orders roots by descending pixel count.
func sortIntsByCount(roots []int, counts map[int]int) {
	sort.Slice(roots, func(i, j int) bool {
		return counts[roots[i]] > counts[roots[j]]
	})
}

// fillPoly rasterizes a polygon (float vertices, already shifted into mask
// space) into a bool mask using the even-odd scanline rule, mirroring the
// inside test cv2.fillPoly performs for a simple (convex) quad.
func fillPoly(mask []bool, mw, mh int, poly [4]pt) {
	// Auto-detect winding and build edge list.
	type edge struct{ y0, y1, x0, x1 float64 }
	var edges []edge
	n := len(poly)
	for i := 0; i < n; i++ {
		a, b := poly[i], poly[(i+1)%n]
		if a.Y == b.Y {
			continue // horizontal edges don't span scanlines
		}
		edges = append(edges, edge{y0: a.Y, y1: b.Y, x0: a.X, x1: b.X})
	}
	for sy := 0; sy < mh; sy++ {
		y := float64(sy) + 0.5
		var xs []float64
		for _, e := range edges {
			if y >= math_min(e.y0, e.y1) && y < math_max(e.y0, e.y1) {
				t := (y - e.y0) / (e.y1 - e.y0)
				xs = append(xs, e.x0+t*(e.x1-e.x0))
			}
		}
		sort.Float64s(xs)
		for k := 0; k+1 < len(xs); k += 2 {
			xStart := int(math.Ceil(xs[k] - 0.5))
			xEnd := int(math.Floor(xs[k+1] - 0.5))
			for x := xStart; x <= xEnd; x++ {
				if x >= 0 && x < mw {
					mask[sy*mw+x] = true
				}
			}
		}
	}
}

func math_min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}
func math_max(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
