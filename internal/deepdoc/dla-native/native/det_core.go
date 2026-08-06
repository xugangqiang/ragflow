package native

// det_core.go — OCR text detection (DB) shared core.
//
// Build-tag split:
//   - det.go        (!gocv) : pure-Go geometry (connected components,
//                   rotating-calipers minAreaRect, scanline fillPoly).
//   - det_gocv.go   (gocv)  : OpenCV geometry (gocv.FindContours /
//                   MinAreaRect / BoxPoints), preprocessing via gocv.Resize.
//
// This file holds everything both paths need: the entry point, types, the
// true round-offset unclip (Clipper JT_ROUND equivalent), and the wire format.
// Under either build exactly one of det.go / det_gocv.go is compiled, so the
// package-level detPreprocess / dbPostProcess that RunDet calls are resolved
// unambiguously.

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sync"
)

// Det parameters mirrored from TextDetector / DBPostProcess.
const (
	detLimitSideLen  = 960
	detThresh        = 0.3
	detBoxThresh     = 0.5
	detMaxCandidates = 1000
	detUnclipRatio   = 1.5
	detMinSize       = 3

	detMean0, detMean1, detMean2 = 0.485, 0.456, 0.406
	detStd0, detStd1, detStd2    = 0.229, 0.224, 0.225
)

// DetBox is one detected text region as a 4-point quad in original-image
// coordinates, clockwise from top-left.
type DetBox struct {
	Pts   [4][2]float32
	Score float32
}

// DetResult is the full detection output.
type DetResult struct {
	Boxes []DetBox
}

// detSessionPools caches ONNX sessions for the DB text detector, keyed by
// (model path, resized height, resized width). RunDet creates one session per
// unique resized shape and reuses it across pages through a sync.Pool, so a
// multi-page document does not pay NewSession/Destroy per page. Sessions are
// pooled per instance — never shared across concurrent Run calls — because
// Session.Run mutates the session's fixed-shape input/output tensors; the
// native det branch runs concurrently across the page worker pool
// (inferOCRDetect returns before the inference limiter's withSlot), so a
// naively shared single session would race.
var detSessionPools sync.Map // detSessKey -> *sync.Pool

type detSessKey struct {
	modelPath string
	rh, rw    int64
}

// getDetSession returns a reusable detector session for the given resized
// shape plus a release func that returns it to the pool. The caller must call
// release exactly once. On a pool miss a fresh session is created; creation
// errors are propagated and nothing is cached.
func getDetSession(modelPath string, rh, rw int64) (*Session, func(), error) {
	key := detSessKey{modelPath, rh, rw}
	v, _ := detSessionPools.LoadOrStore(key, &sync.Pool{})
	pool := v.(*sync.Pool)
	if got := pool.Get(); got != nil {
		if s, ok := got.(*Session); ok {
			return s, func() { pool.Put(s) }, nil
		}
	}
	s, err := NewSession(modelPath, "x",
		[]int64{1, 3, rh, rw}, "sigmoid_0.tmp_0",
		[]int64{1, 1, rh, rw})
	if err != nil {
		return nil, nil, err
	}
	return s, func() { pool.Put(s) }, nil
}

// RunDet runs preprocessing + ONNX inference + DB post-processing and returns
// the detected text-box quads. Both the gocv build (OpenCV findContours) and
// the pure-Go build (connected components) post-process inline; the only
// difference is the contour-extraction backend selected by build tag.
func RunDet(modelDir string, img *Image) (DetResult, error) {
	blob, rh, rw, sh, sw := detPreprocess(img)
	sess, release, e := getDetSession(filepath.Join(modelDir, "det.onnx"), int64(rh), int64(rw))
	if e != nil {
		return DetResult{}, e
	}
	defer release()

	out, e := sess.Run(blob)
	if e != nil {
		return DetResult{}, e
	}
	// out is [1,1,rh,rw]; flatten to [rh,rw].
	p := make([]float32, rh*rw)
	copy(p, out)

	boxes, _ := dbPostProcess(p, rh, rw, sh, sw)
	return DetResult{Boxes: boxes}, nil
}

func round32(v int) int {
	r := int(math.Round(float64(v) / 32.0))
	return r * 32
}

// normalizeCHW applies the DetResizeForTest Normalization (scale 1/255,
// mean/std, hwc->chw) to a BGR byte buffer of size h*w*3.
func normalizeCHW(bgr []byte, h, w, srcW, srcH int) []float32 {
	_ = srcW
	_ = srcH
	blob := make([]float32, 3*h*w)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			for c := 0; c < 3; c++ {
				v := float32(bgr[(y*w+x)*3+c]) / 255.0
				switch c {
				case 0:
					v = (v - detMean0) / detStd0
				case 1:
					v = (v - detMean1) / detStd1
				case 2:
					v = (v - detMean2) / detStd2
				}
				blob[c*h*w+y*w+x] = v
			}
		}
	}
	return blob
}

// ---- geometry primitives shared by both builds ----

type pt struct{ X, Y float64 }

func (p pt) add(o pt) pt        { return pt{p.X + o.X, p.Y + o.Y} }
func (p pt) sub(o pt) pt        { return pt{p.X - o.X, p.Y - o.Y} }
func (p pt) scale(s float64) pt { return pt{p.X * s, p.Y * s} }
func (p pt) len() float64       { return math.Hypot(p.X, p.Y) }

// unclip expands a quad outward by `ratio`, mirroring DBPostProcess.unclip
// (polygon.area * ratio / polygon.length, offset with a round join). It is a
// faithful integer-space port of Clipper1's ClipperOffset (JT_ROUND /
// ET_CLOSEDPOLYGON) — see clipper_offset.go. Clipper1 works in integer
// coordinates: the float quad is truncated to int64, the offset is computed
// with round-half-away, and the result is returned as integer coordinates,
// exactly matching what pyclipper (the deepdoc oracle) does. Returns the
// expanded polygon as a list of points.
func unclip(box [4]pt, ratio float64) []pt {
	return clipperOffset(box, ratio)
}

// S1 diagnostic: collect every contour's pre-unclip min-area rect (the quad
// returned by minAreaRect before unclip/scale) so it can be compared box-for-box
// against deepdoc's pre_box (testdata/contours.json). Gated by DLA_DUMP_QUADS.
// If these quads already match deepdoc at ~0px, the geometry is exact and the
// residual DET error lives entirely in the earlier mask/contour extraction.
var dlaPreUnclip [][4][2]float64

func dlaRecordPreUnclip(q [4]pt) {
	if os.Getenv("DLA_DUMP_QUADS") == "" {
		return
	}
	var v [4][2]float64
	for i := range q {
		v[i] = [2]float64{q[i].X, q[i].Y}
	}
	dlaPreUnclip = append(dlaPreUnclip, v)
}

func dlaFlushPreUnclip() {
	if os.Getenv("DLA_DUMP_QUADS") == "" {
		return
	}
	b, _ := json.Marshal(dlaPreUnclip)
	_ = os.WriteFile("/tmp/go_quads_pre.json", b, 0o644)
	dlaPreUnclip = nil
}

// S2 diagnostic: collect each contour's post-unclip min-area rect (quad2, in
// resized coordinates, before scaling to source). Comparing this against the
// deepdoc oracle's post-unclip quad isolates whether the residual DET error
// lives in the unclip->re-rect stage or in the scale/filter stage.
var dlaPostUnclip [][4][2]float64

func dlaRecordPostUnclip(q [4]pt) {
	if os.Getenv("DLA_DUMP_QUADS") == "" {
		return
	}
	var v [4][2]float64
	for i := range q {
		v[i] = [2]float64{q[i].X, q[i].Y}
	}
	dlaPostUnclip = append(dlaPostUnclip, v)
}

func dlaFlushPostUnclip() {
	if os.Getenv("DLA_DUMP_QUADS") == "" {
		return
	}
	b, _ := json.Marshal(dlaPostUnclip)
	_ = os.WriteFile("/tmp/go_quads_post.json", b, 0o644)
	dlaPostUnclip = nil
}

func polygonArea(p []pt) float64 {
	n := len(p)
	var a float64
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		a += p[i].X*p[j].Y - p[j].X*p[i].Y
	}
	return a / 2
}

func polygonPerimeter(p []pt) float64 {
	n := len(p)
	var L float64
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		L += p[i].sub(p[j]).len()
	}
	return L
}

// convexHull returns the CCW convex hull (Andrew's monotone chain). Shared by
// both builds; only the pure-Go dbPostProcess uses it today, but it is a
// generic geometry helper so it lives here (build-tag free).
func convexHull(pts []pt) []pt {
	n := len(pts)
	if n < 3 {
		out := make([]pt, n)
		copy(out, pts)
		return out
	}
	// sort by x then y
	sorted := make([]pt, n)
	copy(sorted, pts)
	sortPts(sorted)
	cross := func(o, a, b pt) float64 {
		return (a.X-o.X)*(b.Y-o.Y) - (a.Y-o.Y)*(b.X-o.X)
	}
	lower := make([]pt, 0, n)
	for _, p := range sorted {
		for len(lower) >= 2 && cross(lower[len(lower)-2], lower[len(lower)-1], p) <= 0 {
			lower = lower[:len(lower)-1]
		}
		lower = append(lower, p)
	}
	upper := make([]pt, 0, n)
	for i := n - 1; i >= 0; i-- {
		p := sorted[i]
		for len(upper) >= 2 && cross(upper[len(upper)-2], upper[len(upper)-1], p) <= 0 {
			upper = upper[:len(upper)-1]
		}
		upper = append(upper, p)
	}
	hull := append(lower[:len(lower)-1], upper[:len(upper)-1]...)
	return hull
}

// getMiniBoxes replicates DBPostProcess.get_mini_boxes exactly: sort the 4
// corners by x, then emit a canonical clockwise quad from top-left. It is used
// by both builds (the pure-Go minAreaRect and gocv.MinAreaRect both feed it).
func getMiniBoxes(box [4]pt) [4]pt {
	s := []pt{box[0], box[1], box[2], box[3]}
	sortPtsByX(s)
	var idx1, idx2, idx3, idx4 int
	if s[1].Y > s[0].Y {
		idx1, idx4 = 0, 1
	} else {
		idx1, idx4 = 1, 0
	}
	if s[3].Y > s[2].Y {
		idx2, idx3 = 2, 3
	} else {
		idx2, idx3 = 3, 2
	}
	return [4]pt{s[idx1], s[idx2], s[idx3], s[idx4]}
}

// minAreaRect computes the minimum-area enclosing rectangle of a convex
// polygon via rotating calipers, mirroring cv2.minAreaRect + cv2.boxPoints,
// then reorders the 4 corners the way DBPostProcess.get_mini_boxes does
// (sorted by x, canonical clockwise from top-left). Returns the 4 corners and
// the smaller side length (min(w,h)). It is float-precision (no integer
// rounding) so both builds match Python's cv2.minAreaRect exactly — gocv's
// own MinAreaRect rounds center/size to int and would lose sub-pixel accuracy.
func minAreaRect(poly []pt) ([4]pt, float64) {
	var corners [4]pt
	n := len(poly)
	if n == 0 {
		return corners, 0
	}
	if n == 1 {
		corners = [4]pt{poly[0], poly[0], poly[0], poly[0]}
		return corners, 0
	}
	if n == 2 {
		corners = [4]pt{poly[0], poly[1], poly[1], poly[0]}
		return corners, 0
	}
	bestArea := math.MaxFloat64
	var bcx, bcy, bw, bh, bux, buy, bvx, bvy float64
	for i := 0; i < n; i++ {
		p1 := poly[i]
		p2 := poly[(i+1)%n]
		dx := p2.X - p1.X
		dy := p2.Y - p1.Y
		L := math.Hypot(dx, dy)
		if L == 0 {
			continue
		}
		ux, uy := dx/L, dy/L
		vx, vy := -uy, ux
		minU, maxU := math.MaxFloat64, -math.MaxFloat64
		minV, maxV := math.MaxFloat64, -math.MaxFloat64
		for _, p := range poly {
			u := (p.X-p1.X)*ux + (p.Y-p1.Y)*uy
			v := (p.X-p1.X)*vx + (p.Y-p1.Y)*vy
			if u < minU {
				minU = u
			}
			if u > maxU {
				maxU = u
			}
			if v < minV {
				minV = v
			}
			if v > maxV {
				maxV = v
			}
		}
		wdt := maxU - minU
		hgt := maxV - minV
		area := wdt * hgt
		if area < bestArea {
			bestArea = area
			bcx = p1.X + ux*(minU+maxU)/2 + vx*(minV+maxV)/2
			bcy = p1.Y + uy*(minU+maxU)/2 + vy*(minV+maxV)/2
			bw, bh = wdt, hgt
			bux, buy, bvx, bvy = ux, uy, vx, vy
		}
	}
	hwx, hwy := bux*bw/2, buy*bw/2
	hhx, hhy := bvx*bh/2, bvy*bh/2
	box := [4]pt{
		{bcx - hwx - hhx, bcy - hwy - hhy},
		{bcx + hwx - hhx, bcy + hwy - hhy},
		{bcx + hwx + hhx, bcy + hwy + hhy},
		{bcx - hwx + hhx, bcy - hwy + hhy},
	}
	return getMiniBoxes(box), math.Min(bw, bh)
}

// boxScoreFast mirrors DBPostProcess.box_score_fast: rasterize the quad into a
// mask and return the mean of pred over that region (cv2.mean with mask). The
// scanline fillPoly uses the quad's sub-pixel coordinates, matching Python's
// float fillPoly more closely than OpenCV's integer-point fillPoly, so this is
// shared by both builds for consistent thresholding.
func boxScoreFast(pred []float32, w, h int, box [4]pt) float32 {
	xmin := clampi(int(math.Floor(minX(box))), 0, w-1)
	xmax := clampi(int(math.Ceil(maxX(box))), 0, w-1)
	ymin := clampi(int(math.Floor(minY(box))), 0, h-1)
	ymax := clampi(int(math.Ceil(maxY(box))), 0, h-1)
	mw, mh := xmax-xmin+1, ymax-ymin+1
	if mw <= 0 || mh <= 0 {
		return 0
	}
	mask := make([]bool, mw*mh)
	// cv2.fillPoly receives integer-rounded (truncated, int32) points, so
	// match it: truncate each quad vertex toward zero before rasterizing.
	shifted := [4]pt{
		{math.Trunc(box[0].X) - float64(xmin), math.Trunc(box[0].Y) - float64(ymin)},
		{math.Trunc(box[1].X) - float64(xmin), math.Trunc(box[1].Y) - float64(ymin)},
		{math.Trunc(box[2].X) - float64(xmin), math.Trunc(box[2].Y) - float64(ymin)},
		{math.Trunc(box[3].X) - float64(xmin), math.Trunc(box[3].Y) - float64(ymin)},
	}
	fillPoly(mask, mw, mh, shifted)
	var sum, cnt float64
	for y := 0; y < mh; y++ {
		for x := 0; x < mw; x++ {
			if !mask[y*mw+x] {
				continue
			}
			sum += float64(pred[(ymin+y)*w+(xmin+x)])
			cnt++
		}
	}
	if cnt == 0 {
		return 0
	}
	return float32(sum / cnt)
}

// filterTagDetRes mirrors TextDetector.filter_tag_det_res.
func filterTagDetRes(boxes []DetBox, srcH, srcW int) []DetBox {
	out := make([]DetBox, 0, len(boxes))
	for _, b := range boxes {
		ordered := orderPointsClockwise(b.Pts)
		clipped := clipDetRes(ordered, srcH, srcW)
		dx1 := float64(clipped[0][0] - clipped[1][0])
		dy1 := float64(clipped[0][1] - clipped[1][1])
		dx3 := float64(clipped[0][0] - clipped[3][0])
		dy3 := float64(clipped[0][1] - clipped[3][1])
		wdt := int(math.Round(math.Hypot(dx1, dy1)))
		hgt := int(math.Round(math.Hypot(dx3, dy3)))
		if wdt <= 3 || hgt <= 3 {
			continue
		}
		out = append(out, DetBox{Pts: clipped, Score: b.Score})
	}
	return out
}

func orderPointsClockwise(p [4][2]float32) [4][2]float32 {
	pts := [4]pt{{float64(p[0][0]), float64(p[0][1])}, {float64(p[1][0]), float64(p[1][1])},
		{float64(p[2][0]), float64(p[2][1])}, {float64(p[3][0]), float64(p[3][1])}}
	s := getMiniBoxes(pts)
	var out [4][2]float32
	for i := 0; i < 4; i++ {
		out[i] = [2]float32{float32(s[i].X), float32(s[i].Y)}
	}
	return out
}

func clipDetRes(p [4][2]float32, srcH, srcW int) [4][2]float32 {
	var out [4][2]float32
	for i := 0; i < 4; i++ {
		out[i][0] = float32(clampi(int(p[i][0]), 0, srcW-1))
		out[i][1] = float32(clampi(int(p[i][1]), 0, srcH-1))
	}
	return out
}

// Wire emits the detect wire format matched by deepdoc/server/adapters/
// ocr_adapter.py detect mode: {"output": [[ [ [x,y]*4, ... ] ]]}.
// Boxes live at output[0][0] (page -> batch -> boxes).
func (r DetResult) Wire() string {
	quads := make([][][2]float32, 0, len(r.Boxes))
	for _, b := range r.Boxes {
		quads = append(quads, b.Pts[:])
	}
	batch := [][][][2]float32{quads} // [quads]; 1 element (the page batch)
	out, _ := json.Marshal(map[string]any{"output": [][][][][2]float32{batch}})
	return string(out)
}

// ---- small helpers ----

func clampf(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func clampi(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func minX(b [4]pt) float64 {
	m := b[0].X
	for i := 1; i < 4; i++ {
		if b[i].X < m {
			m = b[i].X
		}
	}
	return m
}
func maxX(b [4]pt) float64 {
	m := b[0].X
	for i := 1; i < 4; i++ {
		if b[i].X > m {
			m = b[i].X
		}
	}
	return m
}
func minY(b [4]pt) float64 {
	m := b[0].Y
	for i := 1; i < 4; i++ {
		if b[i].Y < m {
			m = b[i].Y
		}
	}
	return m
}
func maxY(b [4]pt) float64 {
	m := b[0].Y
	for i := 1; i < 4; i++ {
		if b[i].Y > m {
			m = b[i].Y
		}
	}
	return m
}

func norm(v pt) float64 { return v.len() }
