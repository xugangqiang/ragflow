//go:build cgo && native_det

package pdf_test

// TestInProcessVsServiceIoUDiff is the alignment-analysis tool for OCR
// detection post-processing. Unlike TestInProcessVsServiceCompare (which
// aligns raw scores on page 0 with a brittle position-sort 1:1 pairing), this
// tool:
//
//   - renders EVERY page of every PDF with Go (pdfium),
//   - runs OCRDetect on that exact image through BOTH the Go in-process
//     backend and the Python inference service,
//   - greedy-matches the two box sets by polygon IoU (rotated-quad aware),
//     so it no longer depends on the two backends emitting boxes in the same
//     order or the same count,
//   - classifies every unmatched box as split/merge, threshold-fragment, or a
//     genuine one-sided detection,
//   - renders a per-page overlay PNG (Go boxes red, Py boxes blue, unmatched
//     boxes highlighted thick) so the divergence can be located visually,
//   - and writes a per-page / per-PDF / overall diff report classifying the
//     gap by kind.
//
// It is an analysis-only tool: the Python service is contacted read-only over
// HTTP; no Python source is modified. It is intended to be run manually
// (build tag native_det) against a live Python service.
//
// Prerequisites (set via env):
//   ORT_LIB      path to libonnxruntime.so
//   MODEL_DIR    DeepDoc model directory (rag/res/deepdoc)
//   DEEPDOC_URL  Python inference service, default http://localhost:9390
//   INPROC_PDFS  optional comma list of PDF basenames to limit the run
//   INPROC_PAGES optional page cap (int); 0 = all pages of every PDF
//
// Usage:
//   ORT_LIB=... MODEL_DIR=... DEEPDOC_URL=http://localhost:9390 \
//     bash build.sh --test-native -run TestInProcessVsServiceIoUDiff \
//     ./internal/deepdoc/parser/pdf/...
//
// Output (git-ignored, under testdata/output/render_compare/iou/):
//   <pdf>_p<page>.png            overlay for pages that have any divergence
//   <pdf>_p<page>_diff.json      per-page itemized diff
//   iou_diff_report.json         per-PDF and overall classification tally
import (
	"context"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"

	"dla-native/native"
	pdfpkg "ragflow/internal/deepdoc/parser/pdf"
	inf "ragflow/internal/deepdoc/parser/pdf/inference"
	infnative "ragflow/internal/deepdoc/parser/pdf/inference/native"
)

// ---------------------------------------------------------------------------
// rotated-quad geometry (IoU-aware matching)
// ---------------------------------------------------------------------------

type pt2 struct{ X, Y float64 }

func boxToQuad(b jsonBox) [4]pt2 {
	return [4]pt2{
		{b.X0, b.Y0}, {b.X1, b.Y1}, {b.X2, b.Y2}, {b.X3, b.Y3},
	}
}

func quadAABB(q [4]pt2) (minx, miny, maxx, maxy float64) {
	minx, miny, maxx, maxy = q[0].X, q[0].Y, q[0].X, q[0].Y
	for _, p := range q[1:] {
		if p.X < minx {
			minx = p.X
		}
		if p.Y < miny {
			miny = p.Y
		}
		if p.X > maxx {
			maxx = p.X
		}
		if p.Y > maxy {
			maxy = p.Y
		}
	}
	return
}

// signedArea returns the signed area (CCW positive).
func signedArea(p []pt2) float64 {
	var a float64
	n := len(p)
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		a += p[i].X*p[j].Y - p[j].X*p[i].Y
	}
	return a / 2
}

func polyArea(p []pt2) float64 { return math.Abs(signedArea(p)) }

// orientCCW flips winding so the polygon is counter-clockwise.
func orientCCW(p []pt2) []pt2 {
	if signedArea(p) < 0 {
		out := make([]pt2, len(p))
		for i, j := 0, len(p)-1; i < len(p); i, j = i+1, j-1 {
			out[i] = p[j]
		}
		return out
	}
	return p
}

// inside reports whether point p is left of / on edge a->b (CCW polygon).
func inside(p, a, b pt2) bool {
	return (b.X-a.X)*(p.Y-a.Y)-(b.Y-a.Y)*(p.X-a.X) >= 0
}

// intersectConvex clips subject by clip (both convex, CCW) via
// Sutherland-Hodgman, returning the intersection polygon.
func intersectConvex(subject, clip []pt2) []pt2 {
	out := append([]pt2(nil), subject...)
	cn := len(clip)
	for i := 0; i < cn; i++ {
		if len(out) == 0 {
			break
		}
		a := clip[i]
		b := clip[(i+1)%cn]
		in := out
		out = make([]pt2, 0, len(in)+4)
		m := len(in)
		for j := 0; j < m; j++ {
			p := in[j]
			q := in[(j+1)%m]
			pIn := inside(p, a, b)
			qIn := inside(q, a, b)
			if pIn {
				out = append(out, p)
			}
			if pIn != qIn {
				// intersection of segment p-q with edge a-b
				dx1, dy1 := b.X-a.X, b.Y-a.Y
				dx2, dy2 := q.X-p.X, q.Y-p.Y
				den := dx1*dy2 - dy1*dx2
				if den != 0 {
					// standard 2-line intersection: t = ((a-p) × d1) / -(d1 × d2)
					t := ((a.X-p.X)*dy1 - (a.Y-p.Y)*dx1) / (-den)
					out = append(out, pt2{p.X + t*dx2, p.Y + t*dy2})
				}
			}
		}
	}
	return out
}

// quadIoU computes the intersection-over-union of two rotated quads.
func quadIoU(a, b [4]pt2) float64 {
	ca, cb := orientCCW(a[:]), orientCCW(b[:])
	inter := intersectConvex(ca, cb)
	if len(inter) < 3 {
		return 0
	}
	ia := polyArea(inter)
	ua := polyArea(ca) + polyArea(cb) - ia
	if ua <= 0 {
		return 0
	}
	return ia / ua
}

// aabbOverlapRatio returns the intersection area of the two AABBs divided by
// the smaller AABB area. It is a cheap proxy used when the precise polygon IoU
// is too low to be a real match but the boxes clearly occupy the same region.
func aabbOverlapRatio(a, b [4]pt2) float64 {
	ax0, ay0, ax1, ay1 := quadAABB(a)
	bx0, by0, bx1, by1 := quadAABB(b)
	ix0, iy0 := math.Max(ax0, bx0), math.Max(ay0, by0)
	ix1, iy1 := math.Min(ax1, bx1), math.Min(ay1, by1)
	if ix1 <= ix0 || iy1 <= iy0 {
		return 0
	}
	inter := (ix1 - ix0) * (iy1 - iy0)
	aa := (ax1 - ax0) * (ay1 - ay0)
	ab := (bx1 - bx0) * (by1 - by0)
	if aa <= 0 || ab <= 0 {
		return 0
	}
	return inter / math.Min(aa, ab)
}

func quadCenter(q [4]pt2) (float64, float64) {
	var x, y float64
	for _, p := range q {
		x += p.X
		y += p.Y
	}
	return x / 4, y / 4
}

// maxCornerDist returns the largest per-corner distance between two quads
// after pairing corners by index (both come from minAreaRect, same ordering).
func maxCornerDist(a, b [4]pt2) float64 {
	var m float64
	for i := 0; i < 4; i++ {
		d := math.Hypot(a[i].X-b[i].X, a[i].Y-b[i].Y)
		if d > m {
			m = d
		}
	}
	return m
}

// ---------------------------------------------------------------------------
// greedy IoU matching
// ---------------------------------------------------------------------------

const (
	matchIoUThr    = 0.5 // min polygon IoU for two boxes to be the same detection
	driftIoUThr    = 0.9 // below this, a matched pair is not a tight overlap
	driftCornerThr = 3.0 // max per-corner distance (px) above which a matched pair is "drifted"
)

type iouMatch struct {
	GoIdx, PyIdx int
	IoU          float64
}

// greedyIoUMatch pairs Go boxes with Py boxes by descending polygon IoU.
// It returns the matched pairs plus the indices (into the original slices)
// that were left unmatched on each side.
func greedyIoUMatch(goB, pyB []jsonBox) (matches []iouMatch, goOnly, pyOnly []int) {
	type cand struct {
		gi, pi int
		iou    float64
	}
	var cs []cand
	for i, g := range goB {
		gq := boxToQuad(g)
		for j, p := range pyB {
			if iou := quadIoU(gq, boxToQuad(p)); iou >= matchIoUThr {
				cs = append(cs, cand{i, j, iou})
			}
		}
	}
	sort.Slice(cs, func(i, j int) bool { return cs[i].iou > cs[j].iou })
	usedG := make(map[int]bool, len(goB))
	usedP := make(map[int]bool, len(pyB))
	for _, c := range cs {
		if usedG[c.gi] || usedP[c.pi] {
			continue
		}
		usedG[c.gi], usedP[c.pi] = true, true
		matches = append(matches, iouMatch{c.gi, c.pi, c.iou})
	}
	for i := range goB {
		if !usedG[i] {
			goOnly = append(goOnly, i)
		}
	}
	for i := range pyB {
		if !usedP[i] {
			pyOnly = append(pyOnly, i)
		}
	}
	return
}

// ---------------------------------------------------------------------------
// classification
// ---------------------------------------------------------------------------

type diffItem struct {
	Kind          string   `json:"kind"` // matched | go_only | py_only
	SubClass      string   `json:"subclass,omitempty"`
	GoIndex       int      `json:"go_index,omitempty"`
	PyIndex       int      `json:"py_index,omitempty"`
	IoU           float64  `json:"iou,omitempty"`
	MaxCornerDist float64  `json:"max_corner_dist,omitempty"`
	CenterShift   float64  `json:"center_shift,omitempty"`
	GoArea        float64  `json:"go_area,omitempty"`
	PyArea        float64  `json:"py_area,omitempty"`
	GoBox         *jsonBox `json:"go_box,omitempty"`
	PyBox         *jsonBox `json:"py_box,omitempty"`
}

type pageDiff struct {
	PDF             string         `json:"pdf"`
	Page            int            `json:"page"`
	GoCount         int            `json:"go_count"`
	PyCount         int            `json:"py_count"`
	Matched         int            `json:"matched"`
	GoOnly          int            `json:"go_only"`
	PyOnly          int            `json:"py_only"`
	MatchedMeanIoU  float64        `json:"matched_mean_iou"`
	MatchedMinIoU   float64        `json:"matched_min_iou"`
	MatchedMaxCorner float64       `json:"matched_max_corner"`
	MatchedDrift    int            `json:"matched_drift"` // matched pairs failing the tight-overlap test
	SubClasses      map[string]int `json:"subclasses"`
	Items           []diffItem     `json:"items"`
}

// classifyPage builds the itemized per-page diff and tallies subclasses.
func classifyPage(goB, pyB []jsonBox, matches []iouMatch, goOnly, pyOnly []int, page int, pdf string) pageDiff {
	pd := pageDiff{
		PDF:        pdf,
		Page:       page,
		GoCount:    len(goB),
		PyCount:    len(pyB),
		Matched:    len(matches),
		GoOnly:     len(goOnly),
		PyOnly:     len(pyOnly),
		MatchedMinIoU: 2.0,
		SubClasses: map[string]int{},
	}

	var iouSum, maxCorner float64
	for _, m := range matches {
		gq, pq := boxToQuad(goB[m.GoIdx]), boxToQuad(pyB[m.PyIdx])
		gcx, gcy := quadCenter(gq)
		pcx, pcy := quadCenter(pq)
		mcd := maxCornerDist(gq, pq)
		it := diffItem{
			Kind:          "matched",
			GoIndex:       m.GoIdx,
			PyIndex:       m.PyIdx,
			IoU:           m.IoU,
			MaxCornerDist: mcd,
			CenterShift:   math.Hypot(gcx-pcx, gcy-pcy),
			GoArea:        polyArea(orientCCW(gq[:])),
			PyArea:        polyArea(orientCCW(pq[:])),
			GoBox:         &goB[m.GoIdx],
			PyBox:         &pyB[m.PyIdx],
		}
		if m.IoU < driftIoUThr || mcd > driftCornerThr {
			it.SubClass = "coord_drift" // matched but geometrically off: contour-tracing / scaling residue
			pd.MatchedDrift++
			pd.SubClasses["coord_drift"]++
		}
		if m.IoU < pd.MatchedMinIoU {
			pd.MatchedMinIoU = m.IoU
		}
		if mcd > maxCorner {
			maxCorner = mcd
		}
		iouSum += m.IoU
		pd.Items = append(pd.Items, it)
	}
	if pd.Matched > 0 {
		pd.MatchedMeanIoU = iouSum / float64(pd.Matched)
	}
	pd.MatchedMaxCorner = maxCorner

	// helper: how many boxes from `others` overlap `q` (IoU or AABB ratio)?
	countOverlaps := func(q [4]pt2, others []jsonBox) int {
		n := 0
		for _, o := range others {
			oq := boxToQuad(o)
			if quadIoU(q, oq) > 0.15 || aabbOverlapRatio(q, oq) > 0.2 {
				n++
			}
		}
		return n
	}

	classifyOrphan := func(idx int, q [4]pt2, others []jsonBox, side string, area float64) diffItem {
		overlaps := countOverlaps(q, others)
		it := diffItem{
			Kind:    side,
			GoIndex: idx,
			GoArea:  area,
			GoBox:   &goB[idx],
		}
		switch {
		case overlaps >= 2:
			it.SubClass = "split_merge" // one box on this side vs >=2 on the other
		case overlaps == 1:
			it.SubClass = "partial_overlap" // same region, geometry/coords drifted below match IoU
		case area < 100:
			it.SubClass = "threshold_fragment" // small orphan: low-confidence fragment dropped by the other side
		default:
			it.SubClass = "one_sided" // genuine detection difference
		}
		pd.SubClasses[it.SubClass]++
		return it
	}

	for _, i := range goOnly {
		q := boxToQuad(goB[i])
		pd.Items = append(pd.Items, classifyOrphan(i, q, pyB, "go_only", polyArea(orientCCW(q[:]))))
	}
	for _, i := range pyOnly {
		// symmetric: treat the py box as the orphan and count overlaps against go boxes
		q := boxToQuad(pyB[i])
		it := diffItem{
			Kind:    "py_only",
			PyIndex: i,
			PyArea:  polyArea(orientCCW(q[:])),
			PyBox:   &pyB[i],
		}
		overlaps := 0
		for _, g := range goB {
			gq := boxToQuad(g)
			if quadIoU(gq, q) > 0.15 || aabbOverlapRatio(gq, q) > 0.2 {
				overlaps++
			}
		}
		switch {
		case overlaps >= 2:
			it.SubClass = "split_merge"
		case overlaps == 1:
			it.SubClass = "partial_overlap"
		case it.PyArea < 100:
			it.SubClass = "threshold_fragment"
		default:
			it.SubClass = "one_sided"
		}
		pd.SubClasses[it.SubClass]++
		pd.Items = append(pd.Items, it)
	}
	return pd
}

// ---------------------------------------------------------------------------
// overlay rendering (stdlib only: image/draw + image/png)
// ---------------------------------------------------------------------------

var (
	colGo    = color.RGBA{220, 20, 20, 255}
	colPy    = color.RGBA{20, 60, 220, 255}
	colDrift = color.RGBA{220, 20, 200, 255} // magenta: matched but geometrically drifted
)

// drawThickLine draws a line from a to b with the given thickness (px).
func drawThickLine(img *image.RGBA, a, b pt2, c color.RGBA, t int) {
	x0, y0 := int(math.Round(a.X)), int(math.Round(a.Y))
	x1, y1 := int(math.Round(b.X)), int(math.Round(b.Y))
	dx := int(math.Abs(float64(x1 - x0)))
	dy := int(math.Abs(float64(y1 - y0)))
	sx, sy := 1, 1
	if x0 > x1 {
		sx = -1
	}
	if y0 > y1 {
		sy = -1
	}
	err := dx - dy
	half := t / 2
	for {
		for ox := -half; ox <= half; ox++ {
			for oy := -half; oy <= half; oy++ {
				img.Set(x0+ox, y0+oy, c)
			}
		}
		if x0 == x1 && y0 == y1 {
			break
		}
		e2 := 2 * err
		if e2 > -dy {
			err -= dy
			x0 += sx
		}
		if e2 < dx {
			err += dx
			y0 += sy
		}
	}
}

func drawQuad(img *image.RGBA, q [4]pt2, c color.RGBA, t int) {
	for i := 0; i < 4; i++ {
		drawThickLine(img, q[i], q[(i+1)%4], c, t)
	}
}

// renderOverlay draws Go (red) and Py (blue) boxes on a copy of the page
// image. Unmatched boxes (goOnly/pyOnly) are highlighted thick in their side
// colour; matched boxes that failed the tight-overlap test (driftGo/driftPy)
// are highlighted thick magenta so the geometric residue is visible.
func renderOverlay(img image.Image, goB, pyB []jsonBox, matches []iouMatch, goOnly, pyOnly, driftGo, driftPy []int) *image.RGBA {
	b := img.Bounds()
	out := image.NewRGBA(b)
	draw.Draw(out, b, img, b.Min, draw.Src)
	matchedG := make(map[int]bool, len(matches))
	matchedP := make(map[int]bool, len(matches))
	driftG := make(map[int]bool, len(driftGo))
	driftP := make(map[int]bool, len(driftPy))
	for _, m := range matches {
		matchedG[m.GoIdx], matchedP[m.PyIdx] = true, true
	}
	for _, i := range driftGo {
		driftG[i] = true
	}
	for _, i := range driftPy {
		driftP[i] = true
	}
	for i, bx := range pyB {
		switch {
		case driftP[i]:
			drawQuad(out, boxToQuad(bx), colDrift, 3)
		case matchedP[i]:
			drawQuad(out, boxToQuad(bx), colPy, 1)
		default:
			drawQuad(out, boxToQuad(bx), colPy, 3)
		}
	}
	for i, bx := range goB {
		switch {
		case driftG[i]:
			drawQuad(out, boxToQuad(bx), colDrift, 3)
		case matchedG[i]:
			drawQuad(out, boxToQuad(bx), colGo, 1)
		default:
			drawQuad(out, boxToQuad(bx), colGo, 3)
		}
	}
	return out
}

func writePNG(t *testing.T, path string, img *image.RGBA) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Logf("png create %s: %v", path, err)
		return
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		t.Logf("png encode %s: %v", path, err)
	}
}

// ---------------------------------------------------------------------------
// test driver
// ---------------------------------------------------------------------------

type pdfIouSummary struct {
	PDF             string         `json:"pdf"`
	Pages           int            `json:"pages"`
	GoBoxes         int            `json:"go_boxes"`
	PyBoxes         int            `json:"py_boxes"`
	Matched         int            `json:"matched"`
	GoOnly          int            `json:"go_only"`
	PyOnly          int            `json:"py_only"`
	MatchedMeanIoU  float64        `json:"matched_mean_iou"`
	MatchedMinIoU   float64        `json:"matched_min_iou"`
	MatchedMaxCorner float64       `json:"matched_max_corner"`
	MatchedDrift    int            `json:"matched_drift"`
	SubClass        map[string]int `json:"subclasses"`
}

type iouReport struct {
	PDFs             []pdfIouSummary `json:"pdfs"`
	Overall          map[string]int  `json:"overall_subclasses"`
	MatchedTotal     int             `json:"matched_total"`
	MatchedMeanIoU   float64         `json:"matched_mean_iou"`
	MatchedMinIoU    float64         `json:"matched_min_iou"`
	MatchedMaxCorner float64         `json:"matched_max_corner"`
	MatchedDrift     int             `json:"matched_drift"`
	OverlayDir       string          `json:"overlay_dir"`
	PagesWithDiff    int             `json:"pages_with_diff"`
}

func TestInProcessVsServiceIoUDiff(t *testing.T) {
	ortLib := os.Getenv("ORT_LIB")
	modelDir := os.Getenv("MODEL_DIR")
	pyURL := os.Getenv("DEEPDOC_URL")
	if pyURL == "" {
		pyURL = "http://localhost:9390"
	}
	pageCap := 0
	if v := strings.TrimSpace(os.Getenv("INPROC_PAGES")); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			pageCap = n
		}
	}
	if ortLib == "" || modelDir == "" {
		t.Skip("set ORT_LIB and MODEL_DIR to run in-process comparison")
	}
	if ortLib != "" {
		if err := native.InitORT(ortLib); err != nil {
			t.Fatalf("InitORT: %v", err)
		}
	}
	dropScore := infnative.DefaultDropScore
	if v := strings.TrimSpace(os.Getenv("DEEPDOC_DROP_SCORE")); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			dropScore = f
		}
	}
	goAna, err := infnative.NewAnalyzer(modelDir, dropScore)
	if err != nil {
		t.Fatalf("NewAnalyzer: %v", err)
	}
	pyClient, err := inf.NewClient(pyURL)
	if err != nil {
		t.Fatalf("py client: %v", err)
	}
	if !pyClient.Health() {
		t.Fatalf("py client health check failed for %s", pyURL)
	}

	pdfDir := filepath.Join("testdata", "pdfs")
	outDir := filepath.Join("testdata", "output", "render_compare", "iou")
	os.MkdirAll(outDir, 0755)

	var names []string
	if sub := os.Getenv("INPROC_PDFS"); sub != "" {
		for _, s := range strings.Split(sub, ",") {
			s = strings.TrimSpace(s)
			if s != "" {
				names = append(names, s)
			}
		}
	} else {
		entries, err := os.ReadDir(pdfDir)
		if err != nil {
			t.Fatal(err)
		}
		for _, e := range entries {
			if e.IsDir() || !strings.HasSuffix(strings.ToLower(e.Name()), ".pdf") {
				continue
			}
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)

	ctx := context.Background()
	report := iouReport{OverlayDir: outDir, Overall: map[string]int{}, MatchedMinIoU: 2.0}
	var summaries []pdfIouSummary

	for _, name := range names {
		data, err := os.ReadFile(filepath.Join(pdfDir, name))
		if err != nil {
			t.Logf("%s: read: %v", name, err)
			continue
		}
		eng, err := pdfpkg.NewEngine(data)
		if err != nil {
			t.Logf("%s: engine: %v", name, err)
			continue
		}
		pageCount, err := eng.PageCount()
		if err != nil {
			t.Logf("%s: pagecount: %v", name, err)
			eng.Close()
			continue
		}
		if pageCap > 0 && pageCount > pageCap {
			pageCount = pageCap
		}
		base := strings.TrimSuffix(name, ".pdf")
		sum := pdfIouSummary{PDF: name, Pages: pageCount, SubClass: map[string]int{}, MatchedMinIoU: 2.0}

		for p := 0; p < pageCount; p++ {
			img, err := pdfpkg.RenderPageToImage(eng, p)
			if err != nil {
				t.Logf("%s p%d render: %v", name, p, err)
				continue
			}
			goDet, err := goAna.OCRDetect(ctx, img)
			if err != nil {
				t.Logf("%s p%d det(go): %v", name, p, err)
			}
			pyDet, err := pyClient.OCRDetect(ctx, img)
			if err != nil {
				t.Logf("%s p%d det(py): %v", name, p, err)
			}
			gb, pb := boxFromDeep(goDet), boxFromPDF(pyDet)
			matches, goOnly, pyOnly := greedyIoUMatch(gb, pb)
			pd := classifyPage(gb, pb, matches, goOnly, pyOnly, p, name)

			sum.GoBoxes += len(gb)
			sum.PyBoxes += len(pb)
			sum.Matched += len(matches)
			sum.GoOnly += len(goOnly)
			sum.PyOnly += len(pyOnly)
			sum.MatchedDrift += pd.MatchedDrift
			if pd.Matched > 0 {
				sum.MatchedMeanIoU += pd.MatchedMeanIoU * float64(pd.Matched)
			}
			if pd.MatchedMinIoU < sum.MatchedMinIoU {
				sum.MatchedMinIoU = pd.MatchedMinIoU
			}
			if pd.MatchedMaxCorner > sum.MatchedMaxCorner {
				sum.MatchedMaxCorner = pd.MatchedMaxCorner
			}
			for k, v := range pd.SubClasses {
				sum.SubClass[k] += v
				report.Overall[k] += v
			}
			report.MatchedDrift += pd.MatchedDrift
			report.MatchedTotal += len(matches)
			if pd.Matched > 0 {
				report.MatchedMeanIoU += pd.MatchedMeanIoU * float64(pd.Matched)
			}
			if pd.MatchedMinIoU < report.MatchedMinIoU {
				report.MatchedMinIoU = pd.MatchedMinIoU
			}
			if pd.MatchedMaxCorner > report.MatchedMaxCorner {
				report.MatchedMaxCorner = pd.MatchedMaxCorner
			}

			// Persist pages that diverge in ANY way: missing/extra boxes, or
			// matched boxes that are not a tight overlap (coord drift).
			if len(goOnly) > 0 || len(pyOnly) > 0 || pd.MatchedDrift > 0 {
				report.PagesWithDiff++
				var driftGo, driftPy []int
				for _, it := range pd.Items {
					if it.Kind == "matched" && it.SubClass == "coord_drift" {
						driftGo = append(driftGo, it.GoIndex)
						driftPy = append(driftPy, it.PyIndex)
					}
				}
				writeJSONFile(t, outDir, base+"_p"+strconv.Itoa(p)+"_diff.json", pd)
				overlay := renderOverlay(img, gb, pb, matches, goOnly, pyOnly, driftGo, driftPy)
				writePNG(t, filepath.Join(outDir, base+"_p"+strconv.Itoa(p)+".png"), overlay)
			}
		}
		if sum.Matched > 0 {
			sum.MatchedMeanIoU /= float64(sum.Matched)
		}
		eng.Close()
		summaries = append(summaries, sum)
		t.Logf("%s: pages=%d go=%d py=%d matched=%d goOnly=%d pyOnly=%d drift=%d subclasses=%v",
			name, sum.Pages, sum.GoBoxes, sum.PyBoxes, sum.Matched, sum.GoOnly, sum.PyOnly, sum.MatchedDrift, sum.SubClass)
	}

	report.PDFs = summaries
	if report.MatchedTotal > 0 {
		report.MatchedMeanIoU /= float64(report.MatchedTotal)
	}
	writeJSONFile(t, outDir, "iou_diff_report.json", report)
	t.Logf("Wrote %s/iou_diff_report.json (matched_total=%d mean_iou=%.4f min_iou=%.4f max_corner=%.2f drift=%d pages_with_diff=%d overall_subclasses=%v)",
		outDir, report.MatchedTotal, report.MatchedMeanIoU, report.MatchedMinIoU, report.MatchedMaxCorner, report.MatchedDrift, report.PagesWithDiff, report.Overall)
}
