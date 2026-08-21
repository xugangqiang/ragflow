//go:build cgo && native_det

package pdf_test

// TestInProcessVsServiceCompare confirms that the Go in-process DeepDoc
// backend (NativeAnalyzer) and the Python inference service (DEEPDOC_URL)
// produce consistent raw inference output for the same input image.
//
// It does NOT compare assembly logic. For each PDF it renders page 0 with
// Go (pdfium), then runs DLA / OCR detect / OCR recognize / TSR on that exact
// image through BOTH backends and writes the raw results side by side:
//
//	testdata/output/render_compare/<pdf>_go_<stage>.json
//	testdata/output/render_compare/<pdf>_py_<stage>.json
//
// A per-PDF summary is written to compare_report.json.
//
// The Python service is treated as the golden reference and is only ever
// contacted over HTTP (read-only) — no Python source is modified.
//
// Prerequisites (set via env):
//   ORT_LIB      path to libonnxruntime.so
//   MODEL_DIR    DeepDoc model directory (rag/res/deepdoc)
//   DEEPDOC_URL  Python inference service, default http://localhost:9390
//   INPROC_PDFS  optional comma list of PDF basenames to limit the run
//
// Usage:
//   ORT_LIB=... MODEL_DIR=... DEEPDOC_URL=http://localhost:9390 \
//     bash build.sh --test-native -run TestInProcessVsServiceCompare \
//     ./internal/deepdoc/parser/pdf/...
import (
	"context"
	"encoding/json"
	"image"
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
	pdftype "ragflow/internal/deepdoc/parser/pdf/type"
	deepdoctype "ragflow/internal/deepdoc/parser/type"
)

// cropRect crops a rectangular region from an image (local copy of the
// internal pdf.cropImageRect helper, which is not visible from this external
// test package).
func cropRect(img image.Image, x0, y0, x1, y1 int) image.Image {
	b := img.Bounds()
	if x0 < b.Min.X {
		x0 = b.Min.X
	}
	if y0 < b.Min.Y {
		y0 = b.Min.Y
	}
	if x1 > b.Max.X {
		x1 = b.Max.X
	}
	if y1 > b.Max.Y {
		y1 = b.Max.Y
	}
	out := image.NewRGBA(image.Rect(0, 0, x1-x0, y1-y0))
	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			out.Set(x-x0, y-y0, img.At(x, y))
		}
	}
	return out
}

// JSON shapes mirror the cc-workspace parity format so the two are comparable.
type jsonDLA struct {
	X0, Y0, X1, Y1, Confidence float64
	Label                      string
}
type jsonBox struct {
	X0, Y0, X1, Y1, X2, Y2, X3, Y3 float64
}
type jsonText struct {
	Text       string
	Confidence float64
}
type jsonCell struct {
	X0, Y0, X1, Y1 float64
	Label          string
}

func dlaFromDeep(rs []deepdoctype.DLARegion) []jsonDLA {
	out := make([]jsonDLA, 0, len(rs))
	for _, r := range rs {
		out = append(out, jsonDLA{X0: r.X0, Y0: r.Y0, X1: r.X1, Y1: r.Y1, Confidence: r.Confidence, Label: r.Label})
	}
	return out
}
func dlaFromPDF(rs []pdftype.DLARegion) []jsonDLA {
	out := make([]jsonDLA, 0, len(rs))
	for _, r := range rs {
		out = append(out, jsonDLA{X0: r.X0, Y0: r.Y0, X1: r.X1, Y1: r.Y1, Confidence: r.Confidence, Label: r.Label})
	}
	return out
}
func boxFromDeep(bs []deepdoctype.OCRBox) []jsonBox {
	out := make([]jsonBox, 0, len(bs))
	for _, b := range bs {
		out = append(out, jsonBox{X0: b.X0, Y0: b.Y0, X1: b.X1, Y1: b.Y1, X2: b.X2, Y2: b.Y2, X3: b.X3, Y3: b.Y3})
	}
	return out
}
func boxFromPDF(bs []pdftype.OCRBox) []jsonBox {
	out := make([]jsonBox, 0, len(bs))
	for _, b := range bs {
		out = append(out, jsonBox{X0: b.X0, Y0: b.Y0, X1: b.X1, Y1: b.Y1, X2: b.X2, Y2: b.Y2, X3: b.X3, Y3: b.Y3})
	}
	return out
}
func textFromDeep(ts []deepdoctype.OCRText) []jsonText {
	out := make([]jsonText, 0, len(ts))
	for _, t := range ts {
		out = append(out, jsonText{Text: t.Text, Confidence: t.Confidence})
	}
	return out
}
func textFromPDF(ts []pdftype.OCRText) []jsonText {
	out := make([]jsonText, 0, len(ts))
	for _, t := range ts {
		out = append(out, jsonText{Text: t.Text, Confidence: t.Confidence})
	}
	return out
}
func cellFromDeep(cs []deepdoctype.TSRCell) []jsonCell {
	out := make([]jsonCell, 0, len(cs))
	for _, c := range cs {
		out = append(out, jsonCell{X0: c.X0, Y0: c.Y0, X1: c.X1, Y1: c.Y1, Label: c.Label})
	}
	return out
}
func cellFromPDF(cs []pdftype.TSRCell) []jsonCell {
	out := make([]jsonCell, 0, len(cs))
	for _, c := range cs {
		out = append(out, jsonCell{X0: c.X0, Y0: c.Y0, X1: c.X1, Y1: c.Y1, Label: c.Label})
	}
	return out
}

func byPosDLA(s []jsonDLA) {
	sort.Slice(s, func(i, j int) bool {
		return s[i].Y0 < s[j].Y0 || (s[i].Y0 == s[j].Y0 && s[i].X0 < s[j].X0)
	})
}
func byPosBox(s []jsonBox) {
	sort.Slice(s, func(i, j int) bool {
		return s[i].Y0 < s[j].Y0 || (s[i].Y0 == s[j].Y0 && s[i].X0 < s[j].X0)
	})
}
func byPosCell(s []jsonCell) {
	sort.Slice(s, func(i, j int) bool {
		return s[i].Y0 < s[j].Y0 || (s[i].Y0 == s[j].Y0 && s[i].X0 < s[j].X0)
	})
}

type stageResult struct {
	Stage      string  `json:"stage"`
	GoCount    int     `json:"go_count"`
	PyCount    int     `json:"py_count"`
	MatchCount int     `json:"match_count"`
	ConfMAE    float64 `json:"conf_mae,omitempty"`
	ConfMax    float64 `json:"conf_max,omitempty"`
	Note       string  `json:"note,omitempty"`
}
type pdfResult struct {
	PDF    string        `json:"pdf"`
	Stages []stageResult `json:"stages"`
}

const boxTol = 1.0

func compareDLA(goR, pyR []jsonDLA) stageResult {
	byPosDLA(goR)
	byPosDLA(pyR)
	n := len(goR)
	if len(pyR) < n {
		n = len(pyR)
	}
	match := 0
	for i := 0; i < n; i++ {
		g, p := goR[i], pyR[i]
		if g.Label == p.Label &&
			math.Abs(g.X0-p.X0) < boxTol && math.Abs(g.Y0-p.Y0) < boxTol &&
			math.Abs(g.X1-p.X1) < boxTol && math.Abs(g.Y1-p.Y1) < boxTol {
			match++
		}
	}
	return stageResult{Stage: "DLA", GoCount: len(goR), PyCount: len(pyR), MatchCount: match}
}
func compareBox(goB, pyB []jsonBox) stageResult {
	byPosBox(goB)
	byPosBox(pyB)
	n := len(goB)
	if len(pyB) < n {
		n = len(pyB)
	}
	match := 0
	for i := 0; i < n; i++ {
		g, p := goB[i], pyB[i]
		if math.Abs(g.X0-p.X0) < boxTol && math.Abs(g.Y0-p.Y0) < boxTol &&
			math.Abs(g.X1-p.X1) < boxTol && math.Abs(g.Y1-p.Y1) < boxTol &&
			math.Abs(g.X2-p.X2) < boxTol && math.Abs(g.Y2-p.Y2) < boxTol &&
			math.Abs(g.X3-p.X3) < boxTol && math.Abs(g.Y3-p.Y3) < boxTol {
			match++
		}
	}
	return stageResult{Stage: "OCRDetect", GoCount: len(goB), PyCount: len(pyB), MatchCount: match}
}
func compareCell(goC, pyC []jsonCell) stageResult {
	byPosCell(goC)
	byPosCell(pyC)
	n := len(goC)
	if len(pyC) < n {
		n = len(pyC)
	}
	match := 0
	for i := 0; i < n; i++ {
		g, p := goC[i], pyC[i]
		if math.Abs(g.X0-p.X0) < boxTol && math.Abs(g.Y0-p.Y0) < boxTol &&
			math.Abs(g.X1-p.X1) < boxTol && math.Abs(g.Y1-p.Y1) < boxTol {
			match++
		}
	}
	return stageResult{Stage: "TSR", GoCount: len(goC), PyCount: len(pyC), MatchCount: match}
}
func compareText(goT, pyT []jsonText) stageResult {
	n := len(goT)
	if len(pyT) < n {
		n = len(pyT)
	}
	match := 0
	var mae, maxd float64
	for i := 0; i < n; i++ {
		g, p := goT[i], pyT[i]
		if g.Text == p.Text {
			match++
		}
		d := math.Abs(g.Confidence - p.Confidence)
		mae += d
		if d > maxd {
			maxd = d
		}
	}
	if n > 0 {
		mae /= float64(n)
	}
	return stageResult{Stage: "OCRRecognize", GoCount: len(goT), PyCount: len(pyT), MatchCount: match, ConfMAE: mae, ConfMax: maxd}
}

func writeJSONFile(t *testing.T, dir, name string, v any) {
	t.Helper()
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatalf("marshal %s: %v", name, err)
	}
	if err := os.WriteFile(filepath.Join(dir, name), b, 0644); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
}

// TestInProcessVsServiceCompare renders each PDF page 0 with Go (pdfium) and
// compares the in-process backend's raw inference output against the Python
// service's, writing side-by-side JSONs and a summary report.
func TestInProcessVsServiceCompare(t *testing.T) {
	ortLib := os.Getenv("ORT_LIB")
	modelDir := os.Getenv("MODEL_DIR")
	pyURL := os.Getenv("DEEPDOC_URL")
	if pyURL == "" {
		pyURL = "http://localhost:9390"
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
	outDir := filepath.Join("testdata", "output", "render_compare")
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
	var results []pdfResult
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
		img, err := pdfpkg.RenderPageToImage(eng, 0)
		eng.Close()
		if err != nil {
			t.Logf("%s: render: %v", name, err)
			continue
		}
		base := strings.TrimSuffix(name, ".pdf")
		res := pdfResult{PDF: name}

		// DLA
		goDLA, err := goAna.DLA(ctx, img)
		if err != nil {
			t.Logf("%s DLA(go): %v", name, err)
		}
		pyDLA, err := pyClient.DLA(ctx, img)
		if err != nil {
			t.Logf("%s DLA(py): %v", name, err)
		}
		gd, pd := dlaFromDeep(goDLA), dlaFromPDF(pyDLA)
		writeJSONFile(t, outDir, base+"_go_dla.json", gd)
		writeJSONFile(t, outDir, base+"_py_dla.json", pd)
		res.Stages = append(res.Stages, compareDLA(gd, pd))

		// OCR Detect
		goDet, err := goAna.OCRDetect(ctx, img)
		if err != nil {
			t.Logf("%s Det(go): %v", name, err)
		}
		pyDet, err := pyClient.OCRDetect(ctx, img)
		if err != nil {
			t.Logf("%s Det(py): %v", name, err)
		}
		gb, pb := boxFromDeep(goDet), boxFromPDF(pyDet)
		writeJSONFile(t, outDir, base+"_go_ocr_detect.json", gb)
		writeJSONFile(t, outDir, base+"_py_ocr_detect.json", pb)
		res.Stages = append(res.Stages, compareBox(gb, pb))

		// OCR Recognize: crop the first go-detected box and send the SAME crop
		// to both backends so only recognition (not detection) is compared.
		if len(goDet) > 0 {
			b := goDet[0]
			// Use the bounding box of all four corners: for rotated text,
			// corner 0 (top-left) can have a larger x than corner 2
			// (bottom-right), so cropping (X0,Y0)-(X2,Y2) directly would
			// yield a degenerate (negative-width) region.
			x0 := int(math.Min(math.Min(b.X0, b.X1), math.Min(b.X2, b.X3)))
			y0 := int(math.Min(math.Min(b.Y0, b.Y1), math.Min(b.Y2, b.Y3)))
			x1 := int(math.Max(math.Max(b.X0, b.X1), math.Max(b.X2, b.X3)))
			y1 := int(math.Max(math.Max(b.Y0, b.Y1), math.Max(b.Y2, b.Y3)))
			crop := cropRect(img, x0, y0, x1, y1)
			goRec, err := goAna.OCRRecognize(ctx, crop)
			if err != nil {
				t.Logf("%s Rec(go): %v", name, err)
			}
			pyRec, err := pyClient.OCRRecognize(ctx, crop)
			if err != nil {
				t.Logf("%s Rec(py): %v", name, err)
			}
			gt, pt := textFromDeep(goRec), textFromPDF(pyRec)
			writeJSONFile(t, outDir, base+"_go_ocr_rec.json", gt)
			writeJSONFile(t, outDir, base+"_py_ocr_rec.json", pt)
			res.Stages = append(res.Stages, compareText(gt, pt))
		} else {
			res.Stages = append(res.Stages, stageResult{Stage: "OCRRecognize", Note: "no go-det box, skipped"})
		}

		// TSR: crop the first table region from the go DLA output.
		var tableRegion *deepdoctype.DLARegion
		for i := range goDLA {
			if goDLA[i].Label == "table" {
				tableRegion = &goDLA[i]
				break
			}
		}
		if tableRegion != nil {
			crop := cropRect(img, int(tableRegion.X0), int(tableRegion.Y0), int(tableRegion.X1), int(tableRegion.Y1))
			goTSR, err := goAna.TSR(ctx, crop)
			if err != nil {
				t.Logf("%s TSR(go): %v", name, err)
			}
			pyTSR, err := pyClient.TSR(ctx, crop)
			if err != nil {
				t.Logf("%s TSR(py): %v", name, err)
			}
			gc, pc := cellFromDeep(goTSR), cellFromPDF(pyTSR)
			writeJSONFile(t, outDir, base+"_go_tsr.json", gc)
			writeJSONFile(t, outDir, base+"_py_tsr.json", pc)
			res.Stages = append(res.Stages, compareCell(gc, pc))
		} else {
			res.Stages = append(res.Stages, stageResult{Stage: "TSR", Note: "no table region, skipped"})
		}

		results = append(results, res)
		t.Logf("%s: DLA %d/%d match=%d | Det %d/%d match=%d | Rec %d/%d match=%d (conf mae=%.3f max=%.3f) | TSR %d/%d match=%d",
			name,
			res.Stages[0].GoCount, res.Stages[0].PyCount, res.Stages[0].MatchCount,
			res.Stages[1].GoCount, res.Stages[1].PyCount, res.Stages[1].MatchCount,
			res.Stages[2].GoCount, res.Stages[2].PyCount, res.Stages[2].MatchCount, res.Stages[2].ConfMAE, res.Stages[2].ConfMax,
			res.Stages[3].GoCount, res.Stages[3].PyCount, res.Stages[3].MatchCount)
	}

	writeJSONFile(t, outDir, "compare_report.json", results)
	t.Logf("Wrote %s/compare_report.json for %d PDFs", outDir, len(results))
}
