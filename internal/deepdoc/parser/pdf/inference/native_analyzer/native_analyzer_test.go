//go:build native_det && integration

package infnative

import (
	"context"
	"encoding/json"
	"image"
	"image/draw"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"

	"native"
	"ragflow/internal/deepdoc/parser/pdf/inference"
	deepdoctype "ragflow/internal/deepdoc/parser/type"
)

// TestNativeAnalyzerInProcess proves the in-process (Go) DeepDoc backend
// actually runs end-to-end through the doctype.DocAnalyzer interface: ONNX
// Runtime init + model load + DLA/TSR/OCR inference on a real page fixture,
// producing non-empty, in-bounds results. It is the caller-side analogue of the
// native equivalence suite, but exercised through the DocAnalyzer seam the
// PDF parser consumes. Requires libonnxruntime + the InfiniFlow/deepdoc model
// snapshot; skipped unless both are reachable via ORT_LIB / MODEL_DIR.
//
// Run:
//
//	ORT_LIB=/path/libonnxruntime.so.1.23.2 \
//	  MODEL_DIR=/path/to/deepdoc \
//	  go test -tags "native_det integration" -run TestNativeAnalyzerInProcess \
//	  ./internal/deepdoc/parser/pdf/inference/native_analyzer/...
//
// TestNativeAnalyzerUninitializedNegative locks the fail-fast contract the
// server depends on (see registerNativeDeepDoc): before Register wires ONNX
// Runtime, the backend must report not-serving and NewAnalyzer must refuse to
// build. It exercises the negative branches of Serving/NewAnalyzer/Health that
// the happy-path test never hits. It makes no InitORT call, so it is safe to
// run first (ORT is process-global) and even when ORT_LIB/MODEL_DIR are unset.
func TestNativeAnalyzerUninitializedNegative(t *testing.T) {
	if Serving() {
		t.Fatal("Serving() reported true before any Register; backend must be inert until initialized")
	}
	modelDir := os.Getenv("MODEL_DIR")
	if modelDir == "" {
		modelDir = filepath.Join("..", "..", "..", "..", "rag", "res", "deepdoc")
	}
	if _, err := NewAnalyzer(modelDir, DefaultDropScore); err == nil {
		t.Error("NewAnalyzer succeeded before ONNX Runtime init; expected error")
	}
	a := &NativeAnalyzer{modelDir: modelDir}
	if a.Health() {
		t.Error("Health() reported healthy before ONNX Runtime init; expected false")
	}
}

func TestNativeAnalyzerInProcess(t *testing.T) {
	ortLib := os.Getenv("ORT_LIB")
	modelDir := os.Getenv("MODEL_DIR")
	if ortLib == "" || modelDir == "" {
		t.Skip("ORT_LIB and MODEL_DIR required (in-process backend integration)")
	}
	if err := Register(modelDir, ortLib, DefaultDropScore); err != nil {
		t.Fatalf("Register: %v", err)
	}
	if !Serving() {
		t.Skip("in-process backend not serving (ORT/models absent)")
	}
	a, err := NewAnalyzer(modelDir, DefaultDropScore)
	if err != nil {
		t.Fatalf("NewAnalyzer: %v", err)
	}

	// page0.png is a content page with a known DLA golden (see native
	// testdata). Reuse it to prove the DocAnalyzer path runs real inference.
	imgPath := filepath.Join("..", "..", "..", "..", "native", "testdata", "page0.png")
	f, err := os.Open(imgPath)
	if err != nil {
		t.Skipf("fixture unavailable: %v", err)
	}
	defer f.Close()
	src, err := png.Decode(f)
	if err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	b := src.Bounds()
	w, h := float64(b.Dx()), float64(b.Dy())

	ctx := context.Background()

	dla, err := a.DLA(ctx, src)
	if err != nil {
		t.Fatalf("DLA: %v", err)
	}
	if len(dla) == 0 {
		t.Error("DLA returned 0 regions on a content page; expected >0")
	}
	for _, r := range dla {
		if r.X1 < r.X0 || r.Y1 < r.Y0 {
			t.Errorf("DLA region has inverted bounds: %+v", r)
		}
		if r.X0 < 0 || r.Y0 < 0 || r.X1 > w || r.Y1 > h {
			t.Errorf("DLA region out of image bounds (%dx%d): %+v", int(w), int(h), r)
		}
		if r.Confidence < 0 || r.Confidence > 1 {
			t.Errorf("DLA confidence out of [0,1]: %+v", r)
		}
	}
	t.Logf("DLA: %d regions", len(dla))

	tsr, err := a.TSR(ctx, src)
	if err != nil {
		t.Fatalf("TSR: %v", err)
	}
	t.Logf("TSR: %d cells", len(tsr))

	det, err := a.OCRDetect(ctx, src)
	if err != nil {
		t.Fatalf("OCRDetect: %v", err)
	}
	if len(det) == 0 {
		t.Error("OCRDetect returned 0 boxes on a content page; expected >0")
	}
	for _, box := range det {
		if !quadInBounds(box, w, h) {
			t.Errorf("OCRDetect quad out of bounds (%dx%d): %+v", int(w), int(h), box)
		}
	}
	t.Logf("OCRDetect: %d boxes", len(det))

	// Crop a small region around the first detected box and recognize it,
	// proving OCRRecognize runs through the DocAnalyzer seam.
	if len(det) > 0 {
		crop := cropBox(src, det[0])
		rec, err := a.OCRRecognize(ctx, crop)
		if err != nil {
			t.Fatalf("OCRRecognize: %v", err)
		}
		t.Logf("OCRRecognize: %d text run(s)", len(rec))
	}
}

func quadInBounds(b deepdoctype.OCRBox, w, h float64) bool {
	xs := []float64{b.X0, b.X1, b.X2, b.X3}
	ys := []float64{b.Y0, b.Y1, b.Y2, b.Y3}
	for _, x := range xs {
		if x < -1 || x > w+1 {
			return false
		}
	}
	for _, y := range ys {
		if y < -1 || y > h+1 {
			return false
		}
	}
	return true
}

func cropBox(src image.Image, b deepdoctype.OCRBox) image.Image {
	minX, minY := b.X0, b.Y0
	maxX, maxY := b.X0, b.Y0
	for _, x := range []float64{b.X1, b.X2, b.X3} {
		minX = math.Min(minX, x)
		maxX = math.Max(maxX, x)
	}
	for _, y := range []float64{b.Y1, b.Y2, b.Y3} {
		minY = math.Min(minY, y)
		maxY = math.Max(maxY, y)
	}
	// Clamp to the source bounds so a box that pokes past an edge never
	// produces an out-of-range crop. The box coords are in the source image's
	// pixel space (same space OCRDetect reported them in).
	sb := src.Bounds()
	minX = math.Max(minX, float64(sb.Min.X))
	minY = math.Max(minY, float64(sb.Min.Y))
	maxX = math.Min(maxX, float64(sb.Max.X))
	maxY = math.Min(maxY, float64(sb.Max.Y))
	if maxX <= minX || maxY <= minY {
		return src
	}
	// OCRRecognize's OCRBoxes-from-image path assumes the image origin is
	// (0,0), so draw the crop into a fresh zero-origin image rather than
	// returning a SubImage (which would keep a non-zero Min and underflow the
	// pixel index in FromImage).
	r := image.Rect(int(math.Floor(minX)), int(math.Floor(minY)),
		int(math.Ceil(maxX)), int(math.Ceil(maxY)))
	out := image.NewRGBA(image.Rect(0, 0, r.Dx(), r.Dy()))
	draw.Draw(out, out.Bounds(), src, r.Min, draw.Src)
	return out
}

// === Value-level golden equivalence for the in-process DocAnalyzer seam ===
//
// The tests below prove the NativeAnalyzer (the DocAnalyzer the PDF parser
// actually consumes) produces output equivalent to the Python deepdoc
// reference, reusing the SAME Python-reference goldens as the native
// integration suite. This closes the gap noted in EQUIVALENCE.md: the in-
// process backend previously only had a smoke test (non-empty, in-bounds);
// these tests assert value-level parity through the analyzer's public API.

// goldenPath resolves a native testdata fixture from this package's test
// directory (internal/deepdoc/parser/pdf/inference/native_analyzer). Four ".." climb to
// internal/deepdoc, where the native module lives.
func goldenPath(name string) string {
	return filepath.Join("..", "..", "..", "..", "native", "testdata", name)
}

// openFixture decodes a PNG fixture the way the production path does (Go's
// image decode -> NativeAnalyzer), so the comparison exercises the real server
// code path rather than the native internal decoder.
func openFixture(t *testing.T, stem string) image.Image {
	t.Helper()
	f, err := os.Open(goldenPath(stem + ".png"))
	if err != nil {
		t.Skipf("fixture unavailable: %v", err)
	}
	defer f.Close()
	src, err := png.Decode(f)
	if err != nil {
		t.Fatalf("decode fixture %s: %v", stem, err)
	}
	return src
}

// labelKey maps a layout/TSR label to a stable integer key so the analyzer's
// string Label can be matched against the golden's integer class under the
// same key space. Duplicate labels (DLA has two "table caption" entries) map to
// their first index on BOTH sides, mirroring the analyzer's class->label
// expansion.
func labelKey(labels []string, label string) int {
	for i, l := range labels {
		if l == label {
			return i
		}
	}
	return -1
}

// analyzerWithModels builds a NativeAnalyzer after ensuring ONNX Runtime is
// initialized. It mirrors skipIfNoModels in the native suite: the test is
// skipped (not failed) when ORT_LIB/MODEL_DIR are unset, and InitORT is
// idempotent so it composes with the other analyzer tests in this file.
func analyzerWithModels(t *testing.T) *NativeAnalyzer {
	t.Helper()
	ortLib := os.Getenv("ORT_LIB")
	modelDir := os.Getenv("MODEL_DIR")
	if ortLib == "" || modelDir == "" {
		t.Skip("ORT_LIB and MODEL_DIR required (in-process backend integration)")
	}
	if err := native.InitORT(ortLib); err != nil {
		t.Fatalf("InitORT: %v", err)
	}
	a, err := NewAnalyzer(modelDir, DefaultDropScore)
	if err != nil {
		t.Fatalf("NewAnalyzer: %v", err)
	}
	return a
}

// TestAnalyzerDLAGolden proves the analyzer's DLA output matches the Python
// reference golden (class + coordinates + confidence) within the documented
// sub-pixel floor, across the same fixtures the native suite uses.
func TestAnalyzerDLAGolden(t *testing.T) {
	a := analyzerWithModels(t)
	ctx := context.Background()
	labels := inference.DefaultDLALabels()
	stems := []string{"page0", "mp_textbook_en_p0", "mp_whitepaper_cn_p0", "mp_paper_eq_p0", "mp_zhtw_ent_p0",
		"dla_2510_figcap", "dla_bookrag_figcap", "dla_2510_eq",
		"dla_real_cn_report", "dla_real_zhtw", "dla_real_en_paper"}
	for _, stem := range stems {
		stem := stem
		t.Run(stem, func(t *testing.T) {
			img := openFixture(t, stem)
			regions, err := a.DLA(ctx, img)
			if err != nil {
				t.Fatalf("DLA: %v", err)
			}
			got := make([][]float64, 0, len(regions))
			for _, r := range regions {
				got = append(got, []float64{r.X0, r.Y0, r.X1, r.Y1, r.Confidence, float64(labelKey(labels, r.Label))})
			}
			gold := native.LoadGoldenBoxes(t, goldenPath(stem+".dla.golden.json"))
			// Rewrite the golden's integer class to the same label key so both
			// sides match on the analyzer's label semantics.
			for i := range gold {
				gold[i][5] = float64(labelKey(labels, labels[int(gold[i][5])]))
			}
			matched, maxd, unmatched := native.MatchBoxesRelaxed(t, gold, got, 2.0, native.CmpTolScore)
			t.Logf("DLA %s: matched %d/%d, maxd %.3f px, unmatched %d", stem, matched, len(gold), maxd, len(unmatched))
			if matched != len(gold) {
				t.Errorf("DLA %s: matched %d/%d golden regions", stem, matched, len(gold))
			}
		})
	}
}

// TestAnalyzerTSRGolden proves the analyzer's TSR output matches the Python
// reference golden (structure: which cells are table/column/row) within the
// documented floor. The analyzer does not expose a TSR score, so the score
// tolerance is widened to ignore it; only class + coordinates are asserted.
func TestAnalyzerTSRGolden(t *testing.T) {
	a := analyzerWithModels(t)
	ctx := context.Background()
	labels := inference.DefaultTSRLabels()
	stems := []string{"table0", "tsr_table_normal", "tsr_table_rotation",
		"tsr_06_table_content", "tsr_18_table_caption", "tsr_13_crosspage", "tsr_14_interleaved",
		"tsr_real_report"}
	for _, stem := range stems {
		stem := stem
		t.Run(stem, func(t *testing.T) {
			img := openFixture(t, stem)
			cells, err := a.TSR(ctx, img)
			if err != nil {
				t.Fatalf("TSR: %v", err)
			}
			got := make([][]float64, 0, len(cells))
			for _, c := range cells {
				got = append(got, []float64{c.X0, c.Y0, c.X1, c.Y1, 0, float64(labelKey(labels, c.Label))})
			}
			gold := native.LoadGoldenBoxes(t, goldenPath(stem+".tsr.golden.json"))
			for i := range gold {
				gold[i][5] = float64(labelKey(labels, labels[int(gold[i][5])]))
			}
			matched, maxd, unmatched := native.MatchBoxesRelaxed(t, gold, got, native.CmpTolCoord, 1.0)
			t.Logf("TSR %s: matched %d/%d, maxd %.3f px, unmatched %d", stem, matched, len(gold), maxd, len(unmatched))
			if matched != len(gold) {
				t.Errorf("TSR %s: matched %d/%d golden cells", stem, matched, len(gold))
			}
		})
	}
}

// TestAnalyzerOCRRecGolden proves the analyzer's OCR text recognition matches
// the Python reference golden exactly (EN / CJK / mixed / digits).
func TestAnalyzerOCRRecGolden(t *testing.T) {
	a := analyzerWithModels(t)
	ctx := context.Background()
	stems := []string{"line0", "line_cn", "line_mix", "line_num",
		"line_real_cn", "line_real_zhtw", "line_real_en"}
	for _, stem := range stems {
		stem := stem
		t.Run(stem, func(t *testing.T) {
			img := openFixture(t, stem)
			rec, err := a.OCRRecognize(ctx, img)
			if err != nil {
				t.Fatalf("OCRRecognize: %v", err)
			}
			gold := ocrRecGoldText(t, goldenPath(stem+".ocr_rec.golden.json"))
			got := ""
			if len(rec) > 0 {
				got = rec[0].Text
			}
			if got != gold {
				t.Errorf("OCR-rec %s: got %q, gold %q", stem, got, gold)
			}
		})
	}
}

// TestAnalyzerDetGolden proves the analyzer's text detection matches the Python
// reference golden on page0: every Python box has a Go twin by center within the
// documented floor, and Go does not hallucinate beyond the accepted 3/5 orphan
// floor.
func TestAnalyzerDetGolden(t *testing.T) {
	a := analyzerWithModels(t)
	ctx := context.Background()
	img := openFixture(t, "page0")
	boxes, err := a.OCRDetect(ctx, img)
	if err != nil {
		t.Fatalf("OCRDetect: %v", err)
	}
	got := make([][][2]float64, 0, len(boxes))
	for _, b := range boxes {
		got = append(got, [][2]float64{{b.X0, b.Y0}, {b.X1, b.Y1}, {b.X2, b.Y2}, {b.X3, b.Y3}})
	}
	raw, err := os.ReadFile(goldenPath("page0.det.golden.json"))
	if err != nil {
		t.Skipf("golden unavailable: %v", err)
	}
	var gold struct {
		Output [][][][][2]float64 `json:"output"`
	}
	if err := json.Unmarshal(raw, &gold); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	goldBoxes := native.FlattenQuads(gold.Output)
	mG, mGo, maxd := native.MatchBothDirections(goldBoxes, got, native.CmpTolCoord)
	imG, imGo := native.MatchIoUBothDirections(goldBoxes, got, 0.5)
	t.Logf("Det page0: center matched(g/g)=%d/%d maxd=%.3f px | IoU orphan(g/g)=%d/%d",
		mG, mGo, maxd, len(goldBoxes)-imG, len(got)-imGo)
	if mG != len(goldBoxes) {
		t.Errorf("Det page0: %d/%d golden boxes matched by center (missing %d)", mG, len(goldBoxes), len(goldBoxes)-mG)
	}
	const detOrphanSlack = 5 // accepted IoU floor (3/5) + slack
	if len(got)-mGo > detOrphanSlack {
		t.Errorf("Det page0: %d unmatched Go boxes (got %d) exceeds accepted floor %d", len(got)-mGo, len(got), detOrphanSlack)
	}
}

// ocrRecGoldText reads the recognized text from an OCR-rec golden JSON
// ({"output": [[[["<text>"]]]]}).
func ocrRecGoldText(t *testing.T, path string) string {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("golden unavailable: %v", err)
	}
	var gold struct {
		Output [][][][]any `json:"output"`
	}
	if err := json.Unmarshal(raw, &gold); err != nil {
		t.Fatalf("parse golden %s: %v", path, err)
	}
	return gold.Output[0][0][0][0].(string)
}
