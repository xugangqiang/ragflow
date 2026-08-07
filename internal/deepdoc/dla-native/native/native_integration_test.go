//go:build integration

package native

// Integration tests run the real ONNX models on the committed test crops and
// compare the Go output against golden JSON produced by the Python reference
// scripts (ref_dla.py / ref_tsr.py / ref_ocr_rec.py). They require ORT_LIB and
// MODEL_DIR to point at the onnxruntime shared library and the DeepDoc model
// directory (layout.onnx, tsr.onnx, rec.onnx, ocr.res).
//
// Run with:
//   ORT_LIB=... MODEL_DIR=... go test -tags integration ./native/...

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

const (
	// px tolerance on box coordinates. The pure-Go build carries an inherent
	// decode/resize fidelity floor (~3px on real-document fixtures, the same
	// class of artifact as det's 3px floor) that this tolerance sits just above.
	cmpTolCoord = 3.0
	cmpTolScore = 0.05 // tolerance on detection scores
)

func skipIfNoModels(t *testing.T) {
	if os.Getenv("ORT_LIB") == "" || os.Getenv("MODEL_DIR") == "" {
		t.Skip("set ORT_LIB and MODEL_DIR to run integration tests")
	}
	if err := InitORT(os.Getenv("ORT_LIB")); err != nil {
		t.Fatalf("InitORT: %v", err)
	}
}

// compareBoxes matches every golden box to a Go box of the same class by
// nearest center and reports the max per-coordinate difference.
func compareBoxes(t *testing.T, gold, got [][]float64) {
	t.Helper()
	if len(gold) == 0 {
		t.Fatalf("golden has no boxes")
	}
	used := make([]bool, len(got))
	maxd := 0.0
	matched := 0
	for _, gb := range gold {
		cls := int(gb[5])
		bcx, bcy := (gb[0]+gb[2])/2, (gb[1]+gb[3])/2
		best, bd := -1, math.MaxFloat64
		for i, vb := range got {
			if used[i] || int(vb[5]) != cls {
				continue
			}
			vcx, vcy := (vb[0]+vb[2])/2, (vb[1]+vb[3])/2
			d := (bcx-vcx)*(bcx-vcx) + (bcy-vcy)*(bcy-vcy)
			if d < bd {
				bd, best = d, i
			}
		}
		if best < 0 {
			t.Errorf("no Go box matched golden class %d at (%.0f,%.0f)", cls, bcx, bcy)
			continue
		}
		used[best] = true
		matched++
		for j := 0; j < 6; j++ {
			tol := cmpTolCoord
			if j == 4 {
				tol = cmpTolScore
			}
			if math.Abs(gb[j]-got[best][j]) > tol {
				t.Errorf("class %d coord %d diff %.3f > tol %.2f (gold=%v got=%v)",
					cls, j, math.Abs(gb[j]-got[best][j]), tol, gb, got[best])
			}
			if j != 4 {
				maxd = math.Max(maxd, math.Abs(gb[j]-got[best][j]))
			}
		}
	}
	t.Logf("matched %d/%d golden boxes, max coord diff %.4f px", matched, len(gold), maxd)
}

// matchBoxesRelaxed returns (matched count, max coordinate diff among matches,
// unmatched goldens) using caller-supplied tolerances. Unlike compareBoxes it
// does NOT fail the test — callers decide what a match/mismatch means. A
// golden box counts as matched only if its nearest same-class Go box is within
// coordTol (on any coordinate) and scoreTol; otherwise it is returned as
// unmatched. Used by the extreme-aspect boundary test, whose tolerances are
// deliberately wider than the real-table parity floor.
func matchBoxesRelaxed(t *testing.T, gold, got [][]float64, coordTol, scoreTol float64) (matched int, maxd float64, unmatched [][]float64) {
	t.Helper()
	used := make([]bool, len(got))
	for _, gb := range gold {
		cls := int(gb[5])
		bcx, bcy := (gb[0]+gb[2])/2, (gb[1]+gb[3])/2
		best, bd := -1, math.MaxFloat64
		for i, vb := range got {
			if used[i] || int(vb[5]) != cls {
				continue
			}
			vcx, vcy := (vb[0]+vb[2])/2, (vb[1]+vb[3])/2
			d := (bcx-vcx)*(bcx-vcx) + (bcy-vcy)*(bcy-vcy)
			if d < bd {
				bd, best = d, i
			}
		}
		if best < 0 {
			unmatched = append(unmatched, gb)
			continue
		}
		// Enforce the relaxed tolerance: if even the nearest same-class box is
		// farther than the tolerance, treat it as unmatched (structural miss).
		coordDiff, scoreDiff := 0.0, math.Abs(gb[4]-got[best][4])
		for j := 0; j < 4; j++ {
			coordDiff = math.Max(coordDiff, math.Abs(gb[j]-got[best][j]))
		}
		if coordDiff > coordTol || scoreDiff > scoreTol {
			unmatched = append(unmatched, gb)
			continue
		}
		used[best] = true
		matched++
		maxd = math.Max(maxd, coordDiff)
	}
	return matched, maxd, unmatched
}

func loadGoldenBoxes(t *testing.T, path string) [][]float64 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read golden %s: %v", path, err)
	}
	var gold [][]float64
	if err := json.Unmarshal(raw, &gold); err != nil {
		t.Fatalf("parse golden %s: %v", path, err)
	}
	return gold
}

// dlaPages / tsrPages / ocrRecLines enumerate the fixtures the Go port is
// compared against (golden JSON produced by the Python reference). page0 /
// table0 / line0 are the original single-page baselines; the mp_* pages broaden
// DLA/DET coverage to English textbooks, CN whitepapers, equation-heavy papers,
// and ZH-TW enterprise docs. TSR is validated only on real tables (table0 plus
// two crops from table_rotation_test.pdf) — the mp_* pages are not tables and
// only produce whole-page false detections, so they are excluded from tsrPages.
// tsr_table_normal is a moderate 2.65:1 table; tsr_table_rotation is a 1:6.3
// tall rotated table. Both sit comfortably under the 3px tolerance.
var dlaPages = []string{"page0", "mp_textbook_en_p0", "mp_whitepaper_cn_p0", "mp_paper_eq_p0", "mp_zhtw_ent_p0"}
var tsrPages = []string{"table0", "tsr_table_normal", "tsr_table_rotation"}

// ocrRecLines covers EN (regular text + bold/italic/serif font variants of the
// same sentence to exercise font robustness), pure CJK, mixed EN+CJK, and a
// digits/symbols/CJK line. All texts are confined to the model's vocab (basic
// Latin + Chinese + digits/symbols); scripts the model cannot read (kana,
// Cyrillic, accented Latin) are intentionally excluded — the rec dict has
// neither, so goldens would be pure garbage and add no signal.
var ocrRecLines = []string{
	"line0", "line_cn",
	"line_en_bold", "line_en_italic", "line_en_serif",
	"line_mix", "line_num", "line_cn_long",
}

func TestDLAIntegration(t *testing.T) {
	skipIfNoModels(t)
	for _, stem := range dlaPages {
		stem := stem
		t.Run(stem, func(t *testing.T) {
			img, err := Decode(filepath.Join("..", "testdata", stem+".png"))
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			res, err := RunDLA(t.Context(), os.Getenv("MODEL_DIR"), img)
			if err != nil {
				t.Fatalf("RunDLA: %v", err)
			}
			var got struct {
				Boxes [][]float64 `json:"bboxes"`
			}
			if err := json.Unmarshal([]byte(res.Wire()), &got); err != nil {
				t.Fatalf("parse Go wire: %v", err)
			}
			gold := loadGoldenBoxes(t, filepath.Join("..", "testdata", stem+".dla.golden.json"))
			compareBoxes(t, gold, got.Boxes)
		})
	}
}

func TestTSRIntegration(t *testing.T) {
	skipIfNoModels(t)
	for _, stem := range tsrPages {
		stem := stem
		t.Run(stem, func(t *testing.T) {
			img, err := Decode(filepath.Join("..", "testdata", stem+".png"))
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			// TSR decodes via the pure-Go path (tsr_decode.go), matching
			// deepdoc's PIL TSR decode.
			res, err := RunTSR(t.Context(), os.Getenv("MODEL_DIR"), img)
			if err != nil {
				t.Fatalf("RunTSR: %v", err)
			}
			var got struct {
				Boxes [][]float64 `json:"bboxes"`
			}
			if err := json.Unmarshal([]byte(res.Wire()), &got); err != nil {
				t.Fatalf("parse Go wire: %v", err)
			}
			gold := loadGoldenBoxes(t, filepath.Join("..", "testdata", stem+".tsr.golden.json"))
			compareBoxes(t, gold, got.Boxes)
		})
	}
}

// TestTSRExtremeAspect locks the known decode-floor behavior on extreme-aspect
// tables. The tsr_table_caption crop is ~4:1, so the model's 640x640 input
// squishes x by ~1.45x; the residual Go-vs-PIL JPEG decode difference is then
// amplified 1.45x on the way back to pixels, yielding ~8px box shifts (vs
// <1px on moderate tables). This is NOT a logic divergence — it is the same
// floor the 3px-tolerance real-table fixtures sit under, just scaled up by the
// aspect ratio. So this test uses a relaxed 10px tolerance and asserts only
// that the *structure* survives: the table (class 0) and all columns (class 1)
// must match, row count stays within ±1 of golden, and ONLY a near-threshold
// row (class 2, score<0.30) may be dropped — never a table or a column. That
// catches the real regression risk (hallucinated/missed table or column) while
// documenting the accepted floor amplification rather than hiding it.
func TestTSRExtremeAspect(t *testing.T) {
	skipIfNoModels(t)
	img, err := Decode(filepath.Join("..", "testdata", "tsr_table_caption.png"))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	res, err := RunTSR(t.Context(), os.Getenv("MODEL_DIR"), img)
	if err != nil {
		t.Fatalf("RunTSR: %v", err)
	}
	var got struct {
		Boxes [][]float64 `json:"bboxes"`
	}
	if err := json.Unmarshal([]byte(res.Wire()), &got); err != nil {
		t.Fatalf("parse Go wire: %v", err)
	}
	gold := loadGoldenBoxes(t, filepath.Join("..", "testdata", "tsr_table_caption.tsr.golden.json"))

	const (
		relaxCoord = 10.0
		relaxScore = 0.10
	)
	matched, maxd, unmatched := matchBoxesRelaxed(t, gold, got.Boxes, relaxCoord, relaxScore)
	t.Logf("extreme-aspect: matched %d/%d, max coord diff %.3f px, unmatched=%d",
		matched, len(gold), maxd, len(unmatched))
	for _, u := range unmatched {
		// Only a near-threshold ROW may be dropped; a missing table/column is a
		// real regression (the floor should not erase a whole structural box).
		if int(u[5]) != 2 || u[4] >= 0.30 {
			t.Errorf("unmatched non-row or high-score box: class %d score %.3f", int(u[5]), u[4])
		}
	}
	// No hallucinated boxes: the near-threshold row is the only thing that can
	// go missing, so Go must never exceed the golden count.
	if len(got.Boxes) > len(gold) {
		t.Errorf("hallucinated boxes: got %d > golden %d", len(got.Boxes), len(gold))
	}
	// The table (class 0) must always be detected.
	hasTable := false
	for _, b := range got.Boxes {
		if int(b[5]) == 0 {
			hasTable = true
		}
	}
	if !hasTable {
		t.Errorf("extreme-aspect table (class 0) not detected")
	}
}

func TestOCRRecIntegration(t *testing.T) {
	skipIfNoModels(t)
	for _, stem := range ocrRecLines {
		stem := stem
		t.Run(stem, func(t *testing.T) {
			img, err := Decode(filepath.Join("..", "testdata", stem+".png"))
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			res, err := RunOCRRec(t.Context(), os.Getenv("MODEL_DIR"), img)
			if err != nil {
				t.Fatalf("RunOCRRec: %v", err)
			}
			var got struct {
				Output [][][][]any `json:"output"`
			}
			if err := json.Unmarshal([]byte(res.Wire()), &got); err != nil {
				t.Fatalf("parse Go wire: %v", err)
			}
			raw, err := os.ReadFile(filepath.Join("..", "testdata", stem+".ocr_rec.golden.json"))
			if err != nil {
				t.Fatalf("read golden: %v", err)
			}
			var gold struct {
				Output [][][][]any `json:"output"`
			}
			if err := json.Unmarshal(raw, &gold); err != nil {
				t.Fatalf("parse golden: %v", err)
			}
			gotText := got.Output[0][0][0][0].(string)
			goldText := gold.Output[0][0][0][0].(string)
			if gotText != goldText {
				t.Fatalf("OCR-rec text mismatch: got %q, gold %q", gotText, goldText)
			}
			t.Logf("OCR-rec text matches: %q", gotText)
		})
	}
}

// The three recognizers below share the fixed-shape model-session pool. Each
// test runs the recognizer twice on the same crop and asserts byte-identical
// wire output, proving getModelSession hands back a clean session after release
// (no stale input tensor, no cross-call contamination) and that pooling is a
// no-behavior-change over the old per-call NewSession path.

func TestDLASessionReuse(t *testing.T) {
	skipIfNoModels(t)
	img, err := Decode(filepath.Join("..", "testdata", "page0.png"))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	r1, err := RunDLA(t.Context(), os.Getenv("MODEL_DIR"), img)
	if err != nil {
		t.Fatalf("RunDLA #1: %v", err)
	}
	r2, err := RunDLA(t.Context(), os.Getenv("MODEL_DIR"), img)
	if err != nil {
		t.Fatalf("RunDLA #2: %v", err)
	}
	if r1.Wire() != r2.Wire() {
		t.Fatalf("DLA output changed across pooled runs (session reuse not stable)")
	}
}

func TestTSRSessionReuse(t *testing.T) {
	skipIfNoModels(t)
	img, err := Decode(filepath.Join("..", "testdata", "table0.png"))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	r1, err := RunTSR(t.Context(), os.Getenv("MODEL_DIR"), img)
	if err != nil {
		t.Fatalf("RunTSR #1: %v", err)
	}
	r2, err := RunTSR(t.Context(), os.Getenv("MODEL_DIR"), img)
	if err != nil {
		t.Fatalf("RunTSR #2: %v", err)
	}
	if r1.Wire() != r2.Wire() {
		t.Fatalf("TSR output changed across pooled runs (session reuse not stable)")
	}
}

func TestOCRRecSessionReuse(t *testing.T) {
	skipIfNoModels(t)
	img, err := Decode(filepath.Join("..", "testdata", "line0.png"))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	r1, err := RunOCRRec(t.Context(), os.Getenv("MODEL_DIR"), img)
	if err != nil {
		t.Fatalf("RunOCRRec #1: %v", err)
	}
	r2, err := RunOCRRec(t.Context(), os.Getenv("MODEL_DIR"), img)
	if err != nil {
		t.Fatalf("RunOCRRec #2: %v", err)
	}
	if r1.Wire() != r2.Wire() {
		t.Fatalf("OCR-rec output changed across pooled runs (session reuse not stable)")
	}
}

func TestDetIntegration(t *testing.T) {
	skipIfNoModels(t)
	imgPath := filepath.Join("..", "testdata", "page0.png")
	img, err := Decode(imgPath)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	res, err := RunDet(t.Context(), os.Getenv("MODEL_DIR"), img)
	if err != nil {
		t.Fatalf("RunDet: %v", err)
	}
	// Go wire: {"output": [[ [ [x,y]*4, ... ] ]]}; boxes at output[0][0].
	var got struct {
		Output [][][][][2]float64 `json:"output"`
	}
	if err := json.Unmarshal([]byte(res.Wire()), &got); err != nil {
		t.Fatalf("parse Go wire: %v", err)
	}
	if len(got.Output) == 0 || len(got.Output[0]) == 0 {
		t.Fatalf("Go wire missing boxes")
	}
	goBoxes := got.Output[0][0]

	raw, err := os.ReadFile(filepath.Join("..", "testdata", "page0.det.golden.json"))
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var gold struct {
		Output [][][][][2]float64 `json:"output"`
	}
	if err := json.Unmarshal(raw, &gold); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if len(gold.Output) == 0 || len(gold.Output[0]) == 0 {
		t.Fatalf("golden missing boxes")
	}
	refBoxes := gold.Output[0][0]

	// The pure-Go geometry reaches the 3px hard floor (HANDOFF §4.4, box#8).
	// Tolerance is set just above that floor so a regression bumps it over the
	// line.
	detCoordTol := 3.5
	used := make([]bool, len(goBoxes))
	matched, maxd := 0, 0.0
	for _, rb := range refBoxes {
		rcx, rcy := quadCenter(rb)
		best, bd := -1, math.MaxFloat64
		for i, vb := range goBoxes {
			if used[i] {
				continue
			}
			vcx, vcy := quadCenter(vb)
			d := (rcx-vcx)*(rcx-vcx) + (rcy-vcy)*(rcy-vcy)
			if d < bd {
				bd, best = d, i
			}
		}
		if best < 0 {
			t.Errorf("no Go box matched golden quad at (%.0f,%.0f)", rcx, rcy)
			continue
		}
		used[best] = true
		matched++
		for j := 0; j < 4; j++ {
			for k := 0; k < 2; k++ {
				diff := math.Abs(rb[j][k] - goBoxes[best][j][k])
				if diff > detCoordTol {
					t.Errorf("quad coord diff %.3f > tol %.2f (gold=%v got=%v)",
						diff, detCoordTol, rb, goBoxes[best])
				}
				if diff > maxd {
					maxd = diff
				}
			}
		}
	}
	t.Logf("det: matched %d/%d golden quads, max coord diff %.4f px (tol %.1f)",
		matched, len(refBoxes), maxd, detCoordTol)
	if matched != len(refBoxes) {
		t.Errorf("matched %d/%d quads", matched, len(refBoxes))
	}
}

// TestDetSessionPoolBounded is the regression guard for the unbounded
// sync.Map leak: a long-running server ingesting many differently-sized pages
// used to pin one pool + cached tensors per unique (modelPath, rh, rw) forever.
// It now bounds the pool set (detMaxShapePools). We drive far more distinct
// page sizes than the cap and assert the live pool count never exceeds it.
func TestDetSessionPoolBounded(t *testing.T) {
	skipIfNoModels(t)
	modelDir := os.Getenv("MODEL_DIR")

	const n = 80
	for i := 0; i < n; i++ {
		w := 64 + i*14
		h := 128 + (i*9)%500
		if w > 952 {
			w = 952
		}
		if h > 952 {
			h = 952
		}
		// Synthetic gray raster; we only exercise the pool lifecycle here, not
		// detection quality, so the content is irrelevant.
		img := &Image{W: w, H: h, Pix: make([]byte, w*h*3)}
		for j := 0; j < len(img.Pix); j += 3 {
			img.Pix[j], img.Pix[j+1], img.Pix[j+2] = 200, 200, 200
		}
		if _, err := RunDet(t.Context(), modelDir, img); err != nil {
			t.Fatalf("RunDet #%d (%dx%d): %v", i, w, h, err)
		}
	}

	got := detSessions.KeyCount()
	if got > detMaxShapePools {
		t.Fatalf("det session pool set grew unbounded: %d pools (cap %d) after %d distinct sizes",
			got, detMaxShapePools, n)
	}
	t.Logf("det session pool set bounded: %d pools (cap %d) after %d distinct sizes", got, detMaxShapePools, n)
}

func quadCenter(q [][2]float64) (float64, float64) {
	var sx, sy float64
	for _, p := range q {
		sx += p[0]
		sy += p[1]
	}
	return sx / float64(len(q)), sy / float64(len(q))
}
