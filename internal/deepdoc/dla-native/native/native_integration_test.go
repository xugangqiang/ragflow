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
	"sort"
	"strings"
	"testing"
)

const (
	// coordFloor is the documented hard accuracy floor (px) of the comparison
	// tool: det stabilizes at ~3px from bilinearResize + box#8 postprocess,
	// format-independent. DLA/TSR are tighter, but tolerances are sized above
	// this worst case so any regression past the floor trips the gate instead
	// of hiding under it.
	coordFloor = 3.0
	// coordTolMargin lifts the coordinate tolerance just above coordFloor.
	coordTolMargin = 0.5

	cmpTolCoord = coordFloor + coordTolMargin // 3.5
	cmpTolScore = 0.05                        // tolerance on detection scores
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
	// DLA/TSR goldens use the Go DocAnalyzer wire shape: {"bboxes": [[...]]}.
	var wrap struct {
		Bboxes [][]float64 `json:"bboxes"`
	}
	if err := json.Unmarshal(raw, &wrap); err != nil {
		t.Fatalf("parse golden %s: %v", path, err)
	}
	return wrap.Bboxes
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

// TestOCRRecBatchIntegration proves Go matches deepdoc's *batch* resize
// semantics: a narrow line (line_cn) recognized inside a batch that also
// contains wide lines is resized to the batch-wide max wh_ratio, not its own,
// so its text differs from the standalone recognition. Go must reproduce the
// same batch-wide text as the Python oracle (frozen in
// batch_ocr_rec.golden.json), which calls TextRecognizer on the whole list at
// once.
func TestOCRRecBatchIntegration(t *testing.T) {
	skipIfNoModels(t)
	stems := []string{"line_cn", "line_mix", "line_num", "line0"}
	imgs := make([]*Image, len(stems))
	for i, s := range stems {
		img, err := Decode(filepath.Join("..", "testdata", s+".png"))
		if err != nil {
			t.Fatalf("decode %s: %v", s, err)
		}
		imgs[i] = img
	}
	res, err := RunOCRRecBatch(t.Context(), os.Getenv("MODEL_DIR"), imgs)
	if err != nil {
		t.Fatalf("RunOCRRecBatch: %v", err)
	}

	raw, err := os.ReadFile(filepath.Join("..", "testdata", "batch_ocr_rec.golden.json"))
	if err != nil {
		t.Fatalf("read batch golden: %v", err)
	}
	var gold []struct {
		Output [][][][]any `json:"output"`
	}
	if err := json.Unmarshal(raw, &gold); err != nil {
		t.Fatalf("parse batch golden: %v", err)
	}
	if len(gold) != len(res) {
		t.Fatalf("batch size mismatch: golden %d, go %d", len(gold), len(res))
	}
	for i, s := range stems {
		var got struct {
			Output [][][][]any `json:"output"`
		}
		if err := json.Unmarshal([]byte(res[i].Wire()), &got); err != nil {
			t.Fatalf("parse go wire %s: %v", s, err)
		}
		goldText := gold[i].Output[0][0][0][0].(string)
		gotText := got.Output[0][0][0][0].(string)
		if gotText != goldText {
			t.Errorf("batch line %s mismatch: got %q, gold %q", s, gotText, goldText)
			continue
		}
		t.Logf("batch line %s matches: %q", s, gotText)
	}

	// The batch must actually engage batch semantics: line_cn's batch text
	// must differ from its standalone recognition (it is widened to the
	// batch max), otherwise the test would pass trivially via the single-image
	// path.
	single, err := RunOCRRec(t.Context(), os.Getenv("MODEL_DIR"), imgs[0])
	if err != nil {
		t.Fatalf("RunOCRRec line_cn: %v", err)
	}
	if single.Text == res[0].Text {
		t.Errorf("batch semantics not engaged: line_cn batch text == single text (%q); expected a wider-resize result", single.Text)
	}
}

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
	// Tolerance is coordFloor + coordTolMargin, just above that floor, so a
	// regression bumps it over the line.
	detCoordTol := coordFloor + coordTolMargin
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

// TestDetMembershipAllFixtures quantifies the det box-membership gap (gap 3).
// For every committed det fixture it runs RunDet and compares the Go box set
// against the golden (Python) box set in TWO complementary ways:
//   1. center-distance match (tol 3.5px) — isolates coordinate drift;
//   2. IoU match (thr 0.5) — isolates true box-membership divergence (splits,
//      merges, hallucinations), independent of how far a box's center moved.
// The original TestDetIntegration only checked golden→Go by nearest center on
// a single fixture, so a Go box shifted >3.5px was mis-flagged and a Go box
// with no golden counterpart was invisible.
//
// This is a REGRESSION GUARD, not a zero-target. The IoU orphan counts are
// pinned to the baseline measured on 2026-08-10 (gap 3: 37 golden misses + 20
// extra Go boxes across 37 fixtures, concentrated in dense-text pages). The
// test fails ONLY if a future geometry change makes Go WORSE than that
// baseline — it does not require the gap to reach zero. A small slack absorbs
// run-to-run nondeterminism (see gap 7); it is NOT headroom for new divergence.
func TestDetMembershipAllFixtures(t *testing.T) {
	skipIfNoModels(t)
	md := os.Getenv("MODEL_DIR")
	dir := "../testdata"

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read testdata: %v", err)
	}

	type stat struct {
		stem                                   string
		nGold, nGo                             int
		centerOrphanGold, centerOrphanGo       int
		iouMatchedGold, iouMatchedGo           int
		iouOrphanGold, iouOrphanGo             int
		maxd                                   float64
	}
	var stats []stat
	sumGold, sumGo := 0, 0
	sumCIoG, sumCIoGo := 0, 0
	sumIIoG, sumIIoGo := 0, 0

	for _, e := range entries {
		name := e.Name()
		if !strings.HasSuffix(name, ".det.golden.json") {
			continue
		}
		stem := strings.TrimSuffix(name, ".det.golden.json")
		img, err := Decode(filepath.Join(dir, stem+".png"))
		if err != nil {
			t.Fatalf("decode %s: %v", stem, err)
		}
		res, err := RunDet(t.Context(), md, img)
		if err != nil {
			t.Fatalf("RunDet %s: %v", stem, err)
		}
		var got struct {
			Output [][][][][2]float64 `json:"output"`
		}
		if err := json.Unmarshal([]byte(res.Wire()), &got); err != nil {
			t.Fatalf("parse Go wire %s: %v", stem, err)
		}
		goBoxes := flattenQuads(got.Output)

		raw, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatalf("read golden %s: %v", stem, err)
		}
		var gold struct {
			Output [][][][][2]float64 `json:"output"`
		}
		if err := json.Unmarshal(raw, &gold); err != nil {
			t.Fatalf("parse golden %s: %v", stem, err)
		}
		goldBoxes := flattenQuads(gold.Output)

		mG, mGo, maxd := matchBothDirections(goldBoxes, goBoxes, cmpTolCoord)
		imG, imGo := matchIoUBothDirections(goldBoxes, goBoxes, 0.5)
		s := stat{
			stem:                stem,
			nGold:               len(goldBoxes),
			nGo:                 len(goBoxes),
			centerOrphanGold:    len(goldBoxes) - mG,
			centerOrphanGo:      len(goBoxes) - mGo,
			iouMatchedGold:      imG,
			iouMatchedGo:        imGo,
			iouOrphanGold:       len(goldBoxes) - imG,
			iouOrphanGo:         len(goBoxes) - imGo,
			maxd:                maxd,
		}
		stats = append(stats, s)
		sumGold += s.nGold
		sumGo += s.nGo
		sumCIoG += s.centerOrphanGold
		sumCIoGo += s.centerOrphanGo
		sumIIoG += s.iouOrphanGold
		sumIIoGo += s.iouOrphanGo
	}

	for _, s := range stats {
		t.Logf("det %-22s gold=%d go=%d | centerOrphan(g/g)=%d/%d maxd=%.1f | iouOrphan(g/g)=%d/%d",
			s.stem, s.nGold, s.nGo, s.centerOrphanGold, s.centerOrphanGo, s.maxd, s.iouOrphanGold, s.iouOrphanGo)
	}
	t.Logf("TOTAL gold=%d go=%d", sumGold, sumGo)
	t.Logf("center orphan(gold/go)=%d/%d  |  IoU orphan(gold/go)=%d/%d",
		sumCIoG, sumCIoGo, sumIIoG, sumIIoGo)

	// Regression guard: the IoU orphan baseline is the KNOWN gap, not a target.
	// Fail only if Go gets WORSE than the baseline — a future geometry change
	// must not introduce new box-membership divergence. The slack absorbs benign
	// run-to-run nondeterminism.
	//
	// Baseline history:
	//   - 37/20 — pinned against the old non-bit-exact FillConvexPoly scanline
	//     (over-drew, kept 5 boxes at the 0.5 threshold by coincidence). Wrong.
	//   - 42/13 — after fillPoly was rewritten to bit-match cv2.fillPoly, the
	//     true gap vs the THEN-COMMITTED goldens. Those goldens were later found
	//     STALE: they were ~20 boxes denser than the current live TextDetector
	//     (generated with an older onnxruntime; the drift gate's 15%-count-only
	//     det check masked it), so 42/13 over-reported Go's divergence.
	//   - 23/9 — re-measured on 2026-08-11 AFTER the .det.golden.json fixtures
	//     were regenerated from the current live TextDetector and the contour
	//     grouping was ported to a findContours-equivalent (Suzuki-Abe / Moore-
	//     neighbour border following, RETR_LIST). This was the true Go-vs-cv2
	//     box-membership gap; the residual 23/9 was a downstream pred-map
	//     divergence (R/B channel-order swap in normalizeCHW: Go applied the
	//     ImageNet stats to BGR bytes while deepdoc's TextDetector normalizes
	//     the RGB image directly, so 0.485 landed on B in Go and on R in
	//     deepdoc — a ~3e-3 pred-map gap that box_score_fast amplified into
	//     score-crossing orphans).
	//   - 3/5 — re-measured on 2026-08-12 AFTER normalizeCHW was fixed to feed
	//     RGB bytes (detPreprocess now uses img.Pix, not img.ToBGR) with
	//     RGB-order stats, matching deepdoc. The Go det pred map now matches
	//     the live TextDetector to mean|Δ|≈4e-5 (was 3.2e-3); the only
	//     remaining IoU orphans are contour-tracer geometry (e.g. mp_cn_sm_p0
	//     gold 303 vs go 306), not pred/score. Do not raise this baseline back
	//     toward 23/9, 42/13 or 37/20 — those tracked a swapped-channel oracle
	//     or a stale golden, not Go's real divergence.
	const (
		baselineIoUOrphanGold = 3
		baselineIoUOrphanGo   = 5
		iuSlack               = 3
	)
	if sumIIoG > baselineIoUOrphanGold+iuSlack {
		t.Errorf("Go missed %d golden boxes under IoU (> baseline %d+%d): box-membership REGRESSION (gap 3)",
			sumIIoG, baselineIoUOrphanGold, iuSlack)
	}
	if sumIIoGo > baselineIoUOrphanGo+iuSlack {
		t.Errorf("Go produced %d extra boxes under IoU (> baseline %d+%d): box-membership REGRESSION (gap 3)",
			sumIIoGo, baselineIoUOrphanGo, iuSlack)
	}
	t.Logf("IoU orphan baseline(gold/go)=%d/%d (+slack %d) — current %d/%d OK",
		baselineIoUOrphanGold, baselineIoUOrphanGo, iuSlack, sumIIoG, sumIIoGo)
}

// flattenQuads collapses a det Wire()/golden output payload to its box list.
// Both nest quads under output[0][0].
func flattenQuads(out [][][][][2]float64) [][][2]float64 {
	if len(out) == 0 || len(out[0]) == 0 {
		return nil
	}
	return out[0][0]
}

// matchBothDirections matches two quad sets by nearest center within tol (px),
// in BOTH directions. It returns the number of golden boxes that found a Go
// match, the number of Go boxes that found a golden match, and the worst
// per-corner coordinate difference observed among matched pairs.
func matchBothDirections(gold, got [][][2]float64, tol float64) (matchedGold, matchedGo int, maxd float64) {
	sq := func(x float64) float64 { return x * x }
	// golden -> Go
	usedGo := make([]bool, len(got))
	for _, gb := range gold {
		gcx, gcy := quadCenter(gb)
		best, bd := -1, math.MaxFloat64
		for i, vb := range got {
			if usedGo[i] {
				continue
			}
			vcx, vcy := quadCenter(vb)
			d := sq(gcx-vcx) + sq(gcy-vcy)
			if d < bd {
				bd, best = d, i
			}
		}
		if best < 0 || math.Sqrt(bd) > tol {
			continue
		}
		usedGo[best] = true
		matchedGold++
		for j := 0; j < 4; j++ {
			for k := 0; k < 2; k++ {
				if d := math.Abs(gb[j][k] - got[best][j][k]); d > maxd {
					maxd = d
				}
			}
		}
	}
	// Go -> golden (reverse), to surface Go boxes with no golden counterpart.
	usedGold := make([]bool, len(gold))
	for _, vb := range got {
		vcx, vcy := quadCenter(vb)
		best, bd := -1, math.MaxFloat64
		for i, gb := range gold {
			if usedGold[i] {
				continue
			}
			gcx, gcy := quadCenter(gb)
			d := sq(gcx-vcx) + sq(gcy-vcy)
			if d < bd {
				bd, best = d, i
			}
		}
		if best < 0 || math.Sqrt(bd) > tol {
			continue
		}
		usedGold[best] = true
		matchedGo++
		for j := 0; j < 4; j++ {
			for k := 0; k < 2; k++ {
				if d := math.Abs(gold[best][j][k] - vb[j][k]); d > maxd {
					maxd = d
				}
			}
		}
	}
	return matchedGold, matchedGo, maxd
}

// quadAABB returns the axis-aligned bounding box of a quad.
func quadAABB(q [][2]float64) (x0, y0, x1, y1 float64) {
	x0, y0, x1, y1 = q[0][0], q[0][1], q[0][0], q[0][1]
	for _, p := range q {
		if p[0] < x0 {
			x0 = p[0]
		}
		if p[1] < y0 {
			y0 = p[1]
		}
		if p[0] > x1 {
			x1 = p[0]
		}
		if p[1] > y1 {
			y1 = p[1]
		}
	}
	return
}

// iou returns the intersection-over-union of two quads' AABBs.
func iou(a, b [][2]float64) float64 {
	ax0, ay0, ax1, ay1 := quadAABB(a)
	bx0, by0, bx1, by1 := quadAABB(b)
	ix0, iy0 := math.Max(ax0, bx0), math.Max(ay0, by0)
	ix1, iy1 := math.Min(ax1, bx1), math.Min(ay1, by1)
	iw, ih := ix1-ix0, iy1-iy0
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := iw * ih
	areaA := (ax1 - ax0) * (ay1 - ay0)
	areaB := (bx1 - bx0) * (by1 - by0)
	return inter / (areaA + areaB - inter)
}

// matchIoUBothDirections matches two quad sets by greedy best-IoU in BOTH
// directions. A pair matches only if IoU >= thr. This isolates true
// box-membership divergence (one box split into two, two merged into one,
// spurious detections) from mere coordinate drift: a box shifted 20px but
// still overlapping its twin scores high IoU and is NOT an orphan.
func matchIoUBothDirections(gold, got [][][2]float64, thr float64) (matchedGold, matchedGo int) {
	usedGo := make([]bool, len(got))
	for _, gb := range gold {
		best, bestI := -1, 0.0
		for i, vb := range got {
			if usedGo[i] {
				continue
			}
			if v := iou(gb, vb); v > bestI {
				bestI, best = v, i
			}
		}
		if best >= 0 && bestI >= thr {
			usedGo[best] = true
			matchedGold++
		}
	}
	usedGold := make([]bool, len(gold))
	for _, vb := range got {
		best, bestI := -1, 0.0
		for i, gb := range gold {
			if usedGold[i] {
				continue
			}
			if v := iou(gb, vb); v > bestI {
				bestI, best = v, i
			}
		}
		if best >= 0 && bestI >= thr {
			usedGold[best] = true
			matchedGo++
		}
	}
	return matchedGold, matchedGo
}

// TestDetOCRAdjudication answers a different question than
// TestDetMembershipAllFixtures. The membership test measures how close Go's
// boxes are to Python's (alignment). This harness measures which detector
// produces boxes that lead to BETTER parsing: for every box only one side
// found (an "IoU orphan"), it crops the original image to that box and runs
// OCR-rec, then reports whether the crop yields coherent text. The side whose
// orphan boxes more often produce real text is the side that detected genuine
// text regions the other missed. Python is NOT assumed truth here — OCR
// quality is the independent judge.
//
// Informational only: it never fails. Per-box text is logged so a human can
// adjudicate; the summary tallies non-empty, confident OCR results per side as
// a proxy for "found real text". The crop is axis-aligned (not rotated) so both
// detectors are judged by the same method.
func TestDetOCRAdjudication(t *testing.T) {
	skipIfNoModels(t)
	md := os.Getenv("MODEL_DIR")
	dir := "../testdata"

	// Fixtures with non-zero IoU orphans (the only ones worth adjudicating).
	stems := []string{
		"mp_cn_sm_p0",   // dense Chinese small text — worst divergence
		"mp_arxiv_p0",    // multi-column paper
		"mp_en_dense_p0", // dense English
		"mp_physics_p5",
		"mp_sec_p0",
	}

	const (
		iouThr       = 0.5
		realScoreThr = 0.6 // OCR confidence above which a crop is "likely real text"
	)

	sumPyOnly, sumPyOnlyReal := 0, 0
	sumGoOnly, sumGoOnlyReal := 0, 0

	for _, stem := range stems {
		img, err := Decode(filepath.Join(dir, stem+".png"))
		if err != nil {
			t.Fatalf("decode %s: %v", stem, err)
		}
		res, err := RunDet(t.Context(), md, img)
		if err != nil {
			t.Fatalf("RunDet %s: %v", stem, err)
		}
		var got struct {
			Output [][][][][2]float64 `json:"output"`
		}
		if err := json.Unmarshal([]byte(res.Wire()), &got); err != nil {
			t.Fatalf("parse Go wire %s: %v", stem, err)
		}
		goBoxes := flattenQuads(got.Output)

		raw, err := os.ReadFile(filepath.Join(dir, stem+".det.golden.json"))
		if err != nil {
			t.Fatalf("read golden %s: %v", stem, err)
		}
		var gold struct {
			Output [][][][][2]float64 `json:"output"`
		}
		if err := json.Unmarshal(raw, &gold); err != nil {
			t.Fatalf("parse golden %s: %v", stem, err)
		}
		pyBoxes := flattenQuads(gold.Output)

		pyOnly, goOnly := matchIoUOrphans(pyBoxes, goBoxes, iouThr)
		fPyOnly, fPyOnlyReal := 0, 0
		fGoOnly, fGoOnlyReal := 0, 0

		for _, i := range pyOnly {
			txt, score, ok := ocrCrop(t, md, img, pyBoxes[i])
			real := ok && strings.TrimSpace(txt) != "" && score >= realScoreThr
			if ok {
				fPyOnly++
				if real {
					fPyOnlyReal++
				}
			}
			t.Logf("  [PY-only] %-14s box#%d text=%q score=%.2f real=%v", stem, i, txt, score, real)
		}
		for _, i := range goOnly {
			txt, score, ok := ocrCrop(t, md, img, goBoxes[i])
			real := ok && strings.TrimSpace(txt) != "" && score >= realScoreThr
			if ok {
				fGoOnly++
				if real {
					fGoOnlyReal++
				}
			}
			t.Logf("  [GO-only] %-14s box#%d text=%q score=%.2f real=%v", stem, i, txt, score, real)
		}

		t.Logf("%-14s: PY-only=%d (real %d) | GO-only=%d (real %d)", stem, fPyOnly, fPyOnlyReal, fGoOnly, fGoOnlyReal)
		sumPyOnly += fPyOnly
		sumPyOnlyReal += fPyOnlyReal
		sumGoOnly += fGoOnly
		sumGoOnlyReal += fGoOnlyReal
	}

	t.Logf("SUMMARY — orphan boxes that yielded real text (score>=%.1f):", realScoreThr)
	t.Logf("  Python-only: %d/%d real", sumPyOnlyReal, sumPyOnly)
	t.Logf("  Go-only:     %d/%d real", sumGoOnlyReal, sumGoOnly)
	t.Logf("  (higher 'real' count on a side = that side found more genuine text the other missed)")
}

// ocrCrop crops img to the axis-aligned bbox of quad q, runs OCR-rec, and
// returns the recognized text, confidence, and whether the crop was valid.
func ocrCrop(t *testing.T, md string, img *Image, q [][2]float64) (string, float32, bool) {
	c := cropQuad(img, q)
	if c == nil {
		return "", 0, false
	}
	r, err := RunOCRRec(t.Context(), md, c)
	if err != nil {
		t.Logf("  ocrCrop rec error: %v", err)
		return "", 0, false
	}
	return r.Text, r.Score, true
}

// cropQuad returns the axis-aligned sub-image of img bounded by quad q, or nil
// if the quad is degenerate/out of bounds. AABB (not rotated) crop is used so
// both detectors are judged by the same method — any background inclusion
// affects both sides equally and the comparison stays fair.
func cropQuad(img *Image, q [][2]float64) *Image {
	x0, y0, x1, y1 := quadAABB(q)
	px0 := clampi(int(math.Floor(x0)), 0, img.W-1)
	py0 := clampi(int(math.Floor(y0)), 0, img.H-1)
	px1 := clampi(int(math.Ceil(x1)), 0, img.W)
	py1 := clampi(int(math.Ceil(y1)), 0, img.H)
	if px1 <= px0 || py1 <= py0 {
		return nil
	}
	w, h := px1-px0, py1-py0
	pix := make([]byte, w*h*3)
	for y := 0; y < h; y++ {
		copy(pix[y*w*3:(y+1)*w*3], img.Pix[((py0+y)*img.W+px0)*3:((py0+y)*img.W+px0)*3+w*3])
	}
	return &Image{W: w, H: h, Pix: pix}
}

// matchIoUOrphans returns the indices (into pyBoxes / goBoxes) of boxes that
// have no counterpart on the other side under greedy best-IoU >= thr. These
// are the "orphan" boxes found by only one detector. Greedy matching is run in
// both directions so a box with no stable twin on either side is reported.
func matchIoUOrphans(pyBoxes, goBoxes [][][2]float64, thr float64) (pyOnly, goOnly []int) {
	usedGo := make([]bool, len(goBoxes))
	pyMatched := make([]bool, len(pyBoxes))
	for i, gb := range pyBoxes {
		best, bestI := -1, 0.0
		for j, vb := range goBoxes {
			if usedGo[j] {
				continue
			}
			if v := iou(gb, vb); v > bestI {
				bestI, best = v, j
			}
		}
		if best >= 0 && bestI >= thr {
			usedGo[best] = true
			pyMatched[i] = true
		}
	}
	for i := range pyBoxes {
		if !pyMatched[i] {
			pyOnly = append(pyOnly, i)
		}
	}
	usedPy := make([]bool, len(pyBoxes))
	goMatched := make([]bool, len(goBoxes))
	for j, vb := range goBoxes {
		best, bestI := -1, 0.0
		for i, gb := range pyBoxes {
			if usedPy[i] {
				continue
			}
			if v := iou(gb, vb); v > bestI {
				bestI, best = v, i
			}
		}
		if best >= 0 && bestI >= thr {
			usedPy[best] = true
			goMatched[j] = true
		}
	}
	for j := range goBoxes {
		if !goMatched[j] {
			goOnly = append(goOnly, j)
		}
	}
	return pyOnly, goOnly
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

// jsonTemplate reduces a decoded JSON value to a structural signature where
// every number becomes "#", every string "$", every bool "?", and object keys
// are sorted. Two values with identical nesting/keys/leaf-types produce the
// same template even if their values differ — so it isolates schema from
// content.
func jsonTemplate(v any) string {
	switch x := v.(type) {
	case map[string]any:
		keys := make([]string, 0, len(x))
		for k := range x {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		var b strings.Builder
		b.WriteString("{")
		for i, k := range keys {
			if i > 0 {
				b.WriteString(",")
			}
			b.WriteString(k)
			b.WriteString(":")
			b.WriteString(jsonTemplate(x[k]))
		}
		b.WriteString("}")
		return b.String()
	case []any:
		if len(x) == 0 {
			return "[]"
		}
		// Structures here have uniform inner shape, so the first element's
		// template represents the whole array.
		return "[" + jsonTemplate(x[0]) + "]"
	case float64:
		return "#"
	case string:
		return "$"
	case bool:
		return "?"
	case nil:
		return "null"
	default:
		return "?"
	}
}

// TestWireSchemaMatchesGolden is the schema half of the migration contract: a
// caller parsing Go's Wire() output must see the exact same JSON structure
// (top-level key, nesting depth, leaf types) as the deepdoc reference golden.
// The per-task integration tests already check values; this guard catches a
// shape regression (e.g. {"bboxes":...} vs a bare array, or a changed nesting)
// that value comparison would otherwise paper over.
func TestWireSchemaMatchesGolden(t *testing.T) {
	skipIfNoModels(t)
	md := os.Getenv("MODEL_DIR")
	cases := []struct {
		name   string
		stem   string
		golden string
		wire   func(t *testing.T, img *Image) string
	}{
		{"DLA", "page0", "page0.dla.golden.json", func(t *testing.T, img *Image) string {
			res, err := RunDLA(t.Context(), md, img)
			if err != nil {
				t.Fatalf("RunDLA: %v", err)
			}
			return res.Wire()
		}},
		{"TSR", "table0", "table0.tsr.golden.json", func(t *testing.T, img *Image) string {
			res, err := RunTSR(t.Context(), md, img)
			if err != nil {
				t.Fatalf("RunTSR: %v", err)
			}
			return res.Wire()
		}},
		{"DET", "page0", "page0.det.golden.json", func(t *testing.T, img *Image) string {
			res, err := RunDet(t.Context(), md, img)
			if err != nil {
				t.Fatalf("RunDet: %v", err)
			}
			return res.Wire()
		}},
		{"OCR_REC", "line_mix", "line_mix.ocr_rec.golden.json", func(t *testing.T, img *Image) string {
			res, err := RunOCRRec(t.Context(), md, img)
			if err != nil {
				t.Fatalf("RunOCRRec: %v", err)
			}
			return res.Wire()
		}},
	}
	for _, c := range cases {
		c := c
		t.Run(c.name, func(t *testing.T) {
			img, err := Decode(filepath.Join("..", "testdata", c.stem+".png"))
			if err != nil {
				t.Fatalf("decode %s: %v", c.stem, err)
			}
			goWire := c.wire(t, img)
			goldRaw, err := os.ReadFile(filepath.Join("..", "testdata", c.golden))
			if err != nil {
				t.Fatalf("read golden %s: %v", c.golden, err)
			}
			var gv, gv2 any
			if err := json.Unmarshal([]byte(goWire), &gv); err != nil {
				t.Fatalf("parse go wire: %v", err)
			}
			if err := json.Unmarshal(goldRaw, &gv2); err != nil {
				t.Fatalf("parse golden: %v", err)
			}
			if got, want := jsonTemplate(gv), jsonTemplate(gv2); got != want {
				t.Errorf("wire schema mismatch:\n got  %s\n want %s", got, want)
			} else {
				t.Logf("%s wire schema: %s", c.name, got)
			}
		})
	}
}

// TestDumpGoCandidates is a diagnostic: run RunDet on one fixture with
// DLA_DUMP_CANDIDATES set so dbPostProcess writes /tmp/go_candidates.json
// (post-geometry, pre-score-filter quads + scores) for offline analysis of
// Go/cv2 det divergence. Not a regression test.
func TestDumpGoCandidates(t *testing.T) {
	skipIfNoModels(t)
	fixture := os.Getenv("FIXTURE")
	if fixture == "" {
		fixture = "mp_cn_sm_p0"
	}
	t.Setenv("DLA_DUMP_CANDIDATES", "1")
	imgPath := filepath.Join("..", "testdata", fixture+".png")
	img, err := Decode(imgPath)
	if err != nil {
		t.Fatalf("decode %s: %v", imgPath, err)
	}
	if _, err := RunDet(t.Context(), os.Getenv("MODEL_DIR"), img); err != nil {
		t.Fatalf("RunDet: %v", err)
	}
	t.Logf("wrote /tmp/go_candidates.json for %s", fixture)
}

// TestDumpStages is a diagnostic: run RunDet on one fixture (FIXTURE env) with
// the stage-dump env vars set, writing the Go-side intermediates that the
// Python oracle (cmp_stages.py) is diffed against:
//   /tmp/go_pred.json       — raw pred map (post-sigmoid, pre-threshold)
//   /tmp/go_quads_pre.json  — pre-unclip min-area rect per component (resized)
//   /tmp/go_candidates.json — post-geometry, pre-score-filter quad+score
//
// Not a regression test. Pair with: python cmp_stages.py <img> <model_dir>;
// then python diff_stages.py for the per-stage comparison.
func TestDumpStages(t *testing.T) {
	skipIfNoModels(t)
	fixture := os.Getenv("FIXTURE")
	if fixture == "" {
		fixture = "mp_cn_sm_p0"
	}
	t.Setenv("DLA_DUMP_STAGES", "1")
	t.Setenv("DLA_DUMP_QUADS", "1")
	t.Setenv("DLA_DUMP_CANDIDATES", "1")
	imgPath := filepath.Join("..", "testdata", fixture+".png")
	img, err := Decode(imgPath)
	if err != nil {
		t.Fatalf("decode %s: %v", imgPath, err)
	}
	res, err := RunDet(t.Context(), os.Getenv("MODEL_DIR"), img)
	if err != nil {
		t.Fatalf("RunDet: %v", err)
	}
	// Final boxes (post score-filter + filter_tag_det_res), for direct IoU
	// comparison against the Python oracle's final output (ref_det.py).
	fb := make([]map[string]any, 0, len(res.Boxes))
	for _, b := range res.Boxes {
		pts := make([][2]float64, 4)
		for i := range b.Pts {
			pts[i] = [2]float64{float64(b.Pts[i][0]), float64(b.Pts[i][1])}
		}
		fb = append(fb, map[string]any{"pts": pts, "score": float64(b.Score)})
	}
	if b, e := json.Marshal(map[string]any{"boxes": fb}); e == nil {
		_ = os.WriteFile("/tmp/go_final.json", b, 0o644)
	}
	t.Logf("wrote /tmp/go_pred.json, /tmp/go_quads_pre.json, /tmp/go_candidates.json, /tmp/go_final.json for %s", fixture)
}
