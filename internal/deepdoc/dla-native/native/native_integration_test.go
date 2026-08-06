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
	cmpTolCoord = 2.0  // px tolerance on box coordinates
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

func TestDLAIntegration(t *testing.T) {
	skipIfNoModels(t)
	img, err := Decode(filepath.Join("..", "testdata", "page0.jpg"))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	res, err := RunDLA(os.Getenv("MODEL_DIR"), img)
	if err != nil {
		t.Fatalf("RunDLA: %v", err)
	}
	var got struct {
		Boxes [][]float64 `json:"bboxes"`
	}
	if err := json.Unmarshal([]byte(res.Wire()), &got); err != nil {
		t.Fatalf("parse Go wire: %v", err)
	}
	gold := loadGoldenBoxes(t, filepath.Join("..", "testdata", "page0.dla.golden.json"))
	compareBoxes(t, gold, got.Boxes)
}

func TestTSRIntegration(t *testing.T) {
	skipIfNoModels(t)
	img, err := Decode(filepath.Join("..", "testdata", "table0.jpg"))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	res, err := RunTSR(os.Getenv("MODEL_DIR"), img)
	if err != nil {
		t.Fatalf("RunTSR: %v", err)
	}
	var got struct {
		Boxes [][]float64 `json:"bboxes"`
	}
	if err := json.Unmarshal([]byte(res.Wire()), &got); err != nil {
		t.Fatalf("parse Go wire: %v", err)
	}
	gold := loadGoldenBoxes(t, filepath.Join("..", "testdata", "table0.tsr.golden.json"))
	compareBoxes(t, gold, got.Boxes)
}

func TestOCRRecIntegration(t *testing.T) {
	skipIfNoModels(t)
	img, err := Decode(filepath.Join("..", "testdata", "line0.jpg"))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	res, err := RunOCRRec(os.Getenv("MODEL_DIR"), img)
	if err != nil {
		t.Fatalf("RunOCRRec: %v", err)
	}
	var got struct {
		Output [][][][]any `json:"output"`
	}
	if err := json.Unmarshal([]byte(res.Wire()), &got); err != nil {
		t.Fatalf("parse Go wire: %v", err)
	}
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "line0.ocr_rec.golden.json"))
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
}

// The three recognizers below share the fixed-shape model-session pool. Each
// test runs the recognizer twice on the same crop and asserts byte-identical
// wire output, proving getModelSession hands back a clean session after release
// (no stale input tensor, no cross-call contamination) and that pooling is a
// no-behavior-change over the old per-call NewSession path.

func TestDLASessionReuse(t *testing.T) {
	skipIfNoModels(t)
	img, err := Decode(filepath.Join("..", "testdata", "page0.jpg"))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	r1, err := RunDLA(os.Getenv("MODEL_DIR"), img)
	if err != nil {
		t.Fatalf("RunDLA #1: %v", err)
	}
	r2, err := RunDLA(os.Getenv("MODEL_DIR"), img)
	if err != nil {
		t.Fatalf("RunDLA #2: %v", err)
	}
	if r1.Wire() != r2.Wire() {
		t.Fatalf("DLA output changed across pooled runs (session reuse not stable)")
	}
}

func TestTSRSessionReuse(t *testing.T) {
	skipIfNoModels(t)
	img, err := Decode(filepath.Join("..", "testdata", "table0.jpg"))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	r1, err := RunTSR(os.Getenv("MODEL_DIR"), img)
	if err != nil {
		t.Fatalf("RunTSR #1: %v", err)
	}
	r2, err := RunTSR(os.Getenv("MODEL_DIR"), img)
	if err != nil {
		t.Fatalf("RunTSR #2: %v", err)
	}
	if r1.Wire() != r2.Wire() {
		t.Fatalf("TSR output changed across pooled runs (session reuse not stable)")
	}
}

func TestOCRRecSessionReuse(t *testing.T) {
	skipIfNoModels(t)
	img, err := Decode(filepath.Join("..", "testdata", "line0.jpg"))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	r1, err := RunOCRRec(os.Getenv("MODEL_DIR"), img)
	if err != nil {
		t.Fatalf("RunOCRRec #1: %v", err)
	}
	r2, err := RunOCRRec(os.Getenv("MODEL_DIR"), img)
	if err != nil {
		t.Fatalf("RunOCRRec #2: %v", err)
	}
	if r1.Wire() != r2.Wire() {
		t.Fatalf("OCR-rec output changed across pooled runs (session reuse not stable)")
	}
}

func TestDetIntegration(t *testing.T) {
	skipIfNoModels(t)
	imgPath := filepath.Join("..", "testdata", "page0.jpg")
	img, err := Decode(imgPath)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	// Set Path so the gocv build re-decodes via cv2 (gocv.IMRead), matching
	// deepdoc's cv2/PIL decode exactly. Without it the gocv preprocessing blob
	// uses Go's jpeg decoder and the 3px hard floor (HANDOFF §4.4) is not
	// actually exercised. The pure-Go build ignores Path, so this is safe for
	// both build tags.
	img.Path = imgPath
	res, err := RunDet(os.Getenv("MODEL_DIR"), img)
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

	// Both builds reach the 3px hard floor (HANDOFF §4.4, box#8): gocv via the
	// cv2 re-decode + convexify fix, pure-Go via the shared geometry. Tolerance
	// is set just above that floor so a regression bumps it over the line.
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

func quadCenter(q [][2]float64) (float64, float64) {
	var sx, sy float64
	for _, p := range q {
		sx += p[0]
		sy += p[1]
	}
	return sx / float64(len(q)), sy / float64(len(q))
}
