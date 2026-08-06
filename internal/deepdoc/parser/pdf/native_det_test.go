//go:build native_det && integration

package pdf

// Integration test for the native DB text detector wired into the pdf parser
// package (C1). It runs the in-process dla-native detector through
// nativeOCRDetect on a real page image and asserts it produces valid
// OCRBox quads — proving the det→ingestion integration seam works end to end
// on CPU via ONNX Runtime, with no remote DeepDoc service.
//
// Run under the native_det + integration tags with ORT_LIB and MODEL_DIR set:
//   ORT_LIB=... MODEL_DIR=... go test -tags "native_det integration" \
//     ./internal/deepdoc/parser/pdf/ -run TestNativeOCRDetect

import (
	"encoding/json"
	"image/jpeg"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// detFixture is one multi-page robustness case for A2. goldenCount, when
// non-negative, is the Python-oracle (ref_det.py) box count committed as
// <stem>.det.golden.json; the test asserts the Go port tracks it within a
// relative tolerance instead of being overfit to page0.jpg alone.
type detFixture struct {
	stem        string
	goldenCount int
	blank       bool
}

// TestNativeOCRDetectMultiPage exercises the native DB detector across
// heterogeneous real pages (English title/body, English textbook, Chinese
// QA, Chinese manual) plus a synthetic blank page. It proves the port is not
// overfit to the single page0.jpg fixture: every page yields finite,
// in-bounds quads, the blank page yields zero boxes, and per-page box counts
// track the Python oracle within tolerance. See HANDOFF.md §8 A2.
//
// Requires ORT_LIB and MODEL_DIR (skips otherwise). Fixtures + goldens live
// in the sibling dla-native module's testdata (mp_*.jpg / blank.jpg).
func TestNativeOCRDetectMultiPage(t *testing.T) {
	ortLib := os.Getenv("ORT_LIB")
	modelDir := os.Getenv("MODEL_DIR")
	if ortLib == "" || modelDir == "" {
		t.Skip("set ORT_LIB and MODEL_DIR to run the native_det integration test")
	}

	_, thisFile, _, _ := runtime.Caller(0)
	testdata := filepath.Join(filepath.Dir(thisFile), "..", "..", "dla-native", "testdata")

	fixtures := []detFixture{
		{"page0", 15, false},       // baseline single-page fixture
		{"mp_arxiv_p0", 94, false}, // English title/abstract
		{"mp_arxiv_p1", 98, false}, // English two-column body
		{"mp_physics_p5", 20, false}, // English textbook
		{"mp_cn_qa_p0", 83, false},  // Chinese QA doc
		{"mp_cn_sm_p0", 312, false}, // Chinese manual, large page
		{"mp_jp_p0", 55, false},     // Japanese paper (multi-script coverage)
		{"mp_zhtw_p0", 26, false},   // Traditional Chinese service doc
		{"mp_cn_std_p0", 14, false}, // Chinese technical standard (dense CN)
		{"mp_sec_p0", 109, false},   // Chinese securities report (table-heavy)
		{"mp_en_dense_p0", 96, false}, // English dense two-column paper
		{"blank", 0, true},          // synthetic blank page
	}

	for _, fx := range fixtures {
		fx := fx
		t.Run(fx.stem, func(t *testing.T) {
			imgPath := filepath.Join(testdata, fx.stem+".jpg")
			f, err := os.Open(imgPath)
			if err != nil {
				t.Fatalf("open %s: %v", imgPath, err)
			}
			defer f.Close()
			img, err := jpeg.Decode(f)
			if err != nil {
				t.Fatalf("decode %s: %v", imgPath, err)
			}

			boxes, err := nativeOCRDetect(img)
			if err != nil {
				t.Fatalf("nativeOCRDetect(%s): %v", fx.stem, err)
			}

			b := img.Bounds()
			for _, bx := range boxes {
				for _, v := range []float64{bx.X0, bx.Y0, bx.X1, bx.Y1, bx.X2, bx.Y2, bx.X3, bx.Y3} {
					if math.IsNaN(v) || math.IsInf(v, 0) {
						t.Fatalf("non-finite coord on %s: %v", fx.stem, bx)
					}
				}
				for _, c := range []struct{ x, y float64 }{
					{bx.X0, bx.Y0}, {bx.X1, bx.Y1}, {bx.X2, bx.Y2}, {bx.X3, bx.Y3},
				} {
					if c.x < -1 || c.y < -1 || c.x > float64(b.Dx())+1 || c.y > float64(b.Dy())+1 {
						t.Fatalf("quad out of bounds %dx%d on %s: %v", b.Dx(), b.Dy(), fx.stem, bx)
					}
				}
			}

			// Blank page must produce exactly zero boxes.
			if fx.blank {
				if len(boxes) != 0 {
					t.Fatalf("blank page produced %d boxes, want 0", len(boxes))
				}
				t.Logf("%s (blank): 0 boxes as expected", fx.stem)
				return
			}

			if len(boxes) < 3 {
				t.Fatalf("%s produced %d boxes, want >= 3", fx.stem, len(boxes))
			}

			// Compare against the Python oracle within a relative tolerance.
			goldenPath := filepath.Join(testdata, fx.stem+".det.golden.json")
			if gc, ok := loadGoldenCount(t, goldenPath); ok {
				tol := 2
				if d := int(0.15 * float64(gc)); d > tol {
					tol = d
				}
				diff := len(boxes) - gc
				if diff < 0 {
					diff = -diff
				}
				if diff > tol {
					t.Fatalf("%s box count %d deviates from oracle %d by %d (> tol %d)",
						fx.stem, len(boxes), gc, diff, tol)
				}
				t.Logf("%s: go=%d oracle=%d (tol %d) OK", fx.stem, len(boxes), gc, tol)
			}
		})
	}
}

// loadGoldenCount reads a ref_det.py-style golden ({ "output": [[quads]] })
// and returns the number of quads. ok is false when the golden file is absent
// (the test then skips the oracle comparison).
func loadGoldenCount(t *testing.T, path string) (int, bool) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, false
	}
	var doc struct {
		Output [][][][][2]float64 `json:"output"`
	}
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatalf("parse golden %s: %v", path, err)
	}
	if len(doc.Output) == 0 || len(doc.Output[0]) == 0 {
		return 0, true
	}
	return len(doc.Output[0][0]), true
}

// TestNativeOCRDetectBlankEdges verifies blank-page detection across
// boundary cases so a real PDF composed of blank/near-blank pages never
// produces spurious boxes or panics. The Python oracle (ref_det.py) returns
// zero boxes for every case below; the Go port must match. Cases:
//   blank_black  — fully black page (no bright text regions)
//   blank_tiny   — 8x8 page (exercises round32 min-size clamp)
//   blank_large  — 4000x6000 page (exercises limit_side_len downscale)
//   blank_faint  — white with a single faint pixel (sub-threshold noise)
//   blank_border — white with a 1px gray border (thin contour, filtered)
//
// Requires ORT_LIB and MODEL_DIR (skips otherwise).
func TestNativeOCRDetectBlankEdges(t *testing.T) {
	ortLib := os.Getenv("ORT_LIB")
	modelDir := os.Getenv("MODEL_DIR")
	if ortLib == "" || modelDir == "" {
		t.Skip("set ORT_LIB and MODEL_DIR to run the native_det integration test")
	}

	_, thisFile, _, _ := runtime.Caller(0)
	testdata := filepath.Join(filepath.Dir(thisFile), "..", "..", "dla-native", "testdata")

	edges := []string{
		"blank_black", "blank_tiny", "blank_large", "blank_faint", "blank_border",
	}
	for _, stem := range edges {
		stem := stem
		t.Run(stem, func(t *testing.T) {
			imgPath := filepath.Join(testdata, stem+".jpg")
			f, err := os.Open(imgPath)
			if err != nil {
				t.Fatalf("open %s: %v", imgPath, err)
			}
			defer f.Close()
			img, err := jpeg.Decode(f)
			if err != nil {
				t.Fatalf("decode %s: %v", imgPath, err)
			}

			boxes, err := nativeOCRDetect(img)
			if err != nil {
				t.Fatalf("nativeOCRDetect(%s): %v", stem, err)
			}
			// Every blank-edge case must produce exactly zero boxes.
			if len(boxes) != 0 {
				t.Fatalf("%s produced %d boxes on a blank/near-blank page, want 0", stem, len(boxes))
			}
			// And the oracle agrees (guards against a future tolerance slip).
			if gc, ok := loadGoldenCount(t, filepath.Join(testdata, stem+".det.golden.json")); ok && gc != 0 {
				t.Fatalf("%s oracle count is %d, test assumed blank", stem, gc)
			}
			t.Logf("%s: 0 boxes on blank/near-blank page (ok)", stem)
		})
	}
}

// TestNativeOCRDetectDegenerate exercises synthesized degenerate inputs that
// stress rare paths the real-page fixtures never hit: a single glyph, a single
// text line, a rotated line, salt&pepper noise, a solid fill, a gradient, and
// near-threshold low-contrast text. Each case is oracle-backed by ref_det.py;
// blank inputs must yield zero boxes and the rest must track the Python count
// within a tight tolerance (degenerate counts are tiny, so a 1-box slip is the
// max we accept). This guards against the detector regressing on trivial/edge
// inputs — spurious boxes on noise, or dropping a lone real line.
//
// Requires ORT_LIB and MODEL_DIR (skips otherwise).
func TestNativeOCRDetectDegenerate(t *testing.T) {
	ortLib := os.Getenv("ORT_LIB")
	modelDir := os.Getenv("MODEL_DIR")
	if ortLib == "" || modelDir == "" {
		t.Skip("set ORT_LIB and MODEL_DIR to run the native_det integration test")
	}

	_, thisFile, _, _ := runtime.Caller(0)
	testdata := filepath.Join(filepath.Dir(thisFile), "..", "..", "dla-native", "testdata")

	type deg struct {
		stem  string
		blank bool
	}
	cases := []deg{
		{"deg_single_char", false},
		{"deg_single_line", false},
		{"deg_skewed", false},
		{"deg_low_contrast", false},
		{"deg_noise", true},
		{"deg_solid", true},
		{"deg_gradient", true},
	}
	for _, c := range cases {
		c := c
		t.Run(c.stem, func(t *testing.T) {
			imgPath := filepath.Join(testdata, c.stem+".jpg")
			f, err := os.Open(imgPath)
			if err != nil {
				t.Fatalf("open %s: %v", imgPath, err)
			}
			defer f.Close()
			img, err := jpeg.Decode(f)
			if err != nil {
				t.Fatalf("decode %s: %v", imgPath, err)
			}
			boxes, err := nativeOCRDetect(img)
			if err != nil {
				t.Fatalf("nativeOCRDetect(%s): %v", c.stem, err)
			}
			b := img.Bounds()
			for _, bx := range boxes {
				for _, v := range []float64{bx.X0, bx.Y0, bx.X1, bx.Y1, bx.X2, bx.Y2, bx.X3, bx.Y3} {
					if math.IsNaN(v) || math.IsInf(v, 0) {
						t.Fatalf("non-finite coord on %s: %v", c.stem, bx)
					}
				}
				for _, p := range []struct{ x, y float64 }{
					{bx.X0, bx.Y0}, {bx.X1, bx.Y1}, {bx.X2, bx.Y2}, {bx.X3, bx.Y3},
				} {
					if p.x < -1 || p.y < -1 || p.x > float64(b.Dx())+1 || p.y > float64(b.Dy())+1 {
						t.Fatalf("quad out of bounds %dx%d on %s: %v", b.Dx(), b.Dy(), c.stem, bx)
					}
				}
			}
			gc, ok := loadGoldenCount(t, filepath.Join(testdata, c.stem+".det.golden.json"))
			if !ok {
				t.Fatalf("missing golden for %s", c.stem)
			}
			if c.blank {
				if gc != 0 {
					t.Fatalf("%s assumed blank but oracle count is %d", c.stem, gc)
				}
				if len(boxes) != 0 {
					t.Fatalf("%s produced %d boxes on degenerate-blank input, want 0", c.stem, len(boxes))
				}
				t.Logf("%s: 0 boxes on blank input (ok)", c.stem)
				return
			}
			diff := len(boxes) - gc
			if diff < 0 {
				diff = -diff
			}
			if diff > 1 {
				t.Fatalf("%s box count %d deviates from oracle %d by %d (> 1)", c.stem, len(boxes), gc, diff)
			}
			t.Logf("%s: go=%d oracle=%d (tol 1) OK", c.stem, len(boxes), gc)
		})
	}
}

// TestNativeOCRDetectSessionReuse confirms that running the detector twice on
// the same page yields identical results under the pooled-session path (A2
// optimization #2): reuse must not alter numerics or leak/recreate state
// across calls.
func TestNativeOCRDetectSessionReuse(t *testing.T) {
	ortLib := os.Getenv("ORT_LIB")
	modelDir := os.Getenv("MODEL_DIR")
	if ortLib == "" || modelDir == "" {
		t.Skip("set ORT_LIB and MODEL_DIR to run the native_det integration test")
	}

	_, thisFile, _, _ := runtime.Caller(0)
	imgPath := filepath.Join(filepath.Dir(thisFile), "..", "..", "dla-native", "testdata", "page0.jpg")
	f, err := os.Open(imgPath)
	if err != nil {
		t.Fatalf("open %s: %v", imgPath, err)
	}
	defer f.Close()
	img, err := jpeg.Decode(f)
	if err != nil {
		t.Fatalf("decode %s: %v", imgPath, err)
	}

	first, err := nativeOCRDetect(img)
	if err != nil {
		t.Fatalf("nativeOCRDetect #1: %v", err)
	}
	second, err := nativeOCRDetect(img)
	if err != nil {
		t.Fatalf("nativeOCRDetect #2: %v", err)
	}
	if len(first) != len(second) {
		t.Fatalf("box count differs across runs: %d vs %d", len(first), len(second))
	}
	for i := range first {
		a, b := first[i], second[i]
		pairs := []struct{ ax, ay, bx, by float64 }{
			{a.X0, a.Y0, b.X0, b.Y0}, {a.X1, a.Y1, b.X1, b.Y1},
			{a.X2, a.Y2, b.X2, b.Y2}, {a.X3, a.Y3, b.X3, b.Y3},
		}
		for _, c := range pairs {
			if math.Abs(c.ax-c.bx) > 1e-6 || math.Abs(c.ay-c.by) > 1e-6 {
				t.Fatalf("box %d coord differs across runs: %v vs %v", i, a, b)
			}
		}
	}
	t.Logf("session reuse: %d boxes identical across two runs", len(first))
}

func TestNativeOCRDetect(t *testing.T) {
	ortLib := os.Getenv("ORT_LIB")
	modelDir := os.Getenv("MODEL_DIR")
	if ortLib == "" || modelDir == "" {
		t.Skip("set ORT_LIB and MODEL_DIR to run the native_det integration test")
	}

	// page0.jpg lives in the sibling dla-native module's testdata. Resolve
	// relative to this test file so the path is cwd-independent.
	_, thisFile, _, _ := runtime.Caller(0)
	imgPath := filepath.Join(filepath.Dir(thisFile), "..", "..", "dla-native", "testdata", "page0.jpg")
	f, err := os.Open(imgPath)
	if err != nil {
		t.Fatalf("open %s: %v", imgPath, err)
	}
	defer f.Close()
	img, err := jpeg.Decode(f)
	if err != nil {
		t.Fatalf("decode %s: %v", imgPath, err)
	}

	boxes, err := nativeOCRDetect(img)
	if err != nil {
		t.Fatalf("nativeOCRDetect: %v", err)
	}
	if len(boxes) < 10 {
		t.Fatalf("nativeOCRDetect returned %d boxes, want >= 10", len(boxes))
	}
	b := img.Bounds()
	for _, bx := range boxes {
		for _, v := range []float64{bx.X0, bx.Y0, bx.X1, bx.Y1, bx.X2, bx.Y2, bx.X3, bx.Y3} {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("nativeOCRDetect returned non-finite coord: %v", bx)
			}
		}
		// quads must lie within the source image.
		for _, c := range []struct{ x, y float64 }{
			{bx.X0, bx.Y0}, {bx.X1, bx.Y1}, {bx.X2, bx.Y2}, {bx.X3, bx.Y3},
		} {
			if c.x < -1 || c.y < -1 || c.x > float64(b.Dx())+1 || c.y > float64(b.Dy())+1 {
				t.Fatalf("nativeOCRDetect quad out of bounds %dx%d: %v", b.Dx(), b.Dy(), bx)
			}
		}
	}
	t.Logf("nativeOCRDetect produced %d valid OCRBox quads on %s", len(boxes), imgPath)
}
