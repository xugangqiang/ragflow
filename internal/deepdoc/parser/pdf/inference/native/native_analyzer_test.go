//go:build native_det && integration

package infnative

import (
	"context"
	"image"
	"image/draw"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"

	deepdoctype "ragflow/internal/deepdoc/parser/type"
)

// TestNativeAnalyzerInProcess proves the in-process (Go) DeepDoc backend
// actually runs end-to-end through the doctype.DocAnalyzer interface: ONNX
// Runtime init + model load + DLA/TSR/OCR inference on a real page fixture,
// producing non-empty, in-bounds results. It is the caller-side analogue of the
// dla-native equivalence suite, but exercised through the DocAnalyzer seam the
// PDF parser consumes. Requires libonnxruntime + the InfiniFlow/deepdoc model
// snapshot; skipped unless both are reachable via ORT_LIB / MODEL_DIR.
//
// Run:
//
//	ORT_LIB=/path/libonnxruntime.so.1.23.2 \
//	  MODEL_DIR=/path/to/deepdoc \
//	  go test -tags "native_det integration" -run TestNativeAnalyzerInProcess \
//	  ./internal/deepdoc/parser/pdf/inference/native/...
func TestNativeAnalyzerInProcess(t *testing.T) {
	ortLib := os.Getenv("ORT_LIB")
	modelDir := os.Getenv("MODEL_DIR")
	if ortLib == "" || modelDir == "" {
		t.Skip("ORT_LIB and MODEL_DIR required (in-process backend integration)")
	}
	if err := Register(modelDir, ortLib); err != nil {
		t.Fatalf("Register: %v", err)
	}
	if !Serving() {
		t.Skip("in-process backend not serving (ORT/models absent)")
	}
	a, err := NewAnalyzer(modelDir)
	if err != nil {
		t.Fatalf("NewAnalyzer: %v", err)
	}

	// page0.png is a content page with a known DLA golden (see dla-native
	// testdata). Reuse it to prove the DocAnalyzer path runs real inference.
	imgPath := filepath.Join("..", "..", "..", "..", "dla-native", "testdata", "page0.png")
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
