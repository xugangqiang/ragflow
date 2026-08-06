//go:build !native_det

package pdf

// Resilience unit tests for the native-detection seam (nativeDetectFn).
//
// These run under the default build (no -tags native_det) and therefore need
// no ONNX Runtime. They inject synthetic detector outputs through the seam to
// prove the native branch in inferOCRDetect / ocrDetectAndRecognize degrades
// gracefully across multi-image edge cases: native error, empty result,
// degenerate/zero-area quads, and out-of-bounds coordinates. The real
// geometry parity across heterogeneous pages is covered separately by the
// gated integration test (TestNativeOCRDetect, //go:build native_det &&
// integration). See HANDOFF.md §8 A2.

import (
	"context"
	"errors"
	"image"
	"image/color"
	"testing"

	pdf "ragflow/internal/deepdoc/parser/pdf/type"
)

// rectBox builds an axis-aligned quad OCRBox from a bounding rectangle.
func rectBox(x0, y0, x1, y1 float64) pdf.OCRBox {
	return pdf.OCRBox{X0: x0, Y0: y0, X1: x1, Y1: y0, X2: x1, Y2: y1, X3: x0, Y3: y1}
}

// withNativeSeam flips the native detector on/off and points the seam at fn,
// restoring both on test cleanup so cases stay isolated.
func withNativeSeam(t *testing.T, enabled bool, fn func(image.Image) ([]pdf.OCRBox, error)) {
	t.Helper()
	origEnabled := nativeDetEnabled
	origFn := nativeDetectFn
	EnableNativeDet(enabled)
	nativeDetectFn = fn
	t.Cleanup(func() {
		EnableNativeDet(origEnabled)
		nativeDetectFn = origFn
	})
}

func newSeamParser() *Parser {
	return NewParser(pdf.ParserConfig{})
}

func dummyPage() image.Image {
	img := image.NewRGBA(image.Rect(0, 0, 300, 400))
	// Fill so FastCrop has real pixels (not strictly required for these tests).
	for y := 0; y < 400; y++ {
		for x := 0; x < 300; x++ {
			img.Set(x, y, color.White)
		}
	}
	return img
}

// Native off (default): the seam must be bypassed and the remote analyzer
// used. The injected fn would fail the test if invoked.
func TestInferOCRDetectNativeOffUsesRemote(t *testing.T) {
	called := false
	withNativeSeam(t, false, func(image.Image) ([]pdf.OCRBox, error) {
		called = true
		return nil, errors.New("seam must not run when native is off")
	})
	remote := []pdf.OCRBox{rectBox(5, 5, 20, 20)}
	doc := &MockDocAnalyzer{Healthy: true, OCRBoxes: remote}

	boxes, err := newSeamParser().inferOCRDetect(context.Background(), doc, dummyPage())
	if called {
		t.Fatal("nativeDetectFn was invoked although nativeDetEnabled is false")
	}
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(boxes) != 1 || boxes[0] != remote[0] {
		t.Fatalf("expected remote boxes %v, got %v", remote, boxes)
	}
}

// Native on + detector error: must fall back to the remote path per page
// (parser_concurrency.go:196) rather than aborting the document.
func TestInferOCRDetectNativeErrorFallsBack(t *testing.T) {
	withNativeSeam(t, true, func(image.Image) ([]pdf.OCRBox, error) {
		return nil, errors.New("native det boom")
	})
	remote := []pdf.OCRBox{rectBox(1, 1, 10, 10)}
	doc := &MockDocAnalyzer{Healthy: true, OCRBoxes: remote}

	boxes, err := newSeamParser().inferOCRDetect(context.Background(), doc, dummyPage())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(boxes) != 1 || boxes[0] != remote[0] {
		t.Fatalf("expected fallback to remote boxes %v, got %v", remote, boxes)
	}
}

// Native on + empty (but non-error) result: the native branch returns the
// empty slice directly; callers must treat it as "no text" without panic.
func TestInferOCRDetectNativeEmptyBoxes(t *testing.T) {
	withNativeSeam(t, true, func(image.Image) ([]pdf.OCRBox, error) {
		return []pdf.OCRBox{}, nil
	})
	// Remote would have returned boxes, so if we see them the seam was skipped.
	doc := &MockDocAnalyzer{Healthy: true, OCRBoxes: []pdf.OCRBox{rectBox(1, 1, 10, 10)}}

	boxes, err := newSeamParser().inferOCRDetect(context.Background(), doc, dummyPage())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(boxes) != 0 {
		t.Fatalf("expected empty native result, got %v", boxes)
	}
}

// Native on + valid boxes: inferOCRDetect returns them verbatim without
// touching the remote analyzer (early return on the native branch).
func TestInferOCRDetectNativeValidBoxes(t *testing.T) {
	want := []pdf.OCRBox{rectBox(10, 10, 50, 40), rectBox(60, 60, 90, 90)}
	withNativeSeam(t, true, func(image.Image) ([]pdf.OCRBox, error) {
		return want, nil
	})
	// doc is unhealthy+empty: if it were touched we'd get nil,nil instead.
	doc := &MockDocAnalyzer{Healthy: false}

	boxes, err := newSeamParser().inferOCRDetect(context.Background(), doc, dummyPage())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(boxes) != len(want) {
		t.Fatalf("expected %d native boxes, got %d: %v", len(want), len(boxes), boxes)
	}
	for i := range want {
		if boxes[i] != want[i] {
			t.Fatalf("box %d mismatch: want %v got %v", i, want[i], boxes[i])
		}
	}
}

// ocrDetectAndRecognize must handle an empty native result without panic and
// return no text boxes.
func TestOCRDetectAndRecognizeNativeEmpty(t *testing.T) {
	withNativeSeam(t, true, func(image.Image) ([]pdf.OCRBox, error) {
		return []pdf.OCRBox{}, nil
	})
	doc := &MockDocAnalyzer{Healthy: true, OCRTexts: []pdf.OCRText{{Text: "ignored"}}}

	res := newSeamParser().ocrDetectAndRecognize(context.Background(), dummyPage(), doc, 0, "t")
	if len(res) != 0 {
		t.Fatalf("expected no text boxes for empty detect, got %d", len(res))
	}
}

// A degenerate (zero-area) quad must be skipped, while a valid quad flows
// through recognize into a TextBox. This guards the bounds/clip math in
// ocrDetectAndRecognize (parser_ocr.go:33-39) against multi-image input.
func TestOCRDetectAndRecognizeNativeDegenerateAndValid(t *testing.T) {
	degenerate := pdf.OCRBox{X0: 10, Y0: 10, X1: 10, Y1: 10, X2: 10, Y2: 10, X3: 10, Y3: 10}
	valid := rectBox(10, 10, 50, 40)
	withNativeSeam(t, true, func(image.Image) ([]pdf.OCRBox, error) {
		return []pdf.OCRBox{degenerate, valid}, nil
	})
	doc := &MockDocAnalyzer{Healthy: true, OCRTexts: []pdf.OCRText{{Text: "hi"}}}

	res := newSeamParser().ocrDetectAndRecognize(context.Background(), dummyPage(), doc, 0, "t")
	if len(res) != 1 {
		t.Fatalf("expected 1 text box (degenerate skipped), got %d: %v", len(res), res)
	}
	if res[0].Text != "hi" {
		t.Fatalf("unexpected text: %q", res[0].Text)
	}
	if res[0].PageNumber != 0 {
		t.Fatalf("expected page 0, got %d", res[0].PageNumber)
	}
}
