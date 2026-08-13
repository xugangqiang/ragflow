//go:build native_det

// Package infnative provides an in-process DeepDoc DocAnalyzer backend.
//
// It wraps the ONNX Runtime inference library in the standalone dla-native
// module (import path dla-native/native) so the PDF parser can run DLA/TSR/OCR
// locally on CPU, with no Python service in the loop. It is registered as the
// local fallback used when no external DeepDoc HTTP service is configured
// (DEEPDOC_URL unset); when an external service is configured the parser still
// talks to it over HTTP and never silently falls back to this backend.
//
// The package imports onnxruntime_go (cgo), so it is build-tag gated
// (native_det) and only the server binary built with that tag opts into it.
// The parser package itself stays free of the onnxruntime dependency for its
// unit-test build path.
package infnative

import (
	"context"
	"fmt"
	"image"

	"dla-native/native"
	"ragflow/internal/common"
	"ragflow/internal/deepdoc/parser/pdf/inference"
	deepdoctype "ragflow/internal/deepdoc/parser/type"
	"ragflow/internal/parser/parser"
)

// registeredModelDir is the model directory recorded by Register, used by
// Serving for startup diagnostics.
var registeredModelDir string

// NativeAnalyzer runs DeepDoc vision inference in-process. It satisfies
// doctype.DocAnalyzer, so the PDF parser consumes it through the exact same
// interface as the HTTP-backed Client.
type NativeAnalyzer struct {
	modelDir string
}

var _ deepdoctype.DocAnalyzer = (*NativeAnalyzer)(nil)

// NewAnalyzer builds a NativeAnalyzer after verifying ONNX Runtime is
// initialized and every required model file exists. It returns an error when
// the in-process backend cannot serve, letting the caller (the registration
// factory) fall back to the empty analyzer instead of panicking on an
// uninitialized ONNX environment.
func NewAnalyzer(modelDir string) (*NativeAnalyzer, error) {
	if !native.Initialized() {
		return nil, fmt.Errorf("deepdoc native: onnxruntime not initialized")
	}
	if !common.HasModelFiles(modelDir) {
		return nil, fmt.Errorf("deepdoc native: missing required model files in %s", modelDir)
	}
	return &NativeAnalyzer{modelDir: modelDir}, nil
}

// Register wires this backend into the parser as the local in-process
// fallback. Call it once at process start (the server binary) after
// resolving modelDir/ortLib. A non-empty ortLib is handed to native.InitORT;
// if ORT is already initialized (or ortLib is empty and InitORT ran elsewhere)
// this is a no-op for that step. The factory returns false when the backend
// cannot serve, so the parser degrades to the empty analyzer rather than
// crashing.
func Register(modelDir, ortLib string) error {
	registeredModelDir = modelDir
	if ortLib != "" {
		if err := native.InitORT(ortLib); err != nil {
			return fmt.Errorf("deepdoc native: init onnxruntime: %w", err)
		}
	}
	parser.SetNativeDocAnalyzerFactory(func() (deepdoctype.DocAnalyzer, bool) {
		a, err := NewAnalyzer(modelDir)
		if err != nil {
			return nil, false
		}
		return a, true
	})
	return nil
}

// canServe reports whether the backend can serve from modelDir: ONNX Runtime
// is initialized and every required model file is present. Serving and
// NativeAnalyzer.Health share this exact check; they differ only in which
// model directory they probe (the process-registered one vs the instance's).
func canServe(modelDir string) bool {
	if !native.Initialized() {
		return false
	}
	return common.HasModelFiles(modelDir)
}

// Serving reports whether the backend can currently serve from the
// process-registered model directory. Used for startup logging only; the
// parser's factory already gates on the same check via canServe.
func Serving() bool {
	return canServe(registeredModelDir)
}

// DLA runs layout detection on a page image.
func (a *NativeAnalyzer) DLA(ctx context.Context, img image.Image) ([]deepdoctype.DLARegion, error) {
	ni, err := native.FromImage(img)
	if err != nil {
		return nil, err
	}
	res, err := native.RunDLA(ctx, a.modelDir, ni)
	if err != nil {
		return nil, err
	}
	labels := inference.DefaultDLALabels()
	out := make([]deepdoctype.DLARegion, 0, len(res.Boxes))
	for _, b := range res.Boxes {
		label := ""
		if b.Class >= 0 && int(b.Class) < len(labels) {
			label = labels[b.Class]
		}
		out = append(out, deepdoctype.DLARegion{
			X0: float64(b.X0), Y0: float64(b.Y0),
			X1: float64(b.X1), Y1: float64(b.Y1),
			Label:      label,
			Confidence: float64(b.Score),
		})
	}
	return out, nil
}

// TSR recognises table structure from a cropped image.
func (a *NativeAnalyzer) TSR(ctx context.Context, img image.Image) ([]deepdoctype.TSRCell, error) {
	ni, err := native.FromImage(img)
	if err != nil {
		return nil, err
	}
	res, err := native.RunTSR(ctx, a.modelDir, ni)
	if err != nil {
		return nil, err
	}
	out := make([]deepdoctype.TSRCell, 0, len(res.Boxes))
	for _, b := range res.Boxes {
		out = append(out, deepdoctype.TSRCell{
			X0: float64(b.X0), Y0: float64(b.Top),
			X1: float64(b.X1), Y1: float64(b.Bottom),
			Label: b.Label,
		})
	}
	return out, nil
}

// OCRDetect detects text regions (quad boxes) in a cropped image.
func (a *NativeAnalyzer) OCRDetect(ctx context.Context, img image.Image) ([]deepdoctype.OCRBox, error) {
	ni, err := native.FromImage(img)
	if err != nil {
		return nil, err
	}
	res, err := native.RunDet(ctx, a.modelDir, ni)
	if err != nil {
		return nil, err
	}
	out := make([]deepdoctype.OCRBox, 0, len(res.Boxes))
	for _, b := range res.Boxes {
		out = append(out, deepdoctype.OCRBox{
			X0: float64(b.Pts[0][0]), Y0: float64(b.Pts[0][1]),
			X1: float64(b.Pts[1][0]), Y1: float64(b.Pts[1][1]),
			X2: float64(b.Pts[2][0]), Y2: float64(b.Pts[2][1]),
			X3: float64(b.Pts[3][0]), Y3: float64(b.Pts[3][1]),
		})
	}
	return out, nil
}

// OCRRecognize recognizes text in a cropped image region.
func (a *NativeAnalyzer) OCRRecognize(ctx context.Context, img image.Image) ([]deepdoctype.OCRText, error) {
	ni, err := native.FromImage(img)
	if err != nil {
		return nil, err
	}
	res, err := native.RunOCRRec(ctx, a.modelDir, ni)
	if err != nil {
		return nil, err
	}
	return []deepdoctype.OCRText{{Text: res.Text, Confidence: float64(res.Score)}}, nil
}

// Health reports whether the backend can serve from this analyzer's model
// directory: ONNX Runtime is initialized and every required model file is
// present. It delegates to canServe.
func (a *NativeAnalyzer) Health() bool {
	return canServe(a.modelDir)
}
