//go:build native_det

package pdf

// Native DB text-detection integration (dla-native).
//
// This file is compiled only under the `native_det` build tag. It runs the
// in-process Go port of DeepDoc's DBPostProcess detector (internal/deepdoc/
// dla-native) instead of calling the remote DeepDoc OCRDetect HTTP service,
// matching the same OCRBox shape inferOCRDetect already consumes.
//
// It is gated so the default build never pulls the dla-native module (and its
// ONNX Runtime / OpenCV CGO dependencies) into the main module. Opt in with
// `-tags native_det` and flip EnableNativeDet(true) (e.g. from service
// startup when RAGFLOW_NATIVE_DET=1). See HANDOFF.md §8 C1.

import (
	"context"
	"errors"
	"image"
	"os"

	"dla-native/native"
	pdf "ragflow/internal/deepdoc/parser/pdf/type"
)

// nativeDetEnabled toggles the native detector in inferOCRDetect. Off by
// default so production behavior is unchanged unless explicitly opted in.
var nativeDetEnabled bool

// nativeDetectFn is the seam inferOCRDetect calls for the native path. It
// defaults to the real nativeOCRDetect but is reassignable so unit tests can
// inject synthetic detector outputs (empty / error / degenerate boxes)
// without pulling in ONNX Runtime. See native_det_seam_test.go.
var nativeDetectFn = nativeOCRDetect

// EnableNativeDet turns the in-process native DB detector on or off.
func EnableNativeDet(enabled bool) { nativeDetEnabled = enabled }

// nativeOCRDetect runs the native DBPostProcess detector on a rendered page
// image and returns the detected text-region quads in the OCRBox shape
// inferOCRDetect expects.
func nativeOCRDetect(img image.Image) ([]pdf.OCRBox, error) {
	ortLib := os.Getenv("ORT_LIB")
	modelDir := os.Getenv("MODEL_DIR")
	if ortLib == "" || modelDir == "" {
		return nil, errors.New("native det requires ORT_LIB and MODEL_DIR env")
	}
	if err := native.InitORT(ortLib); err != nil {
		return nil, err
	}
	// Build the native Image in memory. The pure-Go build fills the pixel
	// buffer directly (no encode); the gocv build serializes to an in-memory
	// JPEG buffer that det_gocv re-decodes via cv2 (gocv.IMDecode) for
	// byte-exact parity with deepdoc — no temp file in either path.
	nimg, err := native.NewImageForDet(img)
	if err != nil {
		return nil, err
	}
	res, err := native.RunDet(context.Background(), modelDir, nimg)
	if err != nil {
		return nil, err
	}
	boxes := make([]pdf.OCRBox, 0, len(res.Boxes))
	for _, b := range res.Boxes {
		boxes = append(boxes, pdf.OCRBox{
			X0: float64(b.Pts[0][0]), Y0: float64(b.Pts[0][1]),
			X1: float64(b.Pts[1][0]), Y1: float64(b.Pts[1][1]),
			X2: float64(b.Pts[2][0]), Y2: float64(b.Pts[2][1]),
			X3: float64(b.Pts[3][0]), Y3: float64(b.Pts[3][1]),
		})
	}
	return boxes, nil
}
