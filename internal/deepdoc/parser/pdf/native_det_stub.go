//go:build !native_det

package pdf

// Default-build stub for the native DB detector. Keeps the nativeOCRDetect
// symbol resolvable so inferOCRDetect can reference it unconditionally; the
// real implementation lives in native_det.go (//go:build native_det).

import (
	"errors"
	"image"

	pdf "ragflow/internal/deepdoc/parser/pdf/type"
)

// nativeDetEnabled is always false in the default build.
var nativeDetEnabled bool

// nativeDetectFn is the seam inferOCRDetect calls for the native path. In the
// default build it points at the stub nativeOCRDetect (returns an error so
// inferOCRDetect falls back to the remote service), but tests can reassign it
// to exercise the native branch without -tags native_det.
var nativeDetectFn = nativeOCRDetect

// EnableNativeDet is a no-op without -tags native_det.
func EnableNativeDet(enabled bool) { nativeDetEnabled = enabled }

func nativeOCRDetect(img image.Image) ([]pdf.OCRBox, error) {
	return nil, errors.New("native det not compiled in; rebuild with -tags native_det")
}
