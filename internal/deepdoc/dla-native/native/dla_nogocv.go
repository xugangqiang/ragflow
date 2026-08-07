//go:build !gocv

package native

// dla_nogocv.go — DLA preprocessing for the pure-Go build.
//
// Decodes via the shared Go image/jpeg decoder and resizes with the pure-Go
// BilinearResize. This is geometrically faithful to deepdoc's cv2 pipeline but
// NOT bit-exact: the Go JPEG decoder and bilinear sampler yield slightly
// different pixels, which propagate into box coordinates (the "pure-Go floor",
// ~2.6px on the expanded fixtures). The gocv build (dla_gocv.go) swaps the
// decode+resize for cv2 and reaches 1:1 parity with the Python reference.

// dlaPreprocess letterboxes the image into the 1024 canvas and returns the CHW
// blob plus the scale factor. See dlaLetterbox / dlaScaleFactor for details.
func dlaPreprocess(img *Image) (blob []float32, scaleFactor [4]float32) {
	newW, newH, dw, dh := dlaGeom(img)
	bgr := img.ToBGR()
	resized := BilinearResize(bgr, img.W, img.H, newW, newH)
	return dlaLetterbox(resized, newW, newH, dw, dh), dlaScaleFactor(img, newW, newH, dw, dh)
}
