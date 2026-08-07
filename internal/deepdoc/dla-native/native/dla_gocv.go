//go:build gocv

package native

// dla_gocv.go — DLA preprocessing for the gocv build.
//
// Reaches 1:1 parity with the Python reference (ref_dla.py) by using cv2 for
// both decode and resize, exactly like det_gocv.go does for detection.
//
//  1. The source is re-decoded with cv2 (gocv.IMDecode/IMRead) when a path or
//     in-memory JPEG is available; otherwise the Go-decoded pixels are wrapped
//     in a gocv.Mat. cv2 decode matches deepdoc's cv2/PIL decode exactly,
//     whereas Go's image/jpeg yields slightly different pixels.
//  2. gocv.Resize IS cv2.resize (INTER_LINEAR), so the letterboxed CHW blob is
//     bit-exact with what ref_dla.py feeds the ONNX model.
//
// The border fill (114) and CHW assembly are shared via dlaLetterbox, so only
// the resize source differs from the pure-Go path.
//
// ref_dla.py decodes PIL RGB then applies cv2.cvtColor(RGB, BGR2RGB); that swap
// on an RGB buffer yields BGR, so the model is fed BGR. A cv2 decode is BGR
// already, so NO CvtColor is applied here.

import (
	"image"

	"gocv.io/x/gocv"
)

func dlaPreprocess(img *Image) (blob []float32, scaleFactor [4]float32) {
	newW, newH, dw, dh := dlaGeom(img)

	var srcMat gocv.Mat
	switch {
	case len(img.Bytes) > 0:
		srcMat, _ = gocv.IMDecode(img.Bytes, gocv.IMReadColor) // BGR
	case img.Path != "":
		srcMat = gocv.IMRead(img.Path, gocv.IMReadColor) // BGR
	default:
		bgr := img.ToBGR()
		m, err := gocv.NewMatFromBytes(img.H, img.W, gocv.MatTypeCV8UC3, bgr)
		if err != nil {
			resized := BilinearResize(bgr, img.W, img.H, newW, newH)
			return dlaLetterbox(resized, newW, newH, dw, dh), dlaScaleFactor(img, newW, newH, dw, dh)
		}
		srcMat = m
	}
	defer srcMat.Close()

	dst := gocv.NewMat()
	defer dst.Close()
	gocv.Resize(srcMat, &dst, image.Pt(newW, newH), 0, 0, gocv.InterpolationLinear)
	rb := dst.ToBytes() // length newH*newW*3, BGR
	return dlaLetterbox(rb, newW, newH, dw, dh), dlaScaleFactor(img, newW, newH, dw, dh)
}
