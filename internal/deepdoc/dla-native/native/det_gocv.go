//go:build gocv

package native

// det_gocv.go — OCR text detection (DB) OpenCV geometry path.
//
// Build tag "gocv". The shared entry point, types, the true round-offset
// unclip, and the wire format live in det_core.go. The default build (det.go,
// !gocv) is pure-Go.
//
// This path reaches 1:1 with the Python service by combining:
//   1. detPreprocess uses gocv.Resize, which IS cv2.resize (INTER_LINEAR), so
//      the normalized CHW blob is bit-exact with what TextDetector feeds the
//      ONNX model.
//   2. dbPostProcess uses gocv.FindContours — the exact OpenCV call
//      DBPostProcess makes — so the contour set matches Python exactly.
// The minimum-area rectangle and its corners come from the shared pure-Go
// minAreaRect (a float-precision port of cv2.minAreaRect + cv2.boxPoints):
// gocv 0.31.0's own MinAreaRect rounds the center/size to int, which loses
// sub-pixel accuracy, whereas the float port keeps it. unclip stays the
// pure-Go round offset (Clipper JT_ROUND equivalent) because gocv does not
// bind Clipper; it is the canonical geometry and matches the Python
// polygon-area/length expansion.

import (
	"fmt"
	"image"
	"math"
	"os"

	"gocv.io/x/gocv"
)

// detPreprocess uses gocv.Resize (== cv2.resize INTER_LINEAR) for the
// DetResizeForTest step, then the shared normalizeCHW. When the source path is
// known (img.Path set by main.go), the image is re-decoded with cv2
// (gocv.IMRead) so the preprocessing blob matches deepdoc's cv2/PIL decode
// exactly — Go's image/jpeg decoder yields slightly different pixel values
// and that small shift propagates into the segmentation mask and, after
// unclip+scale, into the final boxes. deepdoc's det pipeline feeds RGB
// (PIL, no cvtColor), so a cv2 decode (BGR) is converted BGR->RGB before
// normalize.
func detPreprocess(img *Image) (blob []float32, resizeH, resizeW, srcH, srcW int) {
	srcH, srcW = img.H, img.W
	h, w := srcH, srcW

	ratio := 1.0
	if math.Max(float64(h), float64(w)) > detLimitSideLen {
		ratio = float64(detLimitSideLen) / math.Max(float64(h), float64(w))
	}
	resizeH = int(math.Round(float64(h) * ratio))
	resizeW = int(math.Round(float64(w) * ratio))
	resizeH = int(math.Max(float64(round32(resizeH)), 32))
	resizeW = int(math.Max(float64(round32(resizeW)), 32))

	var srcMat gocv.Mat
	switch {
	case len(img.Bytes) > 0:
		// In-memory JPEG (from NewImageForDet): cv2 decode via IMDecode so the
		// preprocessing blob matches deepdoc's cv2/JPEG decode, no temp file.
		srcMat, _ = gocv.IMDecode(img.Bytes, gocv.IMReadColor) // BGR
	case img.Path != "":
		srcMat = gocv.IMRead(img.Path, gocv.IMReadColor) // BGR
	default:
		bgr := img.ToBGR()
		m, err := gocv.NewMatFromBytes(h, w, gocv.MatTypeCV8UC3, bgr)
		if err != nil {
			resized := bilinearResize(bgr, w, h, resizeW, resizeH)
			return normalizeCHW(resized, resizeH, resizeW, srcW, srcH), resizeH, resizeW, srcH, srcW
		}
		srcMat = m
	}
	defer srcMat.Close()

	// deepdoc feeds RGB; cv2 decode is BGR, so swap to RGB.
	rgb := gocv.NewMat()
	defer rgb.Close()
	gocv.CvtColor(srcMat, &rgb, gocv.ColorBGRToRGB)

	dst := gocv.NewMat()
	defer dst.Close()
	gocv.Resize(rgb, &dst, image.Pt(resizeW, resizeH), 0, 0, gocv.InterpolationLinear)
	rb := dst.ToBytes() // length resizeH*resizeW*3, RGB
	return normalizeCHW(rb, resizeH, resizeW, srcW, srcH), resizeH, resizeW, srcH, srcW
}

// dbPostProcess thresholds pred and post-processes it in-process via OpenCV
// findContours. It mirrors DBPostProcess.boxes_from_bitmap +
// TextDetector.filter_tag_det_res.
func dbPostProcess(pred []float32, h, w, srcH, srcW int) ([]DetBox, []float32) {
	seg := make([]byte, h*w)
	for i, v := range pred {
		if v > detThresh {
			seg[i] = 1
		}
	}
	return dbPostProcessSeg(seg, pred, h, w, srcH, srcW)
}

// dbPostProcessSeg runs contour extraction + box fitting on an already
// thresholded mask. findContours returns the exact contour set cv2 would (the
// mask is the same byte layout), so the raw contours already match Python.
// pred is used solely for box scoring (boxScoreFast).
func dbPostProcessSeg(seg []byte, pred []float32, h, w, srcH, srcW int) ([]DetBox, []float32) {
	segMat := gocv.NewMatWithSize(h, w, gocv.MatTypeCV8UC1)
	defer segMat.Close()
	for i, v := range seg {
		if v != 0 {
			segMat.SetUCharAt(0, i, v)
		}
	}

	contours := gocv.FindContours(segMat, gocv.RetrievalList, gocv.ChainApproxSimple)
	defer contours.Close()
	if os.Getenv("DLA_CNT") != "" {
		fmt.Fprintf(os.Stderr, "DLA_CONTOURS=%d\n", contours.Size())
	}

	boxes := make([]DetBox, 0, contours.Size())
	scores := make([]float32, 0, contours.Size())
	for _, contour := range contours.ToPoints() {
		if len(contour) < 3 {
			continue
		}
		cpts := make([]pt, len(contour))
		for k, p := range contour {
			cpts[k] = pt{X: float64(p.X), Y: float64(p.Y)}
		}
		// cv2.minAreaRect (and thus deepdoc) computes the minimum-area
		// enclosing rectangle of the contour's CONVEX HULL, not the raw
		// contour: the min-area rect of a point set equals that of its hull.
		// The pure-Go rotating-calipers below is only exact on convex input,
		// so convexify first — feeding the raw (dense, non-convex) contour
		// returns an axis-aligned, off-by-pixels rect for slightly-rotated
		// regions. convexHull is idempotent on already-convex input.
		quad, sside := minAreaRect(convexHull(cpts))
		if sside < detMinSize {
			continue
		}
		dlaRecordPreUnclip(quad)
		score := boxScoreFast(pred, w, h, quad)
		if score < detBoxThresh {
			continue
		}
		// unclip (expand) then re-rect.
		expanded := unclip(quad, detUnclipRatio)
		quad2, sside2 := minAreaRect(expanded)
		dlaRecordPostUnclip(quad2)
		if sside2 < detMinSize+2 {
			continue
		}
		// Scale back to source coordinates.
		var q [4][2]float32
		for i := 0; i < 4; i++ {
			qx := clampf(math.Round(quad2[i].X/float64(w)*float64(srcW)), 0, float64(srcW))
			qy := clampf(math.Round(quad2[i].Y/float64(h)*float64(srcH)), 0, float64(srcH))
			q[i] = [2]float32{float32(qx), float32(qy)}
		}
		boxes = append(boxes, DetBox{Pts: q, Score: score})
		scores = append(scores, score)
	}

	// filter_tag_det_res: clockwise order + integer clip + drop tiny boxes.
	dlaFlushPreUnclip()
	dlaFlushPostUnclip()
	return filterTagDetRes(boxes, srcH, srcW), scores
}
