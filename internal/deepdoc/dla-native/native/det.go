//go:build !gocv

package native

// det.go — OCR text detection (DB) PURE-GO geometry path.
//
// Ports deepdoc/vision/ocr.py TextDetector and deepdoc/vision/postprocess.py
// DBPostProcess (box_type="quad"). The shared entry point, types, the
// true round-offset unclip, and the wire format live in det_core.go.
//
// This is the pure-Go path. Inference is bit-exact with the Python service
// (same ONNX Runtime build). The DB geometry — connected-components in place
// of findContours, rotating-calipers minAreaRect, and a scanline fillPoly for
// box_score_fast — is reimplemented in Go. It is geometrically faithful but
// NOT bit-exact to OpenCV/Clipper; box locations match to within a couple of
// pixels. The gocv build (det_gocv.go) swaps the resize and geometry for
// OpenCV C++ calls and reaches 1:1 with Python.

import "math"

// detPreprocess mirrors TextDetector's pre_process_list:
// DetResizeForTest(limit_side_len=960, limit_type="max") ->
// NormalizeImage(scale=1/255, mean, std, order="hwc") -> ToCHWImage.
// Returns the CHW float32 blob plus the resized and source dimensions.
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

	bgr := img.ToBGR()
	resized := BilinearResize(bgr, w, h, resizeW, resizeH)
	return normalizeCHW(resized, resizeH, resizeW, srcW, srcH), resizeH, resizeW, srcH, srcW
}

// dbPostProcess mirrors DBPostProcess.boxes_from_bitmap + TextDetector.filter_tag_det_res.
func dbPostProcess(pred []float32, h, w, srcH, srcW int) ([]DetBox, []float32) {
	// Binary segmentation mask.
	seg := make([]bool, h*w)
	for i, v := range pred {
		seg[i] = v > detThresh
	}

	// Connected components (8-connected) of the foreground.
	comps := connectedComponents(seg, w, h, detMaxCandidates)

	boxes := make([]DetBox, 0, len(comps))
	scores := make([]float32, 0, len(comps))
	for _, comp := range comps {
		hull := convexHull(comp)
		if len(hull) < 3 {
			continue
		}
		// Pre-unclip min-area rect + side check.
		pts, sside := minAreaRect(hull)
		if sside < detMinSize {
			continue
		}
		dlaRecordPreUnclip(pts)
		score := boxScoreFast(pred, w, h, pts)
		if detBoxThresh > score {
			continue
		}
		// unclip (expand) then re-rect.
		expanded := unclip(pts, detUnclipRatio)
		pts2, sside2 := minAreaRect(expanded[:])
		if sside2 < detMinSize+2 {
			continue
		}
		// Scale back to source coordinates (dest = source dims here).
		var q [4][2]float32
		for i := 0; i < 4; i++ {
			qx := clampf(math.Round(float64(pts2[i].X)/float64(w)*float64(srcW)), 0, float64(srcW))
			qy := clampf(math.Round(float64(pts2[i].Y)/float64(h)*float64(srcH)), 0, float64(srcH))
			q[i] = [2]float32{float32(qx), float32(qy)}
		}
		boxes = append(boxes, DetBox{Pts: q, Score: score})
		scores = append(scores, score)
	}

	// filter_tag_det_res: clockwise order + integer clip + drop tiny boxes.
	dlaFlushPreUnclip()
	return filterTagDetRes(boxes, srcH, srcW), scores
}

// connectedComponents labels 8-connected foreground and returns at most
// maxComps components (the largest ones), each as a slice of hull-candidate
// pixels (the full pixel set of the component).
func connectedComponents(seg []bool, w, h, maxComps int) [][]pt {
	labels := make([]int, len(seg))
	parent := []int{0}
	var find func(int) int
	find = func(x int) int {
		for parent[x] != x {
			parent[x] = parent[parent[x]]
			x = parent[x]
		}
		return x
	}
	union := func(a, b int) {
		ra, rb := find(a), find(b)
		if ra != rb {
			parent[ra] = rb
		}
	}
	next := 1
	neigh := [8][2]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			idx := y*w + x
			if !seg[idx] {
				continue
			}
			// Find an already-labeled 8-neighbor to union with.
			lab := 0
			for _, d := range neigh {
				nx, ny := x+d[0], y+d[1]
				if nx < 0 || ny < 0 || nx >= w || ny >= h {
					continue
				}
				nidx := ny*w + nx
				if seg[nidx] && labels[nidx] != 0 {
					if lab == 0 {
						lab = labels[nidx]
					} else {
						union(lab, labels[nidx])
					}
				}
			}
			if lab == 0 {
				lab = next
				parent = append(parent, next)
				next++
			}
			labels[idx] = lab
		}
	}
	// Gather pixels per root label.
	counts := map[int]int{}
	members := map[int][]pt{}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			idx := y*w + x
			if !seg[idx] || labels[idx] == 0 {
				continue
			}
			r := find(labels[idx])
			counts[r]++
			members[r] = append(members[r], pt{float64(x) + 0.5, float64(y) + 0.5})
		}
	}
	// Sort roots by size desc.
	roots := make([]int, 0, len(members))
	for r := range members {
		roots = append(roots, r)
	}
	sortIntsByCount(roots, counts)
	if len(roots) > maxComps {
		roots = roots[:maxComps]
	}
	out := make([][]pt, 0, len(roots))
	for _, r := range roots {
		out = append(out, members[r])
	}
	return out
}

// convexHull is defined in det_core.go (shared by both builds): a generic
// geometry helper used by the pure-Go dbPostProcess.

// minAreaRect is defined in det_core.go (shared by both builds): the pure-Go
// float-precision rotating-calipers port of cv2.minAreaRect.

