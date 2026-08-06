package native

// dla.go — DLA (layout detection) recognizer.
//
// Ports deepdoc/vision/layout_recognizer.py LayoutRecognizer4YOLOv10 and
// deepdoc/server/adapters/dla_adapter.py. Self-contained: owns its
// preprocessing, inference, postprocessing, and wire encoding.

import (
	"encoding/json"
	"math"
	"path/filepath"
	"strings"
)

const dlaInputSize = 1024
const dlaMaxBoxes = 300

// DLABox is one detected layout region in the wire format.
type DLABox struct {
	X0, Y0, X1, Y1 float32
	Score          float32
	Class          int
}

// DLAResult is the full DLA output.
// W/H are the source image dimensions, used to clamp boxes into bounds (mirrors
// dla_adapter.py, which clamps every coordinate to [0, width]/[0, height]).
type DLAResult struct {
	Boxes []DLABox
	W, H  int
}

var (
	// yoloDlaLabels mirrors LayoutRecognizer4YOLOv10.labels (10 classes).
	yoloDlaLabels = []string{
		"title", "Text", "Reference", "Figure", "Figure caption",
		"Table", "Table caption", "Table caption", "Equation", "Figure caption",
	}
	// dlaClassMap mirrors dla_adapter.DLA_CLASS_MAP.
	dlaClassMap = map[string]int{
		"title": 0, "text": 1, "reference": 2, "figure": 3, "figure caption": 4,
		"table": 5, "table caption": 6, "equation": 8,
	}
)

// RunDLA runs layout detection on a page image.
func RunDLA(modelDir string, img *Image) (DLAResult, error) {
	blob, sf := dlaPreprocess(img)
	sess, release, err := getModelSession(filepath.Join(modelDir, "layout.onnx"), "images",
		[]int64{1, 3, dlaInputSize, dlaInputSize}, "output0",
		[]int64{1, dlaMaxBoxes, 6})
	if err != nil {
		return DLAResult{}, err
	}
	defer release()

	out, err := sess.Run(blob)
	if err != nil {
		return DLAResult{}, err
	}
	res := dlaPostprocess(out, sf)
	res.W, res.H = img.W, img.H
	return res, nil
}

func dlaPreprocess(img *Image) (blob []float32, scaleFactor [4]float32) {
	r := math.Min(float64(dlaInputSize)/float64(img.H), float64(dlaInputSize)/float64(img.W))
	newW := int(math.Round(float64(img.W) * r))
	newH := int(math.Round(float64(img.H) * r))
	dw := (float64(dlaInputSize) - float64(newW)) / 2.0
	dh := (float64(dlaInputSize) - float64(newH)) / 2.0

	bgr := img.ToBGR()
	resized := BilinearResize(bgr, img.W, img.H, newW, newH)

	top := int(math.Round(dh - 0.1))
	left := int(math.Round(dw - 0.1))

	blob = make([]float32, 3*dlaInputSize*dlaInputSize)
	for y := 0; y < dlaInputSize; y++ {
		for x := 0; x < dlaInputSize; x++ {
			var cr, cg, cb float32 = 114, 114, 114
			inY, inX := y-top, x-left
			if inY >= 0 && inY < newH && inX >= 0 && inX < newW {
				o := (inY*newW + inX) * 3
				cb = float32(resized[o])
				cg = float32(resized[o+1])
				cr = float32(resized[o+2])
			}
			// CHW; model expects BGR, so channel 0 = blue, 2 = red.
			blob[0*dlaInputSize*dlaInputSize+y*dlaInputSize+x] = cb / 255.0
			blob[1*dlaInputSize*dlaInputSize+y*dlaInputSize+x] = cg / 255.0
			blob[2*dlaInputSize*dlaInputSize+y*dlaInputSize+x] = cr / 255.0
		}
	}
	scaleFactor = [4]float32{
		float32(float64(img.W) / float64(newW)),
		float32(float64(img.H) / float64(newH)),
		float32(dw), float32(dh),
	}
	return
}

func dlaPostprocess(out []float32, sf [4]float32) DLAResult {
	const scoreThr = 0.08
	type cand struct {
		Box
		cls int
	}
	cands := make([]cand, 0, dlaMaxBoxes)
	for i := 0; i < dlaMaxBoxes; i++ {
		base := i * 6
		score := out[base+4]
		if score <= scoreThr {
			continue
		}
		cls := int(out[base+5] + 0.5)
		cands = append(cands, cand{
			Box: Box{
				X0:    (out[base+0] - sf[2]) * sf[0],
				Y0:    (out[base+1] - sf[3]) * sf[1],
				X1:    (out[base+2] - sf[2]) * sf[0],
				Y1:    (out[base+3] - sf[3]) * sf[1],
				Score: score,
			},
			cls: cls,
		})
	}

	byClass := map[int][]int{}
	for i, c := range cands {
		byClass[c.cls] = append(byClass[c.cls], i)
	}
	res := DLAResult{}
	for cls, idxs := range byClass {
		sub := make([]Box, len(idxs))
		for k, i := range idxs {
			sub[k] = cands[i].Box
		}
		for _, keep := range NMS(sub, 0.45, true) {
			res.Boxes = append(res.Boxes, DLABox{
				X0: round2(sub[keep].X0), Y0: round2(sub[keep].Y0),
				X1: round2(sub[keep].X1), Y1: round2(sub[keep].Y1),
				Score: round4(sub[keep].Score), Class: cls,
			})
		}
	}
	// Re-map class ids through the OSS label->Go index map.
	mapped := res.Boxes[:0]
	for _, b := range res.Boxes {
		label := yoloDlaLabels[b.Class]
		goCls, ok := dlaClassMap[strings.ToLower(label)]
		if !ok {
			continue
		}
		b.Class = goCls
		mapped = append(mapped, b)
	}
	res.Boxes = mapped
	return res
}

// Wire encodes the result in the exact format the Go DocAnalyzer consumes:
// {"bboxes": [[x0,y0,x1,y1,score,class_id], ...]}.
func (r DLAResult) Wire() string {
	rows := make([][]float32, 0, len(r.Boxes))
	w, h := float32(r.W), float32(r.H)
	for _, b := range r.Boxes {
		// Clamp into image bounds (mirrors dla_adapter.py).
		x0 := minf(maxf(b.X0, 0), w)
		y0 := minf(maxf(b.Y0, 0), h)
		x1 := minf(maxf(b.X1, 0), w)
		y1 := minf(maxf(b.Y1, 0), h)
		rows = append(rows, []float32{x0, y0, x1, y1, b.Score, float32(b.Class)})
	}
	b, _ := json.Marshal(map[string]any{"bboxes": rows})
	return string(b)
}

func round2(v float32) float32 { return float32(math.Round(float64(v)*100) / 100) }
func round4(v float32) float32 { return float32(math.Round(float64(v)*10000) / 10000) }
