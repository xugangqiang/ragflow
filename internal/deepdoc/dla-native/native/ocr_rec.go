package native

// ocr_rec.go — OCR text recognition (PP-OCRv4 CTC) recognizer.
//
// Ports deepdoc/vision/ocr.py TextRecognizer.resize_norm_img and
// deepdoc/vision/postprocess.py CTCLabelDecode, emitting the wire format from
// deepdoc/server/adapters/ocr_adapter.py (recognize mode).

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
)

const (
	recH        = 48
	recW        = 320
	recSeqLen   = 40
	recVocab    = 6625
	recMaxBatch = 1
)

// OCRRecResult is the recognized text for one cropped line.
type OCRRecResult struct {
	Text  string
	Score float32
}

// RunOCRRec recognizes a single cropped text-line image.
func RunOCRRec(ctx context.Context, modelDir string, img *Image) (OCRRecResult, error) {
	chars, err := loadCharDict(filepath.Join(modelDir, "ocr.res"))
	if err != nil {
		return OCRRecResult{}, err
	}
	blob := ocrRecPreprocess(img)
	// 0 → all cores, matching deepdoc's Python onnxruntime for bit-stable
	// parity (no contour extraction in the OCR-rec Run path).
	sess, release, err := getModelSession(filepath.Join(modelDir, "rec.onnx"), "x",
		[]int64{recMaxBatch, 3, recH, recW}, "softmax_11.tmp_0",
		[]int64{recMaxBatch, recSeqLen, recVocab}, 0)
	if err != nil {
		return OCRRecResult{}, err
	}
	defer release()

	out, err := sess.Run(ctx, blob)
	if err != nil {
		return OCRRecResult{}, err
	}
	return ocrRecCTCDecode(out, chars), nil
}

func ocrRecPreprocess(img *Image) []float32 {
	bgr := img.ToBGR()
	h, w := img.H, img.W
	ratio := float64(w) / float64(h)
	resizedW := int(math.Ceil(recH * ratio))
	if resizedW > recW {
		resizedW = recW
	}
	resized := bilinearResize(bgr, w, h, resizedW, recH)
	blob := make([]float32, 3*recH*recW) // zero-filled (padded right)
	for y := 0; y < recH; y++ {
		for x := 0; x < resizedW; x++ {
			for c := 0; c < 3; c++ {
				v := float32(resized[(y*resizedW+x)*3+c]) / 255.0
				v = (v - 0.5) / 0.5
				blob[c*recH*recW+y*recW+x] = v
			}
		}
	}
	return blob
}

// loadCharDict returns the full decode vocabulary: ["blank"] + <ocr.res lines> + " ".
func loadCharDict(path string) ([]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	lines := strings.Split(string(data), "\n")
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	chars := make([]string, 0, len(lines)+2)
	chars = append(chars, "blank")
	chars = append(chars, lines...)
	chars = append(chars, " ") // use_space_char
	return chars, nil
}

func ocrRecCTCDecode(out []float32, chars []string) OCRRecResult {
	// out layout: [recMaxBatch, recSeqLen, recVocab]; take batch 0.
	var text strings.Builder
	var probs []float32
	prev := -1
	var meanAcc float32
	var meanN int
	for t := 0; t < recSeqLen; t++ {
		base := t * recVocab
		bestIdx, bestProb := 0, float32(-1e9)
		for v := 0; v < recVocab; v++ {
			if out[base+v] > bestProb {
				bestProb = out[base+v]
				bestIdx = v
			}
		}
		if bestIdx == 0 { // blank
			prev = 0
			continue
		}
		if bestIdx != prev {
			if bestIdx < len(chars) {
				text.WriteString(chars[bestIdx])
				probs = append(probs, bestProb)
			}
		}
		prev = bestIdx
	}
	_ = meanN
	for _, p := range probs {
		meanAcc += p
	}
	score := float32(1.0)
	if len(probs) > 0 {
		score = meanAcc / float32(len(probs))
	}
	return OCRRecResult{Text: text.String(), Score: round4(score)}
}

// Wire emits the Go DocAnalyzer OCR-rec format: {"output": [[[text, 1.0]]]}.
func (r OCRRecResult) Wire() string {
	// Mirror ocr_adapter.recognize: confidence filled with 1.0, 4-level nesting.
	pair := []any{r.Text, 1.0}
	arr1 := []any{pair}
	arr2 := []any{arr1}
	arr3 := []any{arr2}
	out, _ := json.Marshal(map[string]any{"output": arr3})
	return string(out)
}
