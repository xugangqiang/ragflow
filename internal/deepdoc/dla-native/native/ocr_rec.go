package native

// ocr_rec.go — OCR text recognition (PP-OCRv4 CTC) recognizer.
//
// Ports deepdoc/vision/ocr.py TextRecognizer.resize_norm_img and
// deepdoc/vision/postprocess.py CTCLabelDecode, emitting the wire format from
// deepdoc/server/adapters/ocr_adapter.py (recognize mode).

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	ort "github.com/yalue/onnxruntime_go"
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

// RunOCRRec recognizes a single cropped text-line image. It is equivalent to a
// one-line batch (see RunOCRRecBatch): the line is resized against its own
// wh_ratio floored at 320/48, never the batch-max, so a caller recognizing
// lines independently gets the same result as a standalone Python
// TextRecognizer call.
func RunOCRRec(ctx context.Context, modelDir string, img *Image) (OCRRecResult, error) {
	chars, err := loadCharDict(filepath.Join(modelDir, "ocr.res"))
	if err != nil {
		return OCRRecResult{}, err
	}
	// A single image is its own batch: max_wh_ratio floors at recW/recH (matching
	// TextRecognizer.__call__'s init) but rises to the line's own ratio when
	// wider, so wide lines are NOT clamped back to 320.
	maxWhRatio := float64(recW) / float64(recH)
	if r := float64(img.W) / float64(img.H); r > maxWhRatio {
		maxWhRatio = r
	}
	return recognizeLine(ctx, modelDir, img, maxWhRatio, chars)
}

// RunOCRRecBatch recognizes a batch of cropped text-line images the way
// deepdoc's TextRecognizer.__call__ does: every line in the batch is resized to
// the SAME width derived from the batch's maximum wh_ratio (floored at 320/48),
// so a narrow line inside a wide batch is widened to the batch max rather than
// its own ratio. This makes Go match production batch semantics, where a page's
// lines are recognized together. Lines are recognized independently after the
// shared resize, so order is preserved and results are deterministic per batch.
func RunOCRRecBatch(ctx context.Context, modelDir string, imgs []*Image) ([]OCRRecResult, error) {
	chars, err := loadCharDict(filepath.Join(modelDir, "ocr.res"))
	if err != nil {
		return nil, err
	}
	maxWhRatio := float64(recW) / float64(recH)
	for _, img := range imgs {
		if r := float64(img.W) / float64(img.H); r > maxWhRatio {
			maxWhRatio = r
		}
	}
	out := make([]OCRRecResult, len(imgs))
	for i, img := range imgs {
		res, err := recognizeLine(ctx, modelDir, img, maxWhRatio, chars)
		if err != nil {
			return nil, err
		}
		out[i] = res
	}
	return out, nil
}

// recognizeLine runs the resize + session + CTC decode for one line at the
// given batch max wh_ratio, mirroring deepdoc TextRecognizer.resize_norm_img
// exactly: the tensor width is imgW = int(48 * max_wh_ratio) (floored at
// 320/48 for narrow batches); the content is resized to resized_w =
// min(ceil(48*ratio), imgW) and zero-padded on the right to imgW. Feeding the
// unpadded own-width (no floor, no pad) — the naive resize — changes
// recognition for narrow lines because the model sees a different width than
// deepdoc.
func recognizeLine(ctx context.Context, modelDir string, img *Image, maxWhRatio float64, chars []string) (OCRRecResult, error) {
	ratio := float64(img.W) / float64(img.H)
	imgW := int(math.Floor(recH * maxWhRatio))
	resizedW := int(math.Ceil(recH * ratio))
	if resizedW > imgW {
		resizedW = imgW
	}
	blob := ocrRecPreprocess(img, resizedW, imgW)
	// 0 → all cores, matching deepdoc's Python onnxruntime for bit-stable
	// parity (no contour extraction in the OCR-rec Run path).
	sess, release, err := getRecSession(filepath.Join(modelDir, "rec.onnx"), "x",
		[]int64{recMaxBatch, 3, recH, int64(imgW)}, "softmax_11.tmp_0", 0)
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

// ocrRecPreprocess builds the CHW float blob (/255, standardized) for a
// text-line image resized to (resizedW, recH) and zero-padded on the right to
// the full tensor width imgW. The session runs at imgW; padding mirrors
// deepdoc's resize_norm_img (padding_im[:, :, 0:resized_w] = resized_image).
func ocrRecPreprocess(img *Image, resizedW, imgW int) []float32 {
	bgr := img.ToBGR()
	w, h := img.W, img.H
	resized := bilinearResize(bgr, w, h, resizedW, recH)
	blob := make([]float32, 3*recH*imgW) // zero-filled (padded right)
	for y := 0; y < recH; y++ {
		for x := 0; x < resizedW; x++ {
			for c := 0; c < 3; c++ {
				v := float32(resized[(y*resizedW+x)*3+c]) / 255.0
				v = (v - 0.5) / 0.5
				blob[c*recH*imgW+y*imgW+x] = v
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
	// out layout: [recMaxBatch, seqLen, recVocab]; take batch 0. The sequence
	// length is dynamic (scales with the input width), so derive it from the
	// tensor length rather than a fixed constant.
	seqLen := len(out) / recVocab
	var text strings.Builder
	var probs []float32
	prev := -1
	var meanAcc float32
	for t := 0; t < seqLen; t++ {
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

// recSession runs rec.onnx, whose output sequence length is dynamic: it scales
// with the input width (≈ width/8), so a fixed-shape AdvancedSession cannot be
// pre-sized per width and even a width-matched session would still emit a
// varying seq length. Instead we use a DynamicAdvancedSession and pass a nil
// output on every Run: onnxruntime allocates the correctly-shaped output
// tensor, which we copy out before destroying it. The input tensor is
// fixed-shape per (model, width), so one recSession is reused per width.
type recSession struct {
	inName   string
	outName  string
	sess     *ort.DynamicAdvancedSession
	in       *ort.Tensor[float32]
	poisoned bool
}

func newRecSession(modelPath, inName string, inShape []int64, outName string, intraOpThreads int) (*recSession, error) {
	in := make([]float32, prod(inShape))
	inT, err := ort.NewTensor(ort.NewShape(inShape...), in)
	if err != nil {
		return nil, err
	}
	opts, err := ort.NewSessionOptions()
	if err != nil {
		inT.Destroy()
		return nil, err
	}
	// 0 → all cores (mirrors Python's onnxruntime default); OCR-rec does no
	// contour extraction in the Run path, so parallelism is safe and matches
	// deepdoc's reduction order for bit-stable parity.
	if err := opts.SetIntraOpNumThreads(intraOpThreads); err != nil {
		opts.Destroy()
		inT.Destroy()
		return nil, err
	}
	sess, err := ort.NewDynamicAdvancedSession(modelPath,
		[]string{inName}, []string{outName}, opts)
	if err != nil {
		opts.Destroy()
		inT.Destroy()
		return nil, err
	}
	return &recSession{inName: inName, outName: outName, sess: sess, in: inT}, nil
}

// Run copies input into the input tensor, executes with an auto-allocated
// (dynamic) output, and returns the output data. The allocated output tensor
// is destroyed before returning; out is a fresh copy the caller owns.
func (s *recSession) Run(ctx context.Context, input []float32) ([]float32, error) {
	if len(input) != len(s.in.GetData()) {
		return nil, fmt.Errorf("recSession %s: input len %d != tensor len %d",
			s.outName, len(input), len(s.in.GetData()))
	}
	opts, err := ort.NewRunOptions()
	if err != nil {
		return nil, err
	}
	defer opts.Destroy()
	// Cancel an in-flight Run when the context is done. done closes once Run
	// returns so the watcher exits even on the success path.
	done := make(chan struct{})
	defer close(done)
	go func() {
		select {
		case <-ctx.Done():
			_ = opts.Terminate()
		case <-done:
		}
	}()

	copy(s.in.GetData(), input)
	// nil output → onnxruntime allocates the actual-shaped tensor.
	outputs := []ort.Value{nil}
	if err := s.sess.RunWithOptions([]ort.Value{s.in}, outputs, opts); err != nil {
		if ctx.Err() != nil {
			s.poisoned = true
		}
		return nil, err
	}
	outVal := outputs[0]
	if outVal == nil {
		return nil, fmt.Errorf("recSession %s: nil output tensor", s.outName)
	}
	defer outVal.Destroy()
	t, ok := outVal.(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("recSession %s: unexpected output type %T", s.outName, outVal)
	}
	data := t.GetData()
	out := make([]float32, len(data))
	copy(out, data)
	return out, nil
}

// Destroy releases the dynamic session and input tensor.
func (s *recSession) Destroy() {
	if s.sess != nil {
		s.sess.Destroy()
	}
	if s.in != nil {
		s.in.Destroy()
	}
}

func (s *recSession) isPoisoned() bool { return s.poisoned }

func (s *recSession) markPoisoned() { s.poisoned = true }
