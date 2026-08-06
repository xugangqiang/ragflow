package native

// session.go — thin wrapper around onnxruntime_go.
//
// Hides all onnxruntime-go specifics from the recognizers so each task module
// only deals with float32 tensors. One model input, one model output (every
// DeepDoc ONNX we port fits this shape). CPU-only by design.
//
// The ONNX Runtime environment is process-global: InitORT sets the shared
// library and initializes it exactly once. Sessions only own their tensors and
// the advanced-session handle, so running several tasks in one process (or one
// task per CLI invocation) never double-initializes or prematurely tears down
// the shared environment.

import (
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	ortOnce    sync.Once
	ortInitErr error
)

// InitORT points ONNX Runtime at its shared library and initializes the
// global environment. Safe to call multiple times; only the first takes
// effect. Call it once at process start (the CLI does this from main).
func InitORT(libPath string) error {
	ortOnce.Do(func() {
		ort.SetSharedLibraryPath(libPath)
		ortInitErr = ort.InitializeEnvironment()
	})
	return ortInitErr
}

// Session loads one ONNX model and runs single-input/single-output inference.
type Session struct {
	inName  string
	outName string
	outSize int64
	sess    *ort.AdvancedSession
	in      *ort.Tensor[float32]
	out     *ort.Tensor[float32]
}

// NewSession opens modelPath. inShape/outShape describe the fixed tensor
// dimensions; outSize is the total element count of the output tensor.
// InitORT must have been called first.
func NewSession(modelPath, inName string, inShape []int64, outName string, outShape []int64) (*Session, error) {
	in := make([]float32, prod(inShape))
	out := make([]float32, prod(outShape))
	inT, err := ort.NewTensor(ort.NewShape(inShape...), in)
	if err != nil {
		return nil, err
	}
	outT, err := ort.NewTensor(ort.NewShape(outShape...), out)
	if err != nil {
		inT.Destroy()
		return nil, err
	}
	opts, err := ort.NewSessionOptions()
	if err != nil {
		inT.Destroy()
		outT.Destroy()
		return nil, err
	}
	// Run ORT single-threaded. A multi-threaded ORT run leaves worker threads
	// settling while the caller (still nested inside Session.Run) invokes
	// OpenCV's findContours, whose parallel_for_ then under-runs and returns
	// contracted contours. Single-threaded inference avoids the conflict and
	// is fully deterministic.
	if err := opts.SetIntraOpNumThreads(1); err != nil {
		opts.Destroy()
		inT.Destroy()
		outT.Destroy()
		return nil, err
	}
	sess, err := ort.NewAdvancedSession(modelPath,
		[]string{inName}, []string{outName},
		[]ort.Value{inT}, []ort.Value{outT}, opts)
	if err != nil {
		opts.Destroy()
		inT.Destroy()
		outT.Destroy()
		return nil, err
	}
	return &Session{
		inName: inName, outName: outName,
		outSize: prod(outShape),
		sess:    sess, in: inT, out: outT,
	}, nil
}

// Run copies input into the input tensor, executes, and returns the output
// tensor contents.
func (s *Session) Run(input []float32) ([]float32, error) {
	copy(s.in.GetData(), input)
	if err := s.sess.Run(); err != nil {
		return nil, err
	}
	out := make([]float32, s.outSize)
	copy(out, s.out.GetData())
	return out, nil
}

// Destroy releases the tensors and advanced-session handle. It does NOT touch
// the process-global environment.
func (s *Session) Destroy() {
	if s.sess != nil {
		s.sess.Destroy()
	}
	if s.in != nil {
		s.in.Destroy()
	}
	if s.out != nil {
		s.out.Destroy()
	}
}

func prod(shape []int64) int64 {
	p := int64(1)
	for _, d := range shape {
		p *= d
	}
	return p
}
