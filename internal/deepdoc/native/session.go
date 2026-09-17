//go:build cgo

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
	"context"
	"fmt"
	"log"
	"sync"

	ort "github.com/infiniflow/onnxruntime_go"
)

// intraOpThreads is the intra-op thread count every session is opened with.
//
// ONNX Runtime gives each session its own intra-op thread pool (the C API
// never switches a session onto a shared/global pool), so the threads DeepDoc
// inference occupies in this process are intraOpThreads × the number of
// concurrently running sessions. Pinning it to 1 keeps every Run to a single
// thread, which is what makes the process ceiling a plain concurrency budget:
// the capacity registered in inference_limit.go bounds how many Runs may be in
// flight, and each of them costs exactly one thread.
const intraOpThreads = 1

var (
	ortOnce    sync.Once
	ortInitErr error
	// ortReady is true once InitializeEnvironment has succeeded. It lets
	// callers decide whether the in-process backend can serve without
	// triggering a panic from a session Run against an uninitialized
	// environment.
	ortReady bool
)

// InitORT initializes the process-global ONNX Runtime environment. Safe to
// call multiple times; only the first takes effect. Call it once at process
// start (the CLI does this from main).
//
// The in-process DeepDoc backend links ONNX Runtime statically:
// libonnxruntime.a is linked in with no --whole-archive, so GNU ld drops the
// kernels and execution providers that nothing references, and only
// OrtGetApiBase is exported, via --dynamic-list (see build.sh:
// ONNXRUNTIME_STATIC_PREFIX). The org
// onnxruntime_go binding (github.com/infiniflow/onnxruntime_go) resolves
// OrtGetApiBase from the running binary itself via dlopen(NULL) (the
// process-global symbol table), so no external libonnxruntime.so is needed and
// there is no dynamic .so deployment. A main executable CANNOT be dlopen'd by
// its own file path (glibc refuses), which is exactly why the binding uses the
// NULL handle instead of a path. InitORT therefore takes no library path;
// ragflow never calls SetSharedLibraryPath, so the binding resolves ORT from the
// running binary via dlopen(NULL).
func InitORT() error {
	ortOnce.Do(func() {
		ortInitErr = ort.InitializeEnvironment()
		if ortInitErr == nil {
			ortReady = true
		}
	})
	return ortInitErr
}

// Initialized reports whether ONNX Runtime's process-global environment has
// been successfully initialized. The in-process DeepDoc backend uses this to
// decide whether it can serve, degrading to an empty analyzer otherwise.
func Initialized() bool { return ortReady }

// session loads one ONNX model and runs single-input/single-output inference.
type session struct {
	inName  string
	outName string
	outSize int64
	sess    *ort.AdvancedSession
	in      *ort.Tensor[float32]
	out     *ort.Tensor[float32]
	// poisoned is set when a Run is cancelled/terminated via context. ONNX
	// Runtime does not guarantee a session is reusable after a forced
	// termination, so the pool must Destroy rather than re-Put it.
	poisoned bool
}

// NewSession opens modelPath. inShape/outShape describe the fixed tensor
// dimensions; outSize is the total element count of the output tensor. The
// session runs intraOpThreads intra-op threads (see the constant in session.go):
// one thread per Run, with the process-wide ceiling owned by the inference budget
// the process owner registers (see inference_limit.go). Weight sharing is applied
// transparently: the model's constant initializers are extracted once per
// modelPath and injected into the session options, so every session of the
// same model shares a single copy of the weight buffers. InitORT must have been
// called first.
func NewSession(modelPath, inName string, inShape []int64, outName string, outShape []int64) (*session, error) {
	weights, werr := sharedWeights(modelPath, inName, inShape, outName)
	if werr != nil {
		// Degrade gracefully: a model still loads and runs correctly without
		// sharing; it just deserializes its own weight copy.
		log.Printf("deepdoc/native: weight sharing unavailable for %s: %v",
			modelPath, werr)
		weights = nil
	}
	return newRawSession(modelPath, inName, inShape, outName, outShape, weights)
}

// newRawSession opens modelPath with no weight sharing unless weights != nil,
// in which case each shared initializer is injected into the session options
// before the model is loaded. It is the single point where the ORT session is
// actually created, so both NewSession and the one-shot extraction source used
// by sharedWeights funnel through it.
func newRawSession(modelPath, inName string, inShape []int64, outName string, outShape []int64, weights *weightSet) (*session, error) {
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
	// One intra-op thread per session: the session's Runs then cost one thread
	// each, so the process-wide inference ceiling is exactly the number of
	// concurrent Runs the caller admits (see the intraOpThreads constant).
	if err := opts.SetIntraOpNumThreads(intraOpThreads); err != nil {
		opts.Destroy()
		inT.Destroy()
		outT.Destroy()
		return nil, err
	}
	// Disable the BFC memory arena for this session. The arena pre-reserves a
	// native block per session and never shrinks it, so every pooled (often
	// idle) session hoards one. Across the rec/det/DLA/TSR pools (~220 live
	// sessions while parsing a large PDF) this dominated the ~14 GB of native
	// memory seen at the ~20 GB OOM peak. With the arena off, idle sessions keep
	// only their weights; activation tensors are allocated directly and freed
	// after each Run.
	if err := opts.SetCpuMemArena(false); err != nil {
		opts.Destroy()
		inT.Destroy()
		outT.Destroy()
		return nil, err
	}
	if weights != nil {
		for i, v := range weights.vals {
			if err := opts.AddInitializer(weights.names[i], v); err != nil {
				opts.Destroy()
				inT.Destroy()
				outT.Destroy()
				return nil, fmt.Errorf("inject shared initializer %q: %w",
					weights.names[i], err)
			}
		}
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
	return &session{
		inName: inName, outName: outName,
		outSize: prod(outShape),
		sess:    sess, in: inT, out: outT,
	}, nil
}

// weightSet holds the shared, user-owned constant-initializer buffers extracted
// once per model. Every session opened for the same modelPath injects these
// buffers (via SessionOptions.AddInitializer) so the weights live in memory a
// single time instead of being deserialized into every session. The buffers
// are owned by this package and must outlive every consumer session; they are
// cached for the process lifetime (freed only at process exit, alongside the
// process-global ORT environment).
type weightSet struct {
	names []string
	vals  []*ort.SharedInitializer
}

var (
	weightMu    sync.Mutex
	weightCache = map[string]*weightSet{}
)

// sharedWeights returns the (lazily extracted and cached) shared initializers
// for modelPath. The source session used for extraction is destroyed before
// returning; the extracted buffers are independent user-owned copies that
// outlive it, so the source session's own weight copy can be released.
func sharedWeights(modelPath, inName string, inShape []int64, outName string) (*weightSet, error) {
	weightMu.Lock()
	defer weightMu.Unlock()
	if ws, ok := weightCache[modelPath]; ok {
		return ws, nil
	}
	// Load a source session purely to read its constant initializers. The
	// output shape is irrelevant for extraction, so a dummy is used.
	src, err := newRawSession(modelPath, inName, inShape, outName, []int64{1}, nil)
	if err != nil {
		return nil, fmt.Errorf("load source session for extraction: %w", err)
	}
	ws := &weightSet{}
	count, err := src.sess.GetInitializerCount()
	if err != nil {
		src.Destroy()
		return nil, fmt.Errorf("GetInitializerCount: %w", err)
	}
	if count > 0 {
		ws.names = make([]string, 0, count)
		ws.vals = make([]*ort.SharedInitializer, 0, count)
		for i := 0; i < count; i++ {
			name, e := src.sess.GetInitializerName(i)
			if e != nil {
				err = fmt.Errorf("GetInitializerName(%d): %w", i, e)
				break
			}
			val, e := src.sess.GetInitializer(name)
			if e != nil {
				err = fmt.Errorf("GetInitializer(%q): %w", name, e)
				break
			}
			ws.names = append(ws.names, name)
			ws.vals = append(ws.vals, val)
		}
		if err != nil {
			for _, v := range ws.vals {
				v.Destroy()
			}
			src.Destroy()
			return nil, err
		}
	}
	src.Destroy()
	weightCache[modelPath] = ws
	return ws, nil
}

// Run copies input into the input tensor, executes, and returns the output
// tensor contents. ctx bounds the inference: if it is cancelled while Run is
// in flight, the underlying ONNX Runtime call is terminated via RunOptions.
// A terminated session is left in an indeterminate state, so it is marked
// poisoned and the pool destroys it instead of reusing it.
func (s *session) Run(ctx context.Context, input []float32) ([]float32, error) {
	if len(input) != len(s.in.GetData()) {
		return nil, fmt.Errorf("session %s: input len %d != tensor len %d",
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
	if err := s.sess.RunWithOptions(opts); err != nil {
		if ctx.Err() != nil {
			s.poisoned = true
		}
		return nil, err
	}
	out := make([]float32, s.outSize)
	copy(out, s.out.GetData())
	return out, nil
}

// Destroy releases the tensors and advanced-session handle. It does NOT touch
// the process-global environment.
func (s *session) Destroy() {
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

func (s *session) isPoisoned() bool { return s.poisoned }

// markPoisoned records that a Run was force-terminated; a poisoned session is
// not safe to reuse, so the pool Destroys it on release.
func (s *session) markPoisoned() { s.poisoned = true }

func prod(shape []int64) int64 {
	p := int64(1)
	for _, d := range shape {
		p *= d
	}
	return p
}
