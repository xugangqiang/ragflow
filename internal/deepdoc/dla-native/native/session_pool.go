package native

// session_pool.go — shared ONNX session pool for fixed-shape recognizers.
//
// RunDLA / RunTSR / RunOCRRec each loaded a fresh session on every call via
// NewSession + defer sess.Destroy(). A long document pays that setup cost once
// per region/page even though every call uses the same constant input/output
// shapes. This pool caches one session per model signature and hands it back to
// a sync.Pool between calls.
//
// Sessions are pooled, not shared concurrently: Session.Run copies the caller's
// input into the session's fixed-shape input tensor and then executes, so a
// single session must never be touched by two goroutines at once. getModelSession
// returns a session owned by the caller until release is called; release returns
// it to the pool for reuse. This keeps the Get/Run/Release window single-owner,
// which is what makes reuse safe under the page/region worker pools.

import (
	"strconv"
	"strings"
	"sync"
)

// modelSessionPools caches ONNX sessions for fixed-shape models (DLA/TSR/OCR-rec),
// keyed by the full (model path, IO names, IO shapes) tuple. Unlike the DB
// detector these models always run at a constant input size, so the tuple is
// constant per modelDir in practice.
var modelSessionPools sync.Map // modelSessKey -> *sync.Pool

type modelSessKey struct {
	modelPath, inName, outName string
	inShape, outShape          string
}

func modelSessKeyOf(modelPath, inName string, inShape []int64, outName string, outShape []int64) modelSessKey {
	return modelSessKey{
		modelPath: modelPath,
		inName:    inName,
		outName:   outName,
		inShape:   shapeKey(inShape),
		outShape:  shapeKey(outShape),
	}
}

func shapeKey(s []int64) string {
	parts := make([]string, len(s))
	for i, d := range s {
		parts[i] = strconv.FormatInt(d, 10)
	}
	return strings.Join(parts, ",")
}

// getModelSession returns a reusable session for the given model signature plus
// a release func. The caller must call release exactly once. On a pool miss a
// fresh session is created; creation errors are propagated and nothing is
// cached.
func getModelSession(modelPath, inName string, inShape []int64, outName string, outShape []int64) (*Session, func(), error) {
	key := modelSessKeyOf(modelPath, inName, inShape, outName, outShape)
	v, _ := modelSessionPools.LoadOrStore(key, &sync.Pool{})
	pool := v.(*sync.Pool)
	if got := pool.Get(); got != nil {
		if s, ok := got.(*Session); ok {
			return s, func() { pool.Put(s) }, nil
		}
	}
	s, err := NewSession(modelPath, inName, inShape, outName, outShape)
	if err != nil {
		return nil, nil, err
	}
	return s, func() { pool.Put(s) }, nil
}
