package native

// session_pool.go — shared ONNX session pool for all recognizers.
//
// RunDLA / RunTSR / RunOCRRec (fixed-shape) and RunDet (variable-shape DB
// detector) all load an ONNX session per call. A long document pays that setup
// cost per region/page even though every call uses the same shapes within a
// model. This pool caches one session per (model signature) tuple and hands it
// back between calls.
//
// Sessions are pooled, not shared concurrently: session.Run copies the caller's
// input into the session's fixed-shape input tensor and then executes, so a
// single session must never be touched by two goroutines at once. Get returns a
// session owned by the caller until release is called; release returns it to
// the pool for reuse. This keeps the Get/Run/Release window single-owner, which
// is what makes reuse safe under the page/region worker pools.
//
// The pool is generic over the key type so both the fixed-shape recognizers
// (modelSessKey) and the variable-shape detector (detSessKey) share one
// implementation. Two bounds apply:
//   - maxKeys caps the number of distinct key-pools; when exceeded the
//     least-recently-used key-pool is evicted and its idle sessions Destroyed
//     (bounds memory for the variable-shape detector, which can see many
//     distinct page sizes).
//   - maxFree caps idle sessions retained per key-pool; extras are Destroyed on
//     release instead of pooled. A maxFree / maxKeys of 0 means unbounded — the
//     degenerate case for the fixed-shape recognizers, whose key set is tiny.

import (
	"strconv"
	"strings"
	"sync"
)

// sessionKeyPool holds reusable sessions for one key.
type sessionKeyPool struct {
	mu   sync.Mutex
	live bool // false once evicted; checked-out sessions self-Destroy on release
	free []*session
}

// sessionPool is a generic reusable ONNX session pool keyed by K.
type sessionPool[K comparable] struct {
	mu      sync.Mutex
	pools   map[K]*sessionKeyPool
	lru     []K // front = least-recently-used
	maxKeys int
	maxFree int
}

func newSessionPool[K comparable](maxKeys, maxFree int) *sessionPool[K] {
	return &sessionPool[K]{
		pools:   make(map[K]*sessionKeyPool),
		maxKeys: maxKeys,
		maxFree: maxFree,
	}
}

// Get returns a reusable session for key, constructing one with newFn on a pool
// miss, plus a release func. The caller must call release exactly once. A
// poisoned session is Destroyed on release rather than pooled (ORT does not
// guarantee reuse safety after a forced termination).
func (p *sessionPool[K]) Get(key K, newFn func() (*session, error)) (*session, func(), error) {
	p.mu.Lock()
	kp := p.pools[key]
	if kp == nil {
		if p.maxKeys > 0 && len(p.pools) >= p.maxKeys {
			p.evictLRU()
		}
		kp = &sessionKeyPool{live: true}
		p.pools[key] = kp
		p.lru = append(p.lru, key)
	} else {
		p.touchLRU(key)
	}
	p.mu.Unlock()

	kp.mu.Lock()
	var s *session
	if n := len(kp.free); n > 0 {
		s = kp.free[n-1]
		kp.free = kp.free[:n-1]
	}
	kp.mu.Unlock()

	if s == nil {
		var err error
		s, err = newFn()
		if err != nil {
			return nil, nil, err
		}
	}

	release := func() {
		if s.poisoned {
			s.Destroy()
			return
		}
		kp.mu.Lock()
		if kp.live && (p.maxFree <= 0 || len(kp.free) < p.maxFree) {
			kp.free = append(kp.free, s)
			kp.mu.Unlock()
		} else {
			kp.mu.Unlock()
			s.Destroy()
		}
	}
	return s, release, nil
}

// KeyCount returns the number of distinct key-pools currently live. Used by
// tests that assert the pool set is bounded.
func (p *sessionPool[K]) KeyCount() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.pools)
}

// touchLRU moves key to the most-recently-used end. Caller holds p.mu.
func (p *sessionPool[K]) touchLRU(key K) {
	for i, k := range p.lru {
		if k == key {
			p.lru = append(p.lru[:i], p.lru[i+1:]...)
			p.lru = append(p.lru, k)
			return
		}
	}
}

// evictLRU drops the least-recently-used key-pool and destroys its idle
// sessions. Caller holds p.mu.
func (p *sessionPool[K]) evictLRU() {
	if len(p.lru) == 0 {
		return
	}
	evict := p.lru[0]
	p.lru = p.lru[1:]
	kp := p.pools[evict]
	delete(p.pools, evict)
	if kp == nil {
		return
	}
	kp.mu.Lock()
	kp.live = false
	for _, s := range kp.free {
		s.Destroy()
	}
	kp.free = nil
	kp.mu.Unlock()
}

// ---- fixed-shape recognizer pool (DLA / TSR / OCR-rec) ----

// modelSessKey is the pool key for the fixed-shape models. Unlike the DB
// detector these models always run at a constant input size, so the tuple is
// constant per modelDir in practice.
type modelSessKey struct {
	modelPath, inName, outName string
	inShape, outShape          string
	intraOpThreads             int
}

func modelSessKeyOf(modelPath, inName string, inShape []int64, outName string, outShape []int64, intraOpThreads int) modelSessKey {
	return modelSessKey{
		modelPath:      modelPath,
		inName:         inName,
		outName:        outName,
		inShape:        shapeKey(inShape),
		outShape:       shapeKey(outShape),
		intraOpThreads: intraOpThreads,
	}
}

func shapeKey(s []int64) string {
	parts := make([]string, len(s))
	for i, d := range s {
		parts[i] = strconv.FormatInt(d, 10)
	}
	return strings.Join(parts, ",")
}

// modelSessions is unbounded (maxKeys/maxFree = 0): the fixed-shape key set is
// tiny, so the degenerate no-eviction case is correct here.
var modelSessions = newSessionPool[modelSessKey](0, 0)

// getModelSession returns a reusable session for the given model signature plus
// a release func. The caller must call release exactly once.
func getModelSession(modelPath, inName string, inShape []int64, outName string, outShape []int64, intraOpThreads int) (*session, func(), error) {
	key := modelSessKeyOf(modelPath, inName, inShape, outName, outShape, intraOpThreads)
	return modelSessions.Get(key, func() (*session, error) {
		return NewSession(modelPath, inName, inShape, outName, outShape, intraOpThreads)
	})
}
