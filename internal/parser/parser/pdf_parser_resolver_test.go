package parser

import (
	"context"
	"image"
	"net/http"
	"net/http/httptest"
	"testing"

	deepdoctype "ragflow/internal/deepdoc/parser/type"
)

// fakeAnalyzer is a minimal DocAnalyzer used to assert the in-process fallback
// branch of resolveDocAnalyzer. It never talks to ORT or the network.
type fakeAnalyzer struct{}

func (fakeAnalyzer) DLA(context.Context, image.Image) ([]deepdoctype.DLARegion, error) {
	return nil, nil
}
func (fakeAnalyzer) TSR(context.Context, image.Image) ([]deepdoctype.TSRCell, error) {
	return nil, nil
}
func (fakeAnalyzer) OCRDetect(context.Context, image.Image) ([]deepdoctype.OCRBox, error) {
	return nil, nil
}
func (fakeAnalyzer) OCRRecognize(context.Context, image.Image) ([]deepdoctype.OCRText, error) {
	return nil, nil
}
func (fakeAnalyzer) Health() bool { return true }

// TestResolveDocAnalyzer pins the P0 backend policy without mutating the
// process-global nativeAnalyzerFactory, so it cannot pollute other tests.
func TestResolveDocAnalyzer(t *testing.T) {
	// Neither configured → error. This is the fail-loudly branch that was
	// previously only reachable by clearing the global; now it is a pure
	// function of its inputs.
	if _, err := resolveDocAnalyzer("", nil); err == nil {
		t.Fatal("resolveDocAnalyzer(\"\", nil) should error: no backend available")
	}

	// In-process factory fallback when no external service is configured.
	a, err := resolveDocAnalyzer("", func() (deepdoctype.DocAnalyzer, bool) {
		return fakeAnalyzer{}, true
	})
	if err != nil {
		t.Fatalf("in-process fallback should succeed: %v", err)
	}
	if a == nil {
		t.Fatal("expected a non-nil analyzer from the fallback")
	}

	// A configured DEEPDOC_URL is used exclusively and fails loudly (no
	// silent fallback) when the service is unreachable.
	down := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer down.Close()
	if _, err := resolveDocAnalyzer(down.URL, nil); err == nil {
		t.Fatal("unreachable DEEPDOC_URL should fail loudly, not fall back")
	}

	// A configured, healthy DEEPDOC_URL is used (no fallback needed).
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer up.Close()
	if _, err := resolveDocAnalyzer(up.URL, nil); err != nil {
		t.Fatalf("healthy DEEPDOC_URL should be used: %v", err)
	}

	// A healthy client still wins over the in-process factory: the factory
	// must NOT be consulted when a URL is set (config-driven isolation).
	called := false
	if _, err := resolveDocAnalyzer(up.URL, func() (deepdoctype.DocAnalyzer, bool) {
		called = true
		return fakeAnalyzer{}, true
	}); err != nil {
		t.Fatalf("healthy DEEPDOC_URL should be used: %v", err)
	}
	if called {
		t.Fatal("in-process factory must not be consulted when DEEPDOC_URL is set")
	}
}
