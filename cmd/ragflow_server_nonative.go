//go:build !cgo

package main

import (
	"go.uber.org/zap"

	"ragflow/internal/common"
)

// registerNativeDeepDoc is a no-op in the default build. The in-process
// (Go) DeepDoc backend is compiled only under the cgo build tag, so
// the unit-test path stays free of the onnxruntime dependency. Production
// requires the in-process backend, so the default (non-cgo) build
// cannot serve DeepDoc inference and must fail fast.
func registerNativeDeepDoc() {
	// Production serves DeepDoc entirely in-process; the external HTTP service
	// is no longer a backend. Without the cgo build tag there is no
	// serving backend at all, so fail fast rather than starting a server that
	// would silently parse without layout/table/OCR analysis.
	common.Fatal("no in-process DeepDoc backend: this build was compiled without "+
		"-tags cgo, which is required for the local DeepDoc (Go) backend. "+
		"Rebuild with -tags cgo and provide ORT + models.",
		zap.String("note", "the external Python DeepDoc HTTP service is no longer a backend"))
}
