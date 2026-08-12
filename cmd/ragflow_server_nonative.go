//go:build !native_det

package main

import (
	"strings"

	"go.uber.org/zap"

	"ragflow/internal/common"
)

// registerNativeDeepDoc is a no-op in the default build. The in-process
// (Go) DeepDoc backend is compiled only under the native_det build tag, so
// the unit-test path stays free of the onnxruntime dependency. In this build
// the only DeepDoc backend is the external HTTP service, which is required.
func registerNativeDeepDoc() {
	// In the default build there is no in-process backend, so an external
	// DeepDoc service (DEEPDOC_URL) is mandatory. Fail fast rather than
	// starting a server that would silently parse without layout/table/OCR
	// analysis.
	if strings.TrimSpace(common.GetEnv(common.EnvDeepDocURL)) == "" {
		common.Fatal("no DeepDoc backend configured: set DEEPDOC_URL to an external "+
			"DeepDoc service (the in-process Go backend requires building with "+
			"-tags native_det)",
			zap.String("env", common.EnvDeepDocURL))
	}
}
