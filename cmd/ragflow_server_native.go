//go:build cgo

package main

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"go.uber.org/zap"

	"ragflow/internal/common"
	infnative "ragflow/internal/deepdoc/parser/pdf/inference/native_analyzer"
)

// registerNativeDeepDoc wires the in-process (Go) DeepDoc backend as the local
// fallback used when no external DeepDoc HTTP service is configured
// (DEEPDOC_URL unset). The registration is best-effort: if ORT/lib/models are
// unavailable the backend simply reports not-serving.
//
// Fail-fast contract (P0): the in-process DeepDoc backend must be available at
// startup (ORT + models present, built with -tags cgo). There is NO
// silent degradation to an empty analyzer: if the in-process backend is not
// serving, the server aborts. This mirrors the runtime resolver
// (deepDocAnalyzerFromEnv), which uses the registered in-process factory
// exclusively and fails loudly when it is unavailable.
//
// This file is compiled only under the cgo build tag; the default build
// uses ragflow_server_nonative.go (a fail-fast no-op) so the unit-test path
// stays free of the onnxruntime dependency.
func registerNativeDeepDoc() {
	modelDir := resolveDeepDocModelDir()
	ortLib := resolveDeepDocORTLib()
	dropScore := resolveDeepDocDropScore()

	if err := infnative.Register(modelDir, ortLib, dropScore); err != nil {
		common.Warn("in-process DeepDoc backend unavailable",
			zap.String("reason", err.Error()))
	}

	// The in-process (Go) DeepDoc backend is the ONLY production backend; the
	// external Python HTTP service is no longer used. Fail fast rather than
	// silently parsing without layout/table/OCR if the local backend cannot
	// serve (ORT + models must be present when built with -tags cgo).
	if !infnative.Serving() {
		common.Fatal("no in-process DeepDoc backend serving: provide the local ORT "+
			"runtime + models and build with -tags cgo",
			zap.String("model_dir", modelDir),
			zap.String("ort_lib", "static (libonnxruntime.a via dlopen(NULL))"))
	}
	common.Info("in-process DeepDoc backend registered (production backend)",
		zap.String("model_dir", modelDir))
}

// resolveDeepDocModelDir picks the model directory: the explicit DEEPDOC_MODEL_DIR
// env, else the RAGFlow default (rag/res/deepdoc, mirroring deepdoc_server.py),
// else the snapshot fetched by ragflow_deps/download_deps.py. The first
// candidate that actually contains the required weights wins.
func resolveDeepDocModelDir() string {
	if v := strings.TrimSpace(common.GetEnv(common.EnvDeepDocModelDir)); v != "" {
		return v
	}
	wd, _ := os.Getwd()
	candidates := []string{
		filepath.Join(wd, "rag", "res", "deepdoc"),
		filepath.Join(wd, "huggingface.co", "InfiniFlow", "deepdoc"),
	}
	for _, c := range candidates {
		if dirHasModels(c) {
			return c
		}
	}
	// None verified; return the canonical default so any error message points
	// at the conventional location.
	return filepath.Join(wd, "rag", "res", "deepdoc")
}

// resolveDeepDocORTLib always returns "" because the in-process DeepDoc
// backend links ONNX Runtime statically (libonnxruntime.a, resolved at
// runtime via dlopen(NULL) from the running binary — see the onnxruntime_go
// fork under internal/deepdoc/native/third_party/onnxruntime_go). There is no
// dynamic .so deployment and therefore no path to resolve; native.InitORT("")
// selects the static path.
func resolveDeepDocORTLib() string {
	return ""
}

// resolveDeepDocDropScore returns the explicit DEEPDOC_DROP_SCORE env, else
// the in-process backend's default (infnative.DefaultDropScore, which mirrors
// the Python inference service's Recognizer.drop_score).
func resolveDeepDocDropScore() float64 {
	if v := strings.TrimSpace(common.GetEnv(common.EnvDeepDocDropScore)); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
		common.Warn("invalid DEEPDOC_DROP_SCORE, using default",
			zap.String("value", v), zap.Float64("default", infnative.DefaultDropScore))
	}
	return infnative.DefaultDropScore
}

// dirHasModels reports whether dir contains every required model file.
func dirHasModels(dir string) bool {
	return common.HasModelFiles(dir)
}
