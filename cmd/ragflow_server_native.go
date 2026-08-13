//go:build native_det

package main

import (
	"os"
	"path/filepath"
	"strings"

	"go.uber.org/zap"

	"ragflow/internal/common"
	infnative "ragflow/internal/deepdoc/parser/pdf/inference/native"
)

// registerNativeDeepDoc wires the in-process (Go) DeepDoc backend as the local
// fallback used when no external DeepDoc HTTP service is configured
// (DEEPDOC_URL unset). The registration is best-effort: if ORT/lib/models are
// unavailable the backend simply reports not-serving.
//
// Fail-fast contract (P0): at least one DeepDoc backend must be available at
// startup — an external service (DEEPDOC_URL set) OR the local in-process
// backend (ORT + models present, built with -tags native_det). There is NO
// silent degradation to an empty analyzer: if DEEPDOC_URL is unset and the
// in-process backend is not serving, the server aborts. This mirrors the
// runtime resolver (deepDocAnalyzerFromEnv), which uses a configured client
// exclusively and fails loudly when it is unreachable, only falling back to the
// in-process backend when no client URL is set.
//
// This file is compiled only under the native_det build tag; the default build
// uses ragflow_server_nonative.go (a fail-fast no-op) so the unit-test path
// stays free of the onnxruntime dependency.
func registerNativeDeepDoc() {
	modelDir := resolveDeepDocModelDir()
	ortLib := resolveDeepDocORTLib()

	if err := infnative.Register(modelDir, ortLib); err != nil {
		common.Warn("in-process DeepDoc backend unavailable",
			zap.String("reason", err.Error()))
	}

	// At least one DeepDoc backend must be available: an external service
	// (DEEPDOC_URL) or the local in-process backend (ORT + models present).
	// Fail fast rather than silently parsing without layout/table/OCR.
	deepDocURL := strings.TrimSpace(common.GetEnv(common.EnvDeepDocURL))
	if deepDocURL == "" && !infnative.Serving() {
		common.Fatal("no DeepDoc backend configured: set DEEPDOC_URL to an external "+
			"DeepDoc service, or provide the local ORT runtime + models and build "+
			"with -tags native_det",
			zap.String("model_dir", modelDir),
			zap.String("ort_lib", ortLib))
	}

	if infnative.Serving() {
		common.Info("in-process DeepDoc backend registered (local fallback)",
			zap.String("model_dir", modelDir))
	} else {
		common.Info("using external DeepDoc service only (in-process backend not serving)",
			zap.String("deepdoc_url", deepDocURL))
	}
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

// resolveDeepDocORTLib returns the explicit DEEPDOC_ORT_LIB env, else the
// onnxruntime shared library extracted by ragflow_deps/download_deps.py,
// whose default location is derived from common.DeepDocORTVersion.
func resolveDeepDocORTLib() string {
	if v := strings.TrimSpace(common.GetEnv(common.EnvDeepDocORTLib)); v != "" {
		return v
	}
	if home, err := os.UserHomeDir(); err == nil {
		if cand := common.DefaultDeepDocORTLib(home); cand != "" {
			if _, err := os.Stat(cand); err == nil {
				return cand
			}
		}
	}
	return ""
}

// dirHasModels reports whether dir contains every required model file.
func dirHasModels(dir string) bool {
	return common.HasModelFiles(dir)
}
