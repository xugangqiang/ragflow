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

// requiredModelFiles are the weights the in-process backend needs. Used to
// validate a candidate model directory before committing to it as the default.
var requiredModelFiles = []string{"det.onnx", "layout.onnx", "tsr.onnx", "rec.onnx", "ocr.res"}

// registerNativeDeepDoc wires the in-process (Go) DeepDoc backend as the local
// fallback used when no external DeepDoc HTTP service is configured
// (DEEPDOC_URL unset). It is a best-effort registration: if ORT/lib/models are
// unavailable the backend reports not-serving and the parser degrades to the
// empty analyzer, so a deployment without the native deps still starts cleanly.
//
// This file is compiled only under the native_det build tag; the default build
// uses ragflow_server_nonative.go (a no-op) so the unit-test path stays free of
// the onnxruntime dependency.
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
// onnxruntime shared library extracted by ragflow_deps/download_deps.py.
func resolveDeepDocORTLib() string {
	if v := strings.TrimSpace(common.GetEnv(common.EnvDeepDocORTLib)); v != "" {
		return v
	}
	if home, err := os.UserHomeDir(); err == nil {
		cand := filepath.Join(home, "ragflow-native-libs", "onnxruntime",
			"onnxruntime-linux-x64-1.23.2", "lib", "libonnxruntime.so.1.23.2")
		if _, err := os.Stat(cand); err == nil {
			return cand
		}
	}
	return ""
}

// dirHasModels reports whether dir contains every required model file.
func dirHasModels(dir string) bool {
	for _, f := range requiredModelFiles {
		if _, err := os.Stat(filepath.Join(dir, f)); err != nil {
			return false
		}
	}
	return true
}
