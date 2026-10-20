package service

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"ragflow/internal/dao"
	"ragflow/internal/dao/transaction"
	"ragflow/internal/entity"
	pipelinepkg "ragflow/internal/ingestion/pipeline"

	"gorm.io/gorm"
)

// ErrPipelineVersionMissing indicates the requested pipeline_version_id does
// not exist or does not belong to the bound canvas. Callers treat this as a
// Missing state: the file cannot be parsed until a valid version is (re)selected.
var ErrPipelineVersionMissing = errors.New("pipeline version missing or not owned by canvas")

// loadCanvasDSLJSON returns the DSL JSON for a custom canvas pipeline.
//
// For dataflow canvases (canvas_category = dataflow_canvas) the read is versioned
// per design §6.1: an explicit versionID resolves that exact version; otherwise
// the latest version is used, and a canvas with no version rows surfaces
// ErrPipelineVersionMissing (no fallback to the legacy draft column). Agent
// canvases are out of the multi-version scope (design §1) and keep reading the
// live draft column directly. It is a package-level function so both document
// and knowledge-base updates reuse it.
func loadCanvasDSLJSON(ctx context.Context, canvasID string, versionID *string) ([]byte, error) {
	if strings.TrimSpace(canvasID) == "" {
		return nil, fmt.Errorf("empty canvas id")
	}
	canvas, err := dao.NewUserCanvasDAO().GetByID(ctx, dao.DB, canvasID)
	if err != nil {
		if errors.Is(err, dao.ErrUserCanvasNotFound) {
			return nil, fmt.Errorf("canvas %s not found", canvasID)
		}
		return nil, fmt.Errorf("load canvas %s: %w", canvasID, err)
	}
	if canvas.CanvasCategory != "dataflow_canvas" {
		// Agent canvases: out of scope for multi-version; read the live draft.
		if len(canvas.DSL) == 0 {
			return nil, fmt.Errorf("canvas %s has no DSL", canvasID)
		}
		return marshalCanvasDSL(canvasID, canvas.DSL)
	}
	if versionID != nil && strings.TrimSpace(*versionID) != "" {
		return loadVersionedCanvasDSLJSON(ctx, canvasID, strings.TrimSpace(*versionID))
	}
	// No explicit version: use the latest version row (design §6.1).
	if v, err := dao.NewUserCanvasVersionDAO().GetLatest(ctx, dao.DB, canvasID); err == nil {
		if len(v.DSL) > 0 {
			return marshalCanvasDSL(canvasID, v.DSL)
		}
	}
	return nil, fmt.Errorf("pipeline version for canvas %s not found: %w", canvasID, ErrPipelineVersionMissing)
}

// loadVersionedCanvasDSLJSON resolves a specific canvas version and validates
// that it belongs to the bound canvas. A missing or mis-owned version maps to
// ErrPipelineVersionMissing so callers can surface a Missing (unparseable) state.
func loadVersionedCanvasDSLJSON(ctx context.Context, canvasID, versionID string) ([]byte, error) {
	v, err := dao.NewUserCanvasVersionDAO().GetByID(ctx, dao.DB, versionID)
	if err != nil {
		if errors.Is(err, dao.ErrUserCanvasVersionNotFound) {
			return nil, fmt.Errorf("pipeline version %s not found: %w", versionID, ErrPipelineVersionMissing)
		}
		return nil, fmt.Errorf("load pipeline version %s: %w", versionID, err)
	}
	if v.UserCanvasID != canvasID {
		return nil, fmt.Errorf("pipeline version %s does not belong to canvas %s: %w", versionID, canvasID, ErrPipelineVersionMissing)
	}
	if len(v.DSL) == 0 {
		return nil, fmt.Errorf("pipeline version %s has no DSL: %w", versionID, ErrPipelineVersionMissing)
	}
	return marshalCanvasDSL(canvasID, v.DSL)
}

func marshalCanvasDSL(canvasID string, dsl entity.JSONMap) ([]byte, error) {
	raw, err := json.Marshal(dsl)
	if err != nil {
		return nil, fmt.Errorf("marshal canvas %s DSL: %w", canvasID, err)
	}
	return raw, nil
}

// LoadPipelineDSL loads the DSL JSON for a pipeline identified by parserID
// (built-in) or pipelineID (custom canvas). When both are provided, isPipeline
// selects which one to use. versionID selects a specific canvas version; when nil
// the latest version (or legacy canvas dsl) is used.
func LoadPipelineDSL(ctx context.Context, isPipeline bool, parserID string, pipelineID *string, versionID *string) ([]byte, error) {
	if isPipeline {
		return loadCanvasDSLJSON(ctx, strings.TrimSpace(*pipelineID), versionID)
	}
	registry, err := pipelinepkg.DefaultRegistry()
	if err != nil {
		return nil, fmt.Errorf("builtin pipeline registry: %w", err)
	}
	if !registry.IsValid(parserID) {
		return nil, fmt.Errorf("unknown builtin parser_id: %s", parserID)
	}
	dslStr, err := pipelinepkg.LoadBuiltinDSL(parserID)
	if err != nil {
		return nil, fmt.Errorf("load builtin DSL for %q: %w", parserID, err)
	}
	return []byte(dslStr), nil
}

// ResolveComponentParamsDefaults loads the DSL for the target pipeline and
// returns the component params defaults as an entity.JSONMap {cpnID: {param: value}}.
// For builtin templates the DSL is loaded from the embedded registry; for custom
// canvas pipelines it is loaded from the canvas version (or legacy canvas row).
func ResolveComponentParamsDefaults(ctx context.Context, parserID string, pipelineID *string, versionID *string) (entity.JSONMap, error) {
	isPipeline := pipelineID != nil && strings.TrimSpace(*pipelineID) != ""
	var cp map[string]map[string]any
	var err error
	if isPipeline {
		dslJSON, lerr := loadCanvasDSLJSON(ctx, strings.TrimSpace(*pipelineID), versionID)
		if lerr != nil {
			return nil, fmt.Errorf("load canvas DSL: %w", lerr)
		}
		cp, err = pipelinepkg.ComponentParamsDefaults(dslJSON)
	} else {
		registry, regErr := pipelinepkg.DefaultRegistry()
		if regErr != nil {
			return nil, fmt.Errorf("builtin registry: %w", regErr)
		}
		if !registry.IsValid(parserID) {
			return nil, fmt.Errorf("unknown builtin parser_id: %q", parserID)
		}
		dslStr, dslErr := pipelinepkg.LoadBuiltinDSL(parserID)
		if dslErr != nil {
			return nil, fmt.Errorf("load builtin DSL: %w", dslErr)
		}
		cp, err = pipelinepkg.ComponentParamsDefaults([]byte(dslStr))
	}
	if err != nil {
		return nil, err
	}
	out := make(entity.JSONMap, len(cp))
	for k, v := range cp {
		out[k] = v
	}
	return out, nil
}

// ValidateDatasetEmbeddingModels checks that all knowledge bases in the list
// either have an embedding model or none do, and that they all resolve to the
// same base embedding model name (e.g. "BAAI/bge-m3"). Embedding references
// are resolved through tenant_model first, so datasets storing a raw
// tenant_model id and datasets storing a legacy "model@instance@provider"
// composite compare equal when they point at the same model.
func ValidateDatasetEmbeddingModels(ctx context.Context, db *gorm.DB, kbs []*entity.Knowledgebase) error {
	embdNames := make(map[string]struct{})
	hasEmbd := false
	noEmbd := false
	names := transaction.NewEmbeddingBaseNameResolver(dao.NewKnowledgebaseDAO().EmbeddingBaseName)
	for _, kb := range kbs {
		if kb.EmbdID != "" {
			hasEmbd = true
			embdNames[names.Resolve(ctx, db, kb)] = struct{}{}
		} else {
			noEmbd = true
		}
	}
	if hasEmbd && noEmbd {
		return fmt.Errorf("cannot search across datasets where some have embedding models and others do not")
	}
	if len(embdNames) > 1 {
		return fmt.Errorf("datasets use different embedding models: %v", getEmbdIDs(kbs))
	}
	return nil
}

func getEmbdIDs(kbs []*entity.Knowledgebase) []string {
	ids := make([]string, len(kbs))
	for i, kb := range kbs {
		ids[i] = kb.EmbdID
	}
	return ids
}
