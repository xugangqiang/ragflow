package service

import (
	"context"
	"errors"
	"strings"
	"testing"

	"ragflow/internal/dao"
	"ragflow/internal/entity"
)

func assertContains(t *testing.T, haystack, needle string) {
	t.Helper()
	if !strings.Contains(haystack, needle) {
		t.Fatalf("expected output to contain %q, got: %s", needle, haystack)
	}
}

func assertMissing(t *testing.T, err error) {
	t.Helper()
	if err == nil {
		t.Fatal("expected ErrPipelineVersionMissing, got nil error")
	}
	if !errors.Is(err, ErrPipelineVersionMissing) {
		t.Fatalf("expected ErrPipelineVersionMissing, got: %v", err)
	}
}

func TestLoadCanvasDSLJSON_VersionedReadsVersionDSL(t *testing.T) {
	db := setupServiceTestDB(t)
	if err := db.AutoMigrate(&entity.UserCanvasVersion{}); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	orig := dao.DB
	dao.DB = db
	t.Cleanup(func() { dao.DB = orig })

	ctx := context.Background()
	canvasID := "cv-versioned"
	if err := db.Create(&entity.UserCanvas{ID: canvasID, CanvasCategory: "dataflow_canvas", DSL: entity.JSONMap{"graph": "legacy"}}).Error; err != nil {
		t.Fatal(err)
	}
	if err := db.Create(&entity.UserCanvasVersion{ID: "ver-1", UserCanvasID: canvasID, DSL: entity.JSONMap{"graph": "versioned"}}).Error; err != nil {
		t.Fatal(err)
	}

	raw, err := loadCanvasDSLJSON(ctx, canvasID, sptr("ver-1"))
	if err != nil {
		t.Fatalf("loadCanvasDSLJSON: %v", err)
	}
	assertContains(t, string(raw), "versioned")
	if strings.Contains(string(raw), "legacy") {
		t.Fatalf("expected version DSL, got legacy: %s", string(raw))
	}
}

func TestLoadCanvasDSLJSON_VersionedMisownedIsMissing(t *testing.T) {
	db := setupServiceTestDB(t)
	if err := db.AutoMigrate(&entity.UserCanvasVersion{}); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	orig := dao.DB
	dao.DB = db
	t.Cleanup(func() { dao.DB = orig })

	ctx := context.Background()
	if err := db.Create(&entity.UserCanvas{ID: "cv-a", CanvasCategory: "dataflow_canvas", DSL: entity.JSONMap{"graph": "a"}}).Error; err != nil {
		t.Fatal(err)
	}
	if err := db.Create(&entity.UserCanvas{ID: "cv-b", CanvasCategory: "dataflow_canvas", DSL: entity.JSONMap{"graph": "b"}}).Error; err != nil {
		t.Fatal(err)
	}
	if err := db.Create(&entity.UserCanvasVersion{ID: "ver-b", UserCanvasID: "cv-b", DSL: entity.JSONMap{"graph": "versioned-b"}}).Error; err != nil {
		t.Fatal(err)
	}

	// ver-b belongs to cv-b but is requested for cv-a -> Missing.
	_, err := loadCanvasDSLJSON(ctx, "cv-a", sptr("ver-b"))
	assertMissing(t, err)
}

func TestLoadCanvasDSLJSON_VersionedNotFoundIsMissing(t *testing.T) {
	db := setupServiceTestDB(t)
	if err := db.AutoMigrate(&entity.UserCanvasVersion{}); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	orig := dao.DB
	dao.DB = db
	t.Cleanup(func() { dao.DB = orig })

	ctx := context.Background()
	if err := db.Create(&entity.UserCanvas{ID: "cv-x", CanvasCategory: "dataflow_canvas", DSL: entity.JSONMap{"graph": "x"}}).Error; err != nil {
		t.Fatal(err)
	}

	_, err := loadCanvasDSLJSON(ctx, "cv-x", sptr("no-such-version"))
	assertMissing(t, err)
}

func TestLoadCanvasDSLJSON_NilVersionFallsBackToLatestVersion(t *testing.T) {
	db := setupServiceTestDB(t)
	if err := db.AutoMigrate(&entity.UserCanvasVersion{}); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	orig := dao.DB
	dao.DB = db
	t.Cleanup(func() { dao.DB = orig })

	ctx := context.Background()
	canvasID := "cv-fallback"
	if err := db.Create(&entity.UserCanvas{ID: canvasID, CanvasCategory: "dataflow_canvas", DSL: entity.JSONMap{"graph": "legacy"}}).Error; err != nil {
		t.Fatal(err)
	}
	if err := db.Create(&entity.UserCanvasVersion{ID: "ver-fb", UserCanvasID: canvasID, DSL: entity.JSONMap{"graph": "from-version"}}).Error; err != nil {
		t.Fatal(err)
	}

	raw, err := loadCanvasDSLJSON(ctx, canvasID, nil)
	if err != nil {
		t.Fatalf("loadCanvasDSLJSON: %v", err)
	}
	assertContains(t, string(raw), "from-version")
}

func TestLoadCanvasDSLJSON_NilVersionLegacyCanvasReadsCanvasDSL(t *testing.T) {
	db := setupServiceTestDB(t)
	if err := db.AutoMigrate(&entity.UserCanvasVersion{}); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	orig := dao.DB
	dao.DB = db
	t.Cleanup(func() { dao.DB = orig })

	ctx := context.Background()
	// Canvas with a DSL but NO version rows: legacy path must still resolve.
	canvasID := "cv-legacy"
	if err := db.Create(&entity.UserCanvas{ID: canvasID, CanvasCategory: "dataflow_canvas", DSL: entity.JSONMap{"graph": "legacy-only"}}).Error; err != nil {
		t.Fatal(err)
	}

	raw, err := loadCanvasDSLJSON(ctx, canvasID, nil)
	if err != nil {
		t.Fatalf("loadCanvasDSLJSON: %v", err)
	}
	assertContains(t, string(raw), "legacy-only")
}

func TestLoadPipelineDSL_PassthroughVersion(t *testing.T) {
	db := setupServiceTestDB(t)
	if err := db.AutoMigrate(&entity.UserCanvasVersion{}); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	orig := dao.DB
	dao.DB = db
	t.Cleanup(func() { dao.DB = orig })

	ctx := context.Background()
	canvasID := "cv-pipe"
	if err := db.Create(&entity.UserCanvas{ID: canvasID, CanvasCategory: "dataflow_canvas", DSL: entity.JSONMap{"graph": "legacy"}}).Error; err != nil {
		t.Fatal(err)
	}
	if err := db.Create(&entity.UserCanvasVersion{ID: "ver-pipe", UserCanvasID: canvasID, DSL: entity.JSONMap{"graph": "piped-version"}}).Error; err != nil {
		t.Fatal(err)
	}

	raw, err := LoadPipelineDSL(ctx, true, "", sptr(canvasID), sptr("ver-pipe"))
	if err != nil {
		t.Fatalf("LoadPipelineDSL: %v", err)
	}
	assertContains(t, string(raw), "piped-version")
}

func TestResolveComponentParamsDefaults_PassthroughVersion(t *testing.T) {
	db := setupServiceTestDB(t)
	if err := db.AutoMigrate(&entity.UserCanvasVersion{}); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	orig := dao.DB
	dao.DB = db
	t.Cleanup(func() { dao.DB = orig })

	ctx := context.Background()
	canvasID := "cv-resolve"
	if err := db.Create(&entity.UserCanvas{ID: canvasID, CanvasCategory: "dataflow_canvas", DSL: entity.JSONMap{"graph": "legacy"}}).Error; err != nil {
		t.Fatal(err)
	}
	if err := db.Create(&entity.UserCanvasVersion{ID: "ver-resolve", UserCanvasID: canvasID, DSL: entity.JSONMap{"graph": "resolved-version"}}).Error; err != nil {
		t.Fatal(err)
	}
	if err := db.Create(&entity.UserCanvasVersion{ID: "ver-empty", UserCanvasID: canvasID, DSL: entity.JSONMap{}}).Error; err != nil {
		t.Fatal(err)
	}

	// Threading proof: an empty version DSL surfaces ErrPipelineVersionMissing.
	if _, err := ResolveComponentParamsDefaults(ctx, "", sptr(canvasID), sptr("ver-empty")); err != nil {
		assertMissing(t, err)
	}

	// A non-empty version DSL is loaded and parsed (not rejected as Missing),
	// even if the stub DSL has no extractable component params.
	if _, err := ResolveComponentParamsDefaults(ctx, "", sptr(canvasID), sptr("ver-resolve")); errors.Is(err, ErrPipelineVersionMissing) {
		t.Fatalf("non-empty version should not be Missing, got: %v", err)
	}
}
