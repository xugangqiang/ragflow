// Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package dataset

import (
	"testing"
	"time"

	"ragflow/internal/common"
	"ragflow/internal/entity"
	"ragflow/internal/service"

	"gorm.io/gorm"
)

func i64ptr(v int64) *int64 { return &v }

// seedDataflowCanvasWithVersion inserts a dataflow canvas owned by tenant-1 with
// a single version row so GetLatest can resolve it.
func seedDataflowCanvasWithVersion(t *testing.T, db *gorm.DB, canvasID, versionID string) {
	t.Helper()
	if err := db.AutoMigrate(&entity.UserCanvasVersion{}); err != nil {
		t.Fatalf("migrate version table: %v", err)
	}
	canvas := &entity.UserCanvas{
		ID:             canvasID,
		UserID:         "tenant-1",
		CanvasCategory: "dataflow_canvas",
		Permission:     "me",
		DSL:            entity.JSONMap{"graph": "legacy"},
	}
	if err := db.Create(canvas).Error; err != nil {
		t.Fatalf("seed canvas: %v", err)
	}
	ver := &entity.UserCanvasVersion{
		ID:           versionID,
		UserCanvasID: canvasID,
		DSL:          entity.JSONMap{"graph": "versioned"},
		CreateTime:   i64ptr(time.Now().UnixMilli()),
	}
	if err := db.Create(ver).Error; err != nil {
		t.Fatalf("seed version: %v", err)
	}
}

// seedBoundKB creates a knowledgebase already bound to a pipeline version.
func seedBoundKB(t *testing.T, db *gorm.DB, kbID, canvasID, versionID string) {
	t.Helper()
	kb := &entity.Knowledgebase{
		ID:                kbID,
		TenantID:          "tenant-1",
		CreatedBy:         "tenant-1",
		Name:              kbID,
		EmbdID:            "BAAI/bge-large-zh-v1.5@Builtin",
		Permission:        string(entity.TenantPermissionMe),
		ParserID:          string(entity.ParserTypeNaive),
		ParserConfig:      entity.JSONMap{},
		Status:            sptr(string(entity.StatusValid)),
		PipelineID:        &canvasID,
		PipelineVersionID: &versionID,
	}
	if err := db.Create(kb).Error; err != nil {
		t.Fatalf("seed kb: %v", err)
	}
	ensureDatasetTestMembership(t, "tenant-1", "tenant-1", "owner")
}

// T6: creating a dataset in Pipeline mode binds the latest pipeline version.
func TestCreateDataset_BindsLatestPipelineVersion(t *testing.T) {
	db := setupServiceTestDB(t)
	pushServiceDB(t, db)
	insertCreateDatasetTenant(t, "tenant-1")
	ctx := t.Context()

	canvasID := "0123456789abcdef0123456789abcdef"
	versionID := "version0000000000000000000000aa"
	seedDataflowCanvasWithVersion(t, db, canvasID, versionID)

	parseType := 2
	result, code, err := testDatasetCreateService(t).CreateDataset(ctx, &service.CreateDatasetRequest{
		Name:       "ds-bind-version",
		PipelineID: &canvasID,
		ParseType:  &parseType,
	}, testDatasetSubject("tenant-1"))
	if err != nil || code != common.CodeSuccess {
		t.Fatalf("CreateDataset err=%v code=%d", err, code)
	}
	var kb entity.Knowledgebase
	if err := db.Where("id = ?", result["id"]).Take(&kb).Error; err != nil {
		t.Fatalf("load kb: %v", err)
	}
	if kb.PipelineVersionID == nil || *kb.PipelineVersionID != versionID {
		t.Fatalf("PipelineVersionID = %v, want %s", kb.PipelineVersionID, versionID)
	}
}

// T6b: BuiltIn mode never binds a pipeline version.
func TestCreateDataset_BuiltInHasNoPipelineVersion(t *testing.T) {
	db := setupServiceTestDB(t)
	pushServiceDB(t, db)
	insertCreateDatasetTenant(t, "tenant-1")
	ctx := t.Context()

	parseType := 1
	parserID := "general"
	result, code, err := testDatasetCreateService(t).CreateDataset(ctx, &service.CreateDatasetRequest{
		Name:      "ds-builtin",
		ParserID:  &parserID,
		ParseType: &parseType,
	}, testDatasetSubject("tenant-1"))
	if err != nil || code != common.CodeSuccess {
		t.Fatalf("CreateDataset err=%v code=%d", err, code)
	}
	var kb entity.Knowledgebase
	if err := db.Where("id = ?", result["id"]).Take(&kb).Error; err != nil {
		t.Fatalf("load kb: %v", err)
	}
	if kb.PipelineVersionID != nil {
		t.Fatalf("PipelineVersionID = %v, want nil for builtin", *kb.PipelineVersionID)
	}
}

// T7: changing the bound pipeline rewrites pipeline_version_id to the new
// canvas's latest version (sticky snapshot, design B4).
func TestUpdateDataset_RewritesVersionOnPipelineChange(t *testing.T) {
	db := setupDatasetUpdateTestDB(t)
	pushServiceDB(t, db)

	canvas1 := "0123456789abcdef0123456789abcdef"
	ver1 := "version0000000000000000000000aa"
	canvas2 := "fedcba9876543210fedcba9876543210"
	ver2 := "version0000000000000000000000bb"
	seedDataflowCanvasWithVersion(t, db, canvas1, ver1)
	seedDataflowCanvasWithVersion(t, db, canvas2, ver2)
	seedBoundKB(t, db, "kb-change", canvas1, ver1)

	parseType := 2
	_, code, err := testDatasetUpdateService(t).UpdateDataset(t.Context(), "kb-change", "tenant-1", service.UpdateDatasetRequest{
		PipelineID: &canvas2,
		ParseType:  &parseType,
	})
	if err != nil || code != common.CodeSuccess {
		t.Fatalf("UpdateDataset err=%v code=%d", err, code)
	}
	var kb entity.Knowledgebase
	if err := db.Where("id = ?", "kb-change").Take(&kb).Error; err != nil {
		t.Fatalf("load kb: %v", err)
	}
	if kb.PipelineVersionID == nil || *kb.PipelineVersionID != ver2 {
		t.Fatalf("PipelineVersionID = %v, want %s after pipeline change", kb.PipelineVersionID, ver2)
	}
}

// T7b: leaving the pipeline unchanged preserves the existing pipeline_version_id.
func TestUpdateDataset_PreservesVersionWhenPipelineUnchanged(t *testing.T) {
	db := setupDatasetUpdateTestDB(t)
	pushServiceDB(t, db)

	canvas1 := "0123456789abcdef0123456789abcdef"
	ver1 := "version0000000000000000000000aa"
	seedDataflowCanvasWithVersion(t, db, canvas1, ver1)
	seedBoundKB(t, db, "kb-preserve", canvas1, ver1)

	parseType := 2
	name := "kb-preserve-renamed"
	_, code, err := testDatasetUpdateService(t).UpdateDataset(t.Context(), "kb-preserve", "tenant-1", service.UpdateDatasetRequest{
		Name:       &name,
		PipelineID: &canvas1,
		ParseType:  &parseType,
	})
	if err != nil || code != common.CodeSuccess {
		t.Fatalf("UpdateDataset err=%v code=%d", err, code)
	}
	var kb entity.Knowledgebase
	if err := db.Where("id = ?", "kb-preserve").Take(&kb).Error; err != nil {
		t.Fatalf("load kb: %v", err)
	}
	if kb.PipelineVersionID == nil || *kb.PipelineVersionID != ver1 {
		t.Fatalf("PipelineVersionID = %v, want preserved %s", kb.PipelineVersionID, ver1)
	}
}

// seedDataflowCanvas inserts a dataflow canvas owned by tenant-1 (no versions).
func seedDataflowCanvas(t *testing.T, db *gorm.DB, canvasID string) {
	t.Helper()
	if err := db.AutoMigrate(&entity.UserCanvasVersion{}); err != nil {
		t.Fatalf("migrate version table: %v", err)
	}
	canvas := &entity.UserCanvas{
		ID:             canvasID,
		UserID:         "tenant-1",
		CanvasCategory: "dataflow_canvas",
		Permission:     "me",
		DSL:            entity.JSONMap{"graph": "legacy"},
	}
	if err := db.Create(canvas).Error; err != nil {
		t.Fatalf("seed canvas: %v", err)
	}
}

// seedCanvasVersion inserts a version row with an explicit create-time offset
// (milliseconds) so GetLatest ordering is deterministic in tests.
func seedCanvasVersion(t *testing.T, db *gorm.DB, canvasID, versionID string, createOffsetMs int64) {
	t.Helper()
	ver := &entity.UserCanvasVersion{
		ID:           versionID,
		UserCanvasID: canvasID,
		DSL:          entity.JSONMap{"graph": "versioned"},
		CreateTime:   i64ptr(time.Now().UnixMilli() + createOffsetMs),
	}
	if err := db.Create(ver).Error; err != nil {
		t.Fatalf("seed version %s: %v", versionID, err)
	}
}

// T7c: an explicit PipelineVersionID (not the latest) is bound as-is when it
// belongs to the bound pipeline (design section 5 + T7 manual pin).
func TestCreateDataset_BindsExplicitPipelineVersion(t *testing.T) {
	db := setupServiceTestDB(t)
	pushServiceDB(t, db)
	insertCreateDatasetTenant(t, "tenant-1")
	ctx := t.Context()

	canvasID := "0123456789abcdef0123456789abcdef"
	verOld := "version0000000000000000000000aa"
	verNew := "version0000000000000000000000bb"
	seedDataflowCanvas(t, db, canvasID)
	seedCanvasVersion(t, db, canvasID, verOld, 0)
	seedCanvasVersion(t, db, canvasID, verNew, 1000) // verNew is the latest

	parseType := 2
	result, code, err := testDatasetCreateService(t).CreateDataset(ctx, &service.CreateDatasetRequest{
		Name:              "ds-explicit-version",
		PipelineID:        &canvasID,
		PipelineVersionID: &verOld, // explicitly pin the older version
		ParseType:         &parseType,
	}, testDatasetSubject("tenant-1"))
	if err != nil || code != common.CodeSuccess {
		t.Fatalf("CreateDataset err=%v code=%d", err, code)
	}
	var kb entity.Knowledgebase
	if err := db.Where("id = ?", result["id"]).Take(&kb).Error; err != nil {
		t.Fatalf("load kb: %v", err)
	}
	if kb.PipelineVersionID == nil || *kb.PipelineVersionID != verOld {
		t.Fatalf("PipelineVersionID = %v, want explicit %s (not latest %s)", kb.PipelineVersionID, verOld, verNew)
	}
}

// T7d: an explicit PipelineVersionID that belongs to a different canvas is
// rejected (design T13 ownership check).
func TestCreateDataset_RejectsForeignPipelineVersion(t *testing.T) {
	db := setupServiceTestDB(t)
	pushServiceDB(t, db)
	insertCreateDatasetTenant(t, "tenant-1")
	ctx := t.Context()

	canvas1 := "0123456789abcdef0123456789abcdef"
	canvas2 := "fedcba9876543210fedcba9876543210"
	ver1 := "version0000000000000000000000aa"
	verForeign := "version0000000000000000000000cc"
	seedDataflowCanvas(t, db, canvas1)
	seedDataflowCanvas(t, db, canvas2)
	seedCanvasVersion(t, db, canvas1, ver1, 0)
	seedCanvasVersion(t, db, canvas2, verForeign, 0)

	parseType := 2
	_, code, err := testDatasetCreateService(t).CreateDataset(ctx, &service.CreateDatasetRequest{
		Name:              "ds-foreign-version",
		PipelineID:        &canvas1,
		PipelineVersionID: &verForeign, // belongs to canvas2
		ParseType:         &parseType,
	}, testDatasetSubject("tenant-1"))
	if err == nil && code == common.CodeSuccess {
		t.Fatalf("expected rejection of foreign pipeline version, got success")
	}
	if code != common.CodeBadRequest {
		t.Fatalf("expected CodeBadRequest, got code=%d err=%v", code, err)
	}
}

// T7e: updating with an explicit PipelineVersionID re-pins to that version even
// when the pipeline itself is unchanged (design T7 manual pin).
func TestUpdateDataset_RebindsExplicitVersion(t *testing.T) {
	db := setupDatasetUpdateTestDB(t)
	pushServiceDB(t, db)

	canvas1 := "0123456789abcdef0123456789abcdef"
	verOld := "version0000000000000000000000aa"
	verNew := "version0000000000000000000000bb"
	seedDataflowCanvas(t, db, canvas1)
	seedCanvasVersion(t, db, canvas1, verOld, 0)
	seedCanvasVersion(t, db, canvas1, verNew, 1000)
	seedBoundKB(t, db, "kb-repin", canvas1, verOld)

	parseType := 2
	_, code, err := testDatasetUpdateService(t).UpdateDataset(t.Context(), "kb-repin", "tenant-1", service.UpdateDatasetRequest{
		PipelineID:        &canvas1,
		PipelineVersionID: &verNew,
		ParseType:         &parseType,
	})
	if err != nil || code != common.CodeSuccess {
		t.Fatalf("UpdateDataset err=%v code=%d", err, code)
	}
	var kb entity.Knowledgebase
	if err := db.Where("id = ?", "kb-repin").Take(&kb).Error; err != nil {
		t.Fatalf("load kb: %v", err)
	}
	if kb.PipelineVersionID == nil || *kb.PipelineVersionID != verNew {
		t.Fatalf("PipelineVersionID = %v, want re-pinned %s", kb.PipelineVersionID, verNew)
	}
}

// T14: pipelineVersionOutdated (! indicator) is true only when a pinned version
// is not the latest of its canvas (design section 6.3).
func TestPipelineVersionOutdated_ComputesFromLatest(t *testing.T) {
	db := setupServiceTestDB(t)
	pushServiceDB(t, db)

	canvasID := "0123456789abcdef0123456789abcdef"
	verOld := "version0000000000000000000000aa"
	verNew := "version0000000000000000000000bb"
	seedDataflowCanvas(t, db, canvasID)
	seedCanvasVersion(t, db, canvasID, verOld, 0)
	seedCanvasVersion(t, db, canvasID, verNew, 1000)

	ctx := t.Context()
	if pipelineVersionOutdated(ctx, &canvasID, &verNew) {
		t.Fatalf("pinned latest version should not be outdated")
	}
	if !pipelineVersionOutdated(ctx, &canvasID, &verOld) {
		t.Fatalf("pinned older version should be outdated")
	}
	if pipelineVersionOutdated(ctx, nil, &verOld) {
		t.Fatalf("no pipeline should never be outdated")
	}
	if pipelineVersionOutdated(ctx, &canvasID, nil) {
		t.Fatalf("no pinned version should never be outdated")
	}
}
