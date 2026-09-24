//
//  Copyright 2026 The InfiniFlow Authors All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

package component

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
)

// withMiniredisCache wires the parser cache seam to a fresh miniredis instance
// and returns a cleanup that restores the original getter. Tests that exercise
// the kvrocks-backed path use this instead of the global kvrocks singleton
// (which is absent in the unit tier).
func withMiniredisCache(t *testing.T) *redis.Client {
	t.Helper()
	mr := miniredis.RunT(t)
	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	t.Cleanup(func() { _ = client.Close() })
	orig := parserCacheClientGetter
	parserCacheClientGetter = func() redis.UniversalClient { return client }
	t.Cleanup(func() { parserCacheClientGetter = orig })
	return client
}

func TestParserCacheEnabled(t *testing.T) {
	t.Setenv("RAGFLOW_CACHE_PARSER", "")
	if parserCacheEnabled() {
		t.Fatal("parser cache should be disabled when env unset")
	}
	t.Setenv("RAGFLOW_CACHE_PARSER", "1")
	if !parserCacheEnabled() {
		t.Fatal("parser cache should be enabled when RAGFLOW_CACHE_PARSER=1")
	}
}

func TestParserCacheKeyShape(t *testing.T) {
	got := parserCacheKey("doc-42", "pdf")
	want := "parser:cp:doc-42:pdf"
	if got != want {
		t.Fatalf("parserCacheKey = %q, want %q", got, want)
	}
}

// When the cache is disabled, save is a no-op and load always misses — even if
// a client is wired. This protects normal (non-debug) runs from any accidental
// kvrocks round-trip.
func TestParserCacheDisabledIsNoOp(t *testing.T) {
	t.Setenv("RAGFLOW_CACHE_PARSER", "")
	withMiniredisCache(t)

	key := parserCacheKey("doc-x", "pdf")
	out := map[string]any{"output_format": "json", "name": "x.pdf"}
	if err := saveParserCache(context.Background(), key, out); err != nil {
		t.Fatalf("saveParserCache (disabled) error: %v", err)
	}
	got, hit, err := tryLoadParserCache(context.Background(), key)
	if err != nil {
		t.Fatalf("tryLoadParserCache (disabled) error: %v", err)
	}
	if hit || got != nil {
		t.Fatalf("expected miss when disabled, got (hit=%v, val=%v)", hit, got)
	}
}

// When the seam returns a nil client, both paths are no-ops regardless of the
// enabled flag (mirrors kvrocks being unconfigured in a Redis-less deploy).
func TestParserCacheNilClientIsNoOp(t *testing.T) {
	t.Setenv("RAGFLOW_CACHE_PARSER", "1")
	orig := parserCacheClientGetter
	parserCacheClientGetter = func() redis.UniversalClient { return nil }
	t.Cleanup(func() { parserCacheClientGetter = orig })

	key := parserCacheKey("doc-y", "pdf")
	out := map[string]any{"output_format": "json"}
	if err := saveParserCache(context.Background(), key, out); err != nil {
		t.Fatalf("saveParserCache (nil client) error: %v", err)
	}
	got, hit, err := tryLoadParserCache(context.Background(), key)
	if err != nil {
		t.Fatalf("tryLoadParserCache (nil client) error: %v", err)
	}
	if hit || got != nil {
		t.Fatalf("expected miss with nil client, got (hit=%v, val=%v)", hit, got)
	}
}

// Enabled path: a stored Parser output round-trips and is byte-for-byte
// equivalent after the JSON marshal/unmarshal (the Chunker re-encodes anyway).
func TestParserCacheRoundTrip(t *testing.T) {
	t.Setenv("RAGFLOW_CACHE_PARSER", "1")
	withMiniredisCache(t)

	key := parserCacheKey("doc-z", "pdf")
	out := map[string]any{
		"output_format": "json",
		"name":          "z.pdf",
		"json": []map[string]any{
			{"content": "page one", "page_num": float64(1)},
			{"content": "page two", "page_num": float64(2)},
		},
	}
	if err := saveParserCache(context.Background(), key, out); err != nil {
		t.Fatalf("saveParserCache error: %v", err)
	}
	got, hit, err := tryLoadParserCache(context.Background(), key)
	if err != nil {
		t.Fatalf("tryLoadParserCache error: %v", err)
	}
	if !hit {
		t.Fatal("expected cache hit")
	}
	wantBytes, _ := json.Marshal(out)
	gotBytes, _ := json.Marshal(got)
	if string(wantBytes) != string(gotBytes) {
		t.Fatalf("round-trip mismatch:\n want %s\n got  %s", wantBytes, gotBytes)
	}
}
