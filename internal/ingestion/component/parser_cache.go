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
	"fmt"
	"os"
	"time"

	kvrocks "ragflow/internal/engine/kvrocks"

	"github.com/redis/go-redis/v9"
)

// parserCacheKeyPrefix namespaces Parser-result caches in the shared Kvrocks
// backend (the same Redis that backs eino's KvrocksCheckPointStore). A distinct
// prefix keeps Parser debug caches from colliding with agent checkpoints
// ("agent:cp:") or the run-tracker hash.
const parserCacheKeyPrefix = "parser:cp:"

// parserCacheTTL matches the chunkcache TTL (7d) so a cached Parser result
// outlives a single debugging session but does not accumulate forever.
const parserCacheTTL = 24 * time.Hour

// parserCacheClientGetter resolves the Kvrocks client used by the Parser cache.
// It is a package-level seam so unit tests can inject a miniredis-backed client
// without Bootstrapping the global kvrocks singleton. When kvrocks is not
// configured it returns nil, and every cache op becomes a no-op.
var parserCacheClientGetter = func() redis.UniversalClient {
	if kvrocks.IsEnabled() {
		if c := kvrocks.Get(); c != nil {
			return c.GetClient()
		}
	}
	return nil
}

// parserCacheEnabled gates the Parser result cache behind RAGFLOW_CACHE_PARSER=1.
// Off by default so production ingestion is unaffected unless a developer
// explicitly opts in for a debugging session.
func parserCacheEnabled() bool {
	return os.Getenv("RAGFLOW_CACHE_PARSER") == "1"
}

// parserCacheKey derives a stable cache key from the document id and the parsed
// file family. The family is included because the same doc id could in theory
// be routed through different parsers; we never want to serve a pdf parse to a
// docx request.
func parserCacheKey(docID, fileTypeExt string) string {
	return fmt.Sprintf("%s%s:%s", parserCacheKeyPrefix, docID, fileTypeExt)
}

// tryLoadParserCache returns the cached Parser output for key.
//
//	(nil, false, nil) — cache disabled, missing, or client absent (no-op).
//	(nil, false, err) — a real Redis error worth logging.
//	(out, true, nil)  — cache hit; out is JSON-equivalent to a fresh parse.
//
// The Chunker re-marshals the incoming map to JSON before decoding it into its
// typed struct (see chunker/token.go decodeChunkerFromUpstream), so the
// []any-vs-[]map[string]any drift introduced by json.Unmarshal is harmless.
func tryLoadParserCache(ctx context.Context, key string) (map[string]any, bool, error) {
	if !parserCacheEnabled() {
		return nil, false, nil
	}
	client := parserCacheClientGetter()
	if client == nil {
		return nil, false, nil
	}
	data, err := client.Get(ctx, key).Bytes()
	if err != nil {
		if err == redis.Nil {
			return nil, false, nil
		}
		return nil, false, err
	}
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, false, err
	}
	return out, true, nil
}

// saveParserCache persists a Parser output under key. No-op when the cache is
// disabled or the client is absent. A marshal error is returned (caller logs
// and proceeds) rather than fatal: caching must never block a real parse.
func saveParserCache(ctx context.Context, key string, out map[string]any) error {
	if !parserCacheEnabled() {
		return nil
	}
	client := parserCacheClientGetter()
	if client == nil {
		return nil
	}
	data, err := json.Marshal(out)
	if err != nil {
		return err
	}
	return client.Set(ctx, key, data, parserCacheTTL).Err()
}
