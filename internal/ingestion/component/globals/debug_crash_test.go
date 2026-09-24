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

package globals

import (
	"context"
	"os"
	"testing"

	"ragflow/internal/agent/runtime"
)

func TestDebugCrashAfterChunker(t *testing.T) {
	const envKey = "RAGFLOW_DEBUG_CRASH_AFTER_CHUNKER"
	cases := []struct {
		name   string
		env    string
		setEnv bool
		global bool
		want   bool
	}{
		{name: "env 1 wins", env: "1", setEnv: true, global: false, want: true},
		{name: "env true wins", env: "true", setEnv: true, global: false, want: true},
		// env present and falsy is authoritative: it must override a true global.
		{name: "env 0 disables global", env: "0", setEnv: true, global: true, want: false},
		// env absent falls back to the CanvasState.Globals bool.
		{name: "env absent + global true", setEnv: false, global: true, want: true},
		{name: "env absent + global false", setEnv: false, global: false, want: false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			os.Unsetenv(envKey)
			t.Cleanup(func() { os.Unsetenv(envKey) })
			if tc.setEnv {
				t.Setenv(envKey, tc.env)
			}
			ctx := context.Background()
			if tc.global {
				st := &runtime.CanvasState{Globals: map[string]any{DebugCrashAfterChunkerKey: true}}
				ctx = runtime.WithState(ctx, st)
			}
			if got := DebugCrashAfterChunker(ctx); got != tc.want {
				t.Fatalf("DebugCrashAfterChunker = %v, want %v", got, tc.want)
			}
		})
	}
}
