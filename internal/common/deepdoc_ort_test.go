package common

import "testing"

func TestDefaultDeepDocORTLib(t *testing.T) {
	got := DefaultDeepDocORTLib("/home/x")
	want := "/home/x/ragflow-native-libs/onnxruntime/onnxruntime-linux-x64-1.23.2/lib/libonnxruntime.so.1.23.2"
	if got != want {
		t.Fatalf("DefaultDeepDocORTLib(/home/x) = %q, want %q", got, want)
	}
	if DefaultDeepDocORTLib("") != "" {
		t.Fatal("DefaultDeepDocORTLib(\"\") should return empty (no home)")
	}
}
