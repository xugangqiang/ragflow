package main

// Minimal runnable local DeepDoc prototype — proves the Python DeepDoc
// inference service can run natively in Go on CPU via ONNX Runtime, with no
// Python service in the loop.
//
// This file is intentionally thin: it wires flags to the native recognizers
// and prints their wire output. All model logic lives in package native.
//
// Usage:
//   ORT_LIB=/path/to/libonnxruntime.so.1.23.2 \
//     go run . -task dla -image testdata/page0.png -modeldir /path/to/models
//
// Tasks: dla | tsr | ocr-rec | det
// Wire formats (matching deepdoc/server/adapters/*):
//   dla/tsr : {"bboxes": [[x0,y0,x1,y1,score,class_id], ...]}
//   ocr-rec : {"output": [[[["text", score]]]]}  (score = real recognition confidence)
//   det     : {"output": [[ [ [x,y]*4, ... ] ]]}

import (
	"context"
	"flag"
	"fmt"
	"os"

	"dla-native/native"
)

func main() {
	task := flag.String("task", "dla", "recognizer: dla | tsr | ocr-rec | det")
	imagePath := flag.String("image", "testdata/page0.png", "input image (any format Go's image package decodes)")
	modelDir := flag.String("modeldir", os.Getenv("MODEL_DIR"), "dir containing *.onnx and ocr.res")
	libPath := flag.String("lib", os.Getenv("ORT_LIB"), "path to libonnxruntime shared library")
	flag.Parse()

	if *libPath == "" {
		fmt.Fprintln(os.Stderr, "error: set ORT_LIB (or -lib) to the onnxruntime shared library")
		os.Exit(2)
	}
	if *modelDir == "" {
		fmt.Fprintln(os.Stderr, "error: set MODEL_DIR (or -modeldir) to the directory with *.onnx and ocr.res")
		os.Exit(2)
	}
	if err := native.InitORT(*libPath); err != nil {
		fmt.Fprintln(os.Stderr, "init onnxruntime:", err)
		os.Exit(1)
	}

	img, err := native.Decode(*imagePath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load image:", err)
		os.Exit(1)
	}

	var out string
	switch *task {
	case "dla":
		res, err := native.RunDLA(context.Background(), *modelDir, img)
		if err != nil {
			fmt.Fprintln(os.Stderr, "dla:", err)
			os.Exit(1)
		}
		out = res.Wire()
	case "tsr":
		res, err := native.RunTSR(context.Background(), *modelDir, img)
		if err != nil {
			fmt.Fprintln(os.Stderr, "tsr:", err)
			os.Exit(1)
		}
		out = res.Wire()
	case "ocr-rec":
		res, err := native.RunOCRRec(context.Background(), *modelDir, img)
		if err != nil {
			fmt.Fprintln(os.Stderr, "ocr-rec:", err)
			os.Exit(1)
		}
		out = res.Wire()
	case "det":
		res, err := native.RunDet(context.Background(), *modelDir, img)
		if err != nil {
			fmt.Fprintln(os.Stderr, "det:", err)
			os.Exit(1)
		}
		out = res.Wire()
	default:
		fmt.Fprintf(os.Stderr, "error: unknown -task %q (want dla|tsr|ocr-rec|det)\n", *task)
		os.Exit(2)
	}

	fmt.Println(out)
}
