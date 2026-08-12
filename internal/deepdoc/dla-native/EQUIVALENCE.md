# dla-native ↔ Python deepdoc: Equivalence Proof

## Scope

This document proves that the Go `dla-native` inference library produces output
**equivalent** to the Python `deepdoc` inference service (the `deepdoc_server`
HTTP service backed by the `deepdoc/vision` recognizers) for the four tasks it
covers:

- **Det** — DB text detection
- **DLA** — document layout analysis
- **TSR** — table structure recognition
- **OCR** — text-line recognition

Both sides load the **same ONNX models** from `InfiniFlow/deepdoc`
(`det.onnx`, `layout.onnx`, `tsr.onnx`, `rec.onnx`) and the same `ocr.res`
character dictionary, and consume the **same raster image bytes**. The Python
service decodes request bytes with PIL / cv2; the Go side decodes with Go's
`image` package — both are format-agnostic raster decoders, so the inference
boundary is identical (raster in, boxes/text out). PDF rasterization happens
upstream of both, outside this boundary.

## How to reproduce

The proof is a reproducible test harness. From
`internal/deepdoc/dla-native/native`:

```bash
ORT_LIB=<path/to/libonnxruntime.so> \
MODEL_DIR=<path/to/InfiniFlow/deepdoc> \
go test -tags integration -run 'TestEquivalenceReport|TestDetMembershipAllFixtures' -v ./...
```

- `TestEquivalenceReport` prints the consolidated summary below to the test
  log (visible in CI).
- `TestDetMembershipAllFixtures` guards the full-fixture detection floor.

Models are fetched once with `ragflow_deps/download_deps.py` (a snapshot of
`InfiniFlow/deepdoc`). The ONNX Runtime shared library is `libonnxruntime.so`
(any 1.23.x build; validated with 1.23.2, ABI-compatible with the
`onnxruntime_go` v1.23.0 binding — the same line the Python goldens were
generated with, so validated parity is preserved).

## Evidence (measured)

| Task | Fixtures | Match | Max Δ | Status |
|------|----------|-------|-------|--------|
| DLA  | 5 | 46/46 boxes | < 0.13 px | OK |
| TSR  | 3 | 36/36 boxes | < 0.9 px (≤ 10 px on a 4:1 aspect crop, structure preserved) | OK |
| OCR  | 8 | exact text (EN / CJK / mixed / digits) | — | OK |
| Det  | all | IoU box-membership orphan **3/5** (gold 1125 / go 1127) | — | OK (accepted floor) |

The detection orphan boxes were adjudicated with OCR: the boxes unique to one
side yield essentially no *real* text (Python-only real text 0/3; Go-only real
text 3/5), i.e. the divergence is benign and Go occasionally recovers genuine
text the Python side misses.

## Why the divergence is bounded and deterministic

- **Det 3/5** — contour-tracer geometry. Go's pure-Go boundary follower
  (Moore-neighbour, Suzuki-Abe style, `RETR_LIST`) selects boundary pixels
  slightly differently from cv2's `findContours` at 8-connected diagonal / hole
  junctions. 100% of the orphans are SCORE-FLIPs: a different convex hull →
  different `minAreaRect` → `box_score_fast` crosses 0.5 at a handful of
  regions on dense pages. It does not flip any segmentation.
- **≤ 1/255 residual** — ONNX Runtime's fixed-point `uint8` resize introduces a
  pixel-level noise ceiling (max |Δ| = 1 gray level vs Go's float bilinear). It
  is irreducible for any pure-float implementation and is far below the 0.5 /
  0.3 score thresholds, so it never changes a detection or score.

Both effects are **deterministic and reproducible**, not random accuracy loss.

## What is NOT proven here (honest boundaries)

1. **Not bit-identical.** "Equivalent within bounded, accepted floors" is the
   correct claim; "perfect / pixel-identical" is not. The Det 3/5 floor is a
   known, accepted divergence.
2. **No production caller is wired yet.** The library is validated and callable
   in-process, but nothing in the server / ingestion pipeline consumes it yet;
   replacing the running Python service end-to-end has not been done.
3. **Coverage confirmation required.** Go implements {det, dla, tsr, ocr}.
   Confirm the Python path being replaced uses only these recognizers (e.g. a
   separate table-cell recognizer would not be covered).
4. **Runtime version.** Validated with onnxruntime 1.23.2; re-verify if the
   production Python uses a different ORT version (stable within 1.23.x).
5. **HTTP server shape.** If the goal is a standalone HTTP service mirroring
   `deepdoc_server`, that surface is not built — only the in-process library and
   a CLI (`main.go`) exist.
