# native — DeepDoc pure-Go port and drift-comparison harness

> This file is the authoritative design record (ADR style) for `native`.
> The equivalence proof lives in `EQUIVALENCE.md`; the historical session
> handoff notes (HANDOFF*) were deleted — their still-valid points are folded
> into this file.

## 1. Positioning and boundary

`native` is a **standalone verification harness**, not a production path:

- It reimplements the DeepDoc Python inference pipeline (OCR text detection
  `det` / layout analysis `DLA` / table structure `TSR` / text recognition
  `OCR-rec`) in **pure Go + ONNX Runtime**, running on CPU.
- It compares against the Python reference implementation (`ref_*.py`, which
  imports the repo-root `deepdoc`) to catch regressions in the port.
- **Wired into production as the sole backend**: `internal/deepdoc` no longer
  talks to the remote Python service over HTTP — production serves DeepDoc
  entirely in-process via `infnative.NativeAnalyzer` (built with `-tags
  cgo`). The Python service survives only as a read-only equivalence
  oracle in the `inprocess_vs_service_*` tests (via the test-only `PyOracle`
  client), never as a production path. The `native` harness and the
  production backend share the same ONNX Runtime port, so `native`'s
  regression checks directly guard the production inference.

It is a regular package inside the main module
(`ragflow/internal/deepdoc/native`), gated by the `cgo` build tag so the
ONNX Runtime (`onnxruntime_go`) cgo binding stays out of the default (no-cgo)
build — the same isolation used for `office_oxide` / `pdfium` / `pdf_oxide`.

## 2. Why pure Go (P3 decision)

A `gocv` / `nogocv` dual build once existed: the gocv path used cv2 decode +
resize and reached 1:1 parity, while pure Go had a ~3 px floor. The code has
**converged to a single pure-Go path** (`image_gocv.go` / `det_gocv.go` /
`dla_gocv.go` deleted, `gocv.io/x/gocv` removed from `go.mod`, the CI
`go-native-gocv` job removed).

Trade-off: give up cv2 1:1 parity in exchange for **zero OpenCV / CGO
dependency**.

## 3. Where the 3px floor comes from (known hard floor)

The port's maximum coordinate residual vs the Python reference is stable at
**~3 px**, from:

- `bilinearResize` (Go float weights) vs cv2 fixed-point `INTER_LINEAR`
  implementation difference;
- the contour minimum-area rect in the `box#8` post-process step
  (`minAreaRect`).

The floor is **input-format independent** (measured 3.0 px for both JPG and
PNG; decode contributes ~0 because PNG losslessly wraps JPEG-decoded pixels).
The geometry core is not touched unless a decision is made to emulate cv2's
bit-exact float `minAreaRect` (which would require reintroducing OpenCV).

## 4. How goldens are generated / what the drift gate proves

- **golden**: output of `ref_*.py` (the Python oracle) on fixed fixtures,
  frozen to `testdata/<stem>.<task>.golden.json`. Fixtures are now **PNG**
  (losslessly transcoded from JPG, pixel-equivalent, verified 47/47 zero
  difference), matching the production `EncodePNG` wire.
- **`python-drift` job**: re-runs the Python oracle and compares against the
  golden — it alerts only when **the Python logic drifts and the golden is
  regenerated and committed**. It does **not** independently prove Python is
  correct (Python is the trust anchor / oracle).
- **`go-native` job**: runs the Go port against the same golden — catches
  **Go regressions**.

> Conclusion: the drift gate is a "Go vs frozen Python snapshot" comparison.
> It reliably catches Go-side regressions but does **not** prove the Python
> side is correct — this is inherent to the "re-implement-to-verify" pattern.

## 5. Test tiers and safety

- **unit** (no tag): pure geometry/post-process unit tests
  (`clipper_offset_test.go` cross-checked against pyclipper at 0 px,
  `minAreaRect` cross-check, `image_test.go` decode limits); run by default
  via `go test ./native/`.
- **integration** (`//go:build integration`, needs `MODEL_DIR`; ORT is always
  resolved from the statically-linked binary via dlopen(NULL), so no `ORT_LIB`
  is needed and self-skips only when the binary was not built with static ORT):
  full-component Go-vs-golden comparison.
- **decode safety**: `Decode` validates the decoded raster's size/pixel
  limits (`maxImageDim=16384`, `maxImagePixels=100MP`) to defend against
  decompression bombs. It currently runs only on fixed fixtures with no
  production exposure; the limits activate if untrusted input is ever wired
  in.

## 6. Comparison tolerance

Coordinate tolerance = **`coordFloor(3.0) + coordTolMargin(0.5)` = 3.5**,
computed from constants rather than a literal (`native_integration_test.go`).
`coordFloor` is the 3 px hard floor of §3; `coordTolMargin` lifts the
tolerance just above the floor so that a regression crossing the floor trips
the gate instead of hiding under it.

Because the tolerance derives from `coordFloor`, adjusting the floor later
**follows automatically** with no manual sync. The gate can only catch
regressions **> 3 px (the floor)**, which is enough for "prevent large
breakage"; finer regressions (below the floor) are inherently
indistinguishable — this is the tool's sensitivity floor, not a defect.

## 7. How to regenerate goldens

Goldens are the `ref_*.py` (Python oracle) output on fixed fixtures, frozen to
`testdata/<stem>.<task>.golden.json`. After a deepdoc Python-logic change,
re-run the oracle and write the output back into the goldens, then commit —
otherwise the `python-drift` job alerts (`check_drift.py` only compares, never
writes).

Prerequisites (some venv with `deepdoc` + `onnxruntime` + `opencv-python`):

```bash
export MODEL_DIR=<deepdoc model dir>
export PYTHONPATH=<ragflow repo root>   # so ref_det.py can import deepdoc
```

Each `ref_*.py` prints the wire JSON to stdout; redirect to write it to disk
(format identical to the existing goldens):

```bash
cd internal/deepdoc/native
# single fixture + single task
python ref_det.py     testdata/page0.png    "$MODEL_DIR" > testdata/page0.det.golden.json
python ref_dla.py     testdata/page0.png    "$MODEL_DIR" > testdata/page0.dla.golden.json
python ref_tsr.py     testdata/table0.png   "$MODEL_DIR" > testdata/table0.tsr.golden.json
python ref_ocr_rec.py testdata/line0.png    "$MODEL_DIR" > testdata/line0.ocr_rec.golden.json

# all fixtures: enumerate the existing goldens in testdata, auto-match task and image stem
for f in testdata/*.golden.json; do
  bn=$(basename "$f" .golden.json)   # e.g. page0.dla
  task=${bn##*.}                     # dla / det / tsr / ocr_rec
  imgstem=${bn%.*}                   # page0
  case "$task" in
    det)     python ref_det.py     "testdata/$imgstem.png" "$MODEL_DIR" > "$f" ;;
    dla)     python ref_dla.py     "testdata/$imgstem.png" "$MODEL_DIR" > "$f" ;;
    tsr)     python ref_tsr.py     "testdata/$imgstem.png" "$MODEL_DIR" > "$f" ;;
    ocr_rec) python ref_ocr_rec.py "testdata/$imgstem.png" "$MODEL_DIR" > "$f" ;;
  esac
done
```

After regeneration, run `check_drift.py` to confirm no other drift, then
commit the changed goldens together.
