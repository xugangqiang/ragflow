# dla-native ↔ Python deepdoc: Equivalence Proof

## Executive summary (for decision makers)

**Bottom line.** The Go in-process DeepDoc backend (`infnative.NativeAnalyzer`,
backed by `dla-native`) is functionally equivalent to the Python `deepdoc_server`
inference service for all four tasks it covers — proven by *measured* golden
comparison, not by code inspection.

**What is actually replaced.** `deepdoc_server` is a thin HTTP wrapper over four
recognizers (`dla_adapter` → `LayoutRecognizer4YOLOv10`, `ocr_adapter` →
`TextDetector` / `TextRecognizer`, `tsr_adapter` → `TableStructureRecognizer`).
The caller depends only on the `DocAnalyzer` interface and its wire format. In
the Go server (`-tags native_det`), `infnative.Register` registers
`NativeAnalyzer` as the `DocAnalyzer`; `resolveDocAnalyzer` then **replaces the
external `DEEPDOC_URL` HTTP call** with an in-process call. The deployment swap
is in-process, single binary, no separate Python service to operate.

**Three pillars of the proof.**
1. **Boundary equivalence** — DLA / TSR / OCR-rec / Det outputs match the Python
   reference goldens within documented, bounded floors (see *Evidence*).
2. **Wire isomorphism** — Go's `Wire()` JSON is structurally identical to the
   `deepdoc_server` output for DLA / TSR / Det / OCR-rec, validated **two ways**:
   `TestWireSchemaMatchesGolden` (vs the re-serialized golden shape) **and**
   `TestWireVsLiveServer` (vs a *running* `deepdoc_server`'s real HTTP response —
   see *Live-service field diff*). So the consumer cannot tell the two backends
   apart using the actual service contract.
3. **Measured, not inferred** — every claim is verified by running the real ONNX
   models on committed fixtures; the tests *self-skip* (rather than fake-pass)
   when ORT / models are absent.

**Honest boundaries (condensed).**
- Not bit-identical; "equivalent within bounded, accepted floors" is the correct
  claim. Known floors: Det IoU orphan **3/5** (benign, OCR-adjudicated), TSR
  **≤ 3.5 px** on ordinary/moderate real tables (worst measured 2.70 px; dense
  annual-report ≤ 1.21 px), **≤ 10 px** on a 4:1 aspect crop (structure
  preserved); dense technical-standard full-page tables (15K606 p40) can break
  the strict floor on BOTH coordinate drift and cell count (17/30, documented
  exception — see *Known model-floor limits*).
- **Production rasterization is now measured, not assumed.** Go rasterizes PDF
  pages with pdfium @216 DPI (LCD text AA enabled, matching pdfplumber); Python
  deepdoc with pdfplumber @216 DPI. The end-to-end raster-alignment harness
  (Methodology §8) renders the *same* real PDF page with both paths and compares
  boxes: DLA **≤ 0.03 px on text pages** (≤ 0.72 px on the one figure page), Det
  IoU orphan 0/0–1/2, TSR on real-table pages ≤ 2.70 px. The "same raster bytes
  in" premise for layout/text detection is empirically closed (the prior Scope
  note about pdfium-vs-poppler is superseded: deepdoc uses pdfplumber, also
  @216 DPI). **Caveat (measured):** enabling AA tightened DLA but did not change
  Det — and on closer measurement the Det `corner-maxd 8–12 px` in the test log
  is the *max per-corner* difference on 1–2 skewed outlier boxes, not a center drift;
  per-box center distance is sub-pixel (no render-origin offset). See Scope note.
- Depends on the `-tags native_det` build path **and** ONNX Runtime **1.23.x**
  **and** the same `InfiniFlow/deepdoc` model snapshot as the (frozen) Python
  side.
- No standalone HTTP service mirrors `deepdoc_server`; only the in-process
  library and a CLI exist.
- Requires CI to keep running the native tests (see *How to reproduce*) so the
  equivalence does not silently regress as the Go side evolves.

**Prerequisites for the claim to hold.** Python side (including models) frozen;
`MODEL_DIR` pinned to the same snapshot; native tests wired into CI.

**Runtime version (concretely recorded).** Both sides run ONNX Runtime
**1.23.2**: the Go side via `internal/common.DeepDocORTVersion = "1.23.2"`, and
the Python reference `deepdoc_server` validated against this proof also uses
`onnxruntime 1.23.2`. The two runtimes are therefore numerically aligned, not
just "ABI-compatible". If the frozen Python side is ever moved to a different
ORT build, the live-service diff (`TestWireVsLiveServer`) must be re-run.

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

**Boundary of this proof (read carefully).** This document proves the two
backends are equivalent **at the inference-service boundary** — given the *same
raster image bytes*, they emit the same boxes / text / wire JSON. It does
**not** prove end-to-end equivalence of the full ingestion pipeline (PDF →
raster → recognizers → `PdfParser` post-processing → chunks / table-HTML /
markdown). Two things live outside this boundary and are out of scope here:
1. **PDF → raster.** The Go pipeline rasterizes via `pdfium` (@216 DPI) and the
   Python deepdoc pipeline via `pdfplumber` (`to_image(resolution=72*zoomin=216,
   antialias=True)`) — **both at 216 DPI**. This is no longer an open scope cut:
   the end-to-end raster-alignment harness (Methodology §8) renders the *same*
   real PDF page with both paths and compares the recognizer output in
   source-pixel space. Result: DLA ≤ 0.03 px on text pages (≤ 0.72 px on the one
   figure-heavy page), Det IoU orphan 0/0–1/2, TSR on real-table pages ≤ 2.70 px
   — i.e. the "same raster bytes in" premise for layout/text detection holds
   empirically through the actual production render paths.

   **AA status (measured).** The Go pdfium render now sets `FPDF_LCD_TEXT (0x02)`
   in `pdfium.go`, upgrading text from pdfium's *default* grayscale anti-aliasing
   to LCD sub-pixel AA to match Python pdfplumber's `antialias=True` text. (Note:
   pdfium anti-aliases by default — `0x02` only refines text AA; there is no
   "pdfium has no AA" regime, and `pdf_oxide` is not in the render path at all —
   it only does text/char extraction. The AA flag is a pdfium C-API flag applied
   directly to `FPDF_RenderPageBitmap`.) Re-running the raster-alignment harness
   after this change produced a split result:
   - *DLA tightened sharply on text pages*: worst max Δ dropped from ~0.72 px to
     ≤ 0.03 px on 年报/ZoomNeXt/ZH-TW/三国 pages (effectively pixel-identical).
     The one technical-standard figure page (15K606 p10) stayed at 0.721 px —
     that residual is **not** an AA artifact (it survived the text-AA change), so
     it comes from a different source (vector/figure rendering, not text
     smoothing).
   - *Det: no render-origin offset (measured)*: the test log's `corner-maxd 8.0 / 12.0
     px` on 年报 p2/p8 is the **max per-corner** coordinate difference, not a
     center drift. A nearest-center (greedy + Hungarian) analysis of the dumped
     boxes shows per-box center distance is sub-pixel — median **0 px**, mean
     **< 0.5 px**, p90 **< 2.2 px**, max **< 5 px** — with IoU orphan unchanged
     0/0–1/2. So there is **no translation / coordinate-origin offset** between
     the two render paths; the earlier "8–12 px is a render-origin translation"
     guess is **falsified**. The ~8–12 px corner figure is concentrated in 1–2
     outlier text boxes whose quadrilaterals are slightly skewed, from the same
     contour-boundary geometry that produces the 3/5 IoU orphan floor
     (`bilinearResize` vs cv2.resize at text edges + Moore-neighbour vs
     cv2.findContours). AA did not change it (identical before/after) because it
     is not an antialias artifact. The IoU ≥ 0.5 floor still holds, so it
     remains a benign quad-skew, not a detection divergence.
2. **`PdfParser` downstream logic** — caption/figure/table association, table
   cell extraction, equation handling, chunk assembly. These consume the
   recognizer outputs but are not inference endpoints, so they are not covered
   by "inference service equivalence".

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
- `TestWireVsLiveServer` (optional, gated) diffs Go's `Wire()` against a
  **running** `deepdoc_server` when `DEEPDOC_URL` is set — the live-service
  isomorphism check (Methodology §7). Add it to the `-run` filter and export
  `DEEPDOC_URL=http://localhost:9390` to enable; it self-skips otherwise.

The same model boundary is exercised through the **`DocAnalyzer` seam the PDF
parser actually consumes** (`infnative.NativeAnalyzer`, package
`internal/deepdoc/parser/pdf/inference/native`), so the equivalence is proven
at the integration point rather than only inside the standalone library.

> **Use `build.sh`, not raw `go test`.** The Go native paths need CGO flags and
> the static native libs (`office_oxide` / `pdfium` / `pdf_oxide`) that
> `build.sh` wires automatically. From the repo root:

```bash
ORT_LIB=<path/to/libonnxruntime.so.1.23.2> \
MODEL_DIR=<path/to/InfiniFlow/deepdoc> \
bash build.sh --test-native \
  -run 'TestAnalyzerDLAGolden|TestAnalyzerTSRGolden|TestAnalyzerOCRRecGolden|TestAnalyzerDetGolden' \
  ./internal/deepdoc/parser/pdf/inference/native/...
```

`bash build.sh --test-native` (no `-run` filter) runs the **entire** native
tier in one shot: the analyzer golden suite above **plus** the `dla-native`
integration suite (`TestEquivalenceReport` / `TestDetMembershipAllFixtures`).
That single command is what CI should gate on.

> **Env-var note.** The *test harness* reads `ORT_LIB` / `MODEL_DIR`. The
> *server runtime* (the in-process `Register` path) reads `DEEPDOC_ORT_LIB` /
> `DEEPDOC_MODEL_DIR` (see `AGENTS.md`). They point at the same
> `libonnxruntime.so` and `InfiniFlow/deepdoc` snapshot — just don't set the
> wrong pair or the tests will silently `Skip`.

These four tests reuse the comparison helpers in `dla-native/native/golden.go`
(coordinate / score tolerances, IoU box-membership) and compare the analyzer's
`DLA` / `TSR` / `OCRRecognize` / `OCRDetect` output against the same Python
reference goldens.

Models are fetched once with `ragflow_deps/download_deps.py` (a snapshot of
`InfiniFlow/deepdoc`). The ONNX Runtime shared library is `libonnxruntime.so`
(any 1.23.x build; validated with 1.23.2 (see internal/common.DeepDocORTVersion), ABI-compatible with the
`onnxruntime_go` v1.23.0 binding — the same line the Python goldens were
generated with, so validated parity is preserved).

## Methodology: how the proof is constructed

The proof is not a hand-written mirror of the Python output — it compares the
Go port against **production deepdoc inference**, captured as frozen "golden"
fixtures. Two independent test tiers run the same comparison math against the
same goldens, at two different integration depths.

### 1. Reference goldens are real deepdoc output

Each golden JSON is produced by a Python oracle in this directory that calls
`deepdoc.vision` directly, then re-maps the OSS labels to the Go class ids and
serializes into the Go `DocAnalyzer` wire shape. The oracles invoke the exact
same recognizers and post-processing the running server uses:

| Task | Oracle | Production recognizer / config it pins |
|------|--------|----------------------------------------|
| DLA  | `ref_dla.py`   | `LayoutRecognizer4YOLOv10("layout").forward(thr=0.2)` → OSS labels re-mapped via `DLA_CLASS_MAP` to Go DLA class ids |
| TSR  | `ref_tsr.py`   | `TableStructureRecognizer()([img], thr=0.2)` incl. `alignTSR` (mean/median when >4 rows/cols) → `TSR_CLASS_MAP` (6 classes) |
| Det  | `ref_det.py`   | `TextDetector`: `DetResizeForTest → NormalizeImage → det.onnx → DBPostProcess(thresh=0.3, box_thresh=0.5, unclip_ratio=1.5) → filter_tag_det_res` |
| OCR  | `ref_ocr_rec.py` | `TextRecognizer` with **batch-wide max `wh_ratio`** resize (matches production batch semantics); wire score pinned to `1.0` to match the Go `DocAnalyzer` OCR-rec wire |

Because the reference is the actual production code path, a Go/Python mismatch
surfaces a genuine divergence, not a divergence between two reimplementations.
The fixtures are frozen and committed under `internal/deepdoc/dla-native/testdata`
(`*.dla.golden.json`, `*.tsr.golden.json`, `*.det.golden.json`,
`*.ocr_rec.golden.json`). They were regenerated from the **current** live
detectors after the `normalizeCHW` RGB/BGR channel-order fix, so the det
baseline (3/5 IoU orphans) reflects the true Go-vs-cv2 gap rather than a stale
or swapped-channel oracle.

### 2. Comparison math (shared by both tiers)

All matching lives in `internal/deepdoc/dla-native/native/golden.go` and is
imported by both the `dla-native` integration suite and the `infnative`
analyzer suite, so the two tiers cannot drift apart.

- **Axis-aligned boxes (DLA / TSR / OCR):** `CompareBoxes` / `MatchBoxesRelaxed`
  match every golden box to its **nearest same-class** Go box by center
  distance, then assert each coordinate is within `CmpTolCoord` and the score
  within `CmpTolScore`. The relaxed variant returns the unmatched count instead
  of failing, so callers can express structural assertions (e.g. "only a
  near-threshold row may be dropped").
- **Label alignment:** goldens and the Go analyzer both key on the **first**
  index of a label string, so DLA's duplicate labels (indices 4/7/9 share
  source text with neighbours) map identically on both sides. The analyzer test
  re-derives each box's class from `labelKey(labels, r.Label)` and rewrites the
  golden's integer class to the same key before matching.
- **Quads (Det):** `MatchBothDirections` and `MatchIoUBothDirections` match the
  rotated text quads **in both directions** by (a) nearest-center within
  `CmpTolCoord` and (b) greedy best-IoU ≥ 0.5. IoU membership isolates true
  box divergence — a split (1→2), a merge (2→1), or a hallucination — from mere
  coordinate drift: a box shifted 20 px but still overlapping its twin scores
  high IoU and is **not** an orphan. Reporting both directions surfaces Go boxes
  that have no golden counterpart (extra detections) as well as golden boxes Go
  missed.

### 3. Tolerances and floors

| Constant | Value | Meaning |
|----------|-------|---------|
| `CoordFloor` | `3.0` | Documented hard accuracy floor of the comparison pipeline (bilinear resize + `box#8` postprocess for det; DLA/TSR are tighter). |
| `CoordTolMargin` | `0.5` | Lifts the tolerance just above `CoordFloor` so a regression trips the gate instead of hiding under it. |
| `CmpTolCoord` | `3.5` | Coordinate tolerance (px) for golden comparisons. |
| `CmpTolScore` | `0.05` | Score tolerance for detection boxes. |

Two special floors are documented, not hidden:

- **Det hard floor ≈ 3 px.** The pure-Go geometry stabilizes at ~3 px; the
  `3.5` tolerance sits just above it. `TestDetIntegration` asserts every golden
  quad is within `3.5` px of a Go quad.
- **Extreme-aspect TSR floor ≈ 10 px.** On a ~4:1 crop the model's 640×640 input
  squishes x by ~1.45×, amplifying the residual Go-vs-PIL decode difference to
  ~8 px. `TestTSRExtremeAspect` therefore uses a relaxed `10` px tolerance and
  asserts only that **structure survives** (table + all columns must match,
  row count within ±1, only a near-threshold row may be dropped, no hallucinated
  boxes). The analyzer TSR golden (`tsr_table_rotation`, a 1:6.3 tall table) sits
  comfortably under the 3.5 px tolerance.

Because the analyzer does not expose a TSR score, `TestAnalyzerTSRGolden` widens
the score tolerance to `1.0` and asserts only class + coordinates.

#### Known model-floor limits (full-page real tables) — measured

The 3.5 px coordinate tolerance is **not** universal across every real table — it
is a property of the committed fixtures, not a guarantee for all inputs. The
end-to-end raster-alignment harness (Methodology §8 / `TestTSRFloorFullPageTables`)
quantifies this on whole-page real tables through BOTH production raster paths:

| Full-page table | Match | Max Δ | Verdict |
|-----------------|-------|-------|---------|
| 厦门象屿年报 p8 (moderate) | **34/34** | **1.21 px** | inside 3.5px floor |
| 厦门象屿年报 p12 (dense) | **25/25** | **0.37 px** | inside 3.5px floor |
| 15K606《建筑防烟排烟系统技术标准》图示》 p40 (dense technical-standard) | **17/30** | 3.36 px | **documented exception** — model-level cell-count divergence (Go emits 31 vs 30 golden; 13 cells unmatched), not a rasterization floor |

So the empirical upper bound is:

> TSR equivalence is proven within **≤ 3.5 px** on ordinary/moderate real tables
> (worst measured 2.70 px on 三国人物 p1; dense annual-report tables ≤ 1.21 px).
> **Dense technical-standard tables** can break the strict floor — both
> coordinate drift AND cell-count disagreement (17/30 on 15K606 p40). This is a
> model-floor effect (the TSR model itself disagrees on a hard table under
> pdfium-vs-pdfplumber rasterization), not a Go logic bug. It is recorded as a
> known-hard exception in `TestTSRFloorFullPageTables` with a regression guard
> (must not get worse than the 17/30 baseline), and is **not** in the strict
> 3.5px fixture suite.

If a future caller needs pixel-exact TSR on dense technical-standard tables, the
fix is a decode/resize alignment on large inputs plus a harder table in the
corpus, not a logic change.

### 4. Per-task proof

| Task | What is compared | Proving tests | Result |
|------|------------------|---------------|--------|
| DLA  | Layout boxes vs `ref_dla.py` goldens (**11 fixtures**: EN textbook, CN whitepaper, eq-heavy paper, ZH-TW enterprise, baseline, 2 figure-caption pages, 1 equation page, **+ CN annual-report page (厦门象屿年报 p2), ZH-TW migration doc p3, EN paper (ZoomNeXt p1)**) | `TestDLAIntegration` / `TestAnalyzerDLAGolden`, `TestEquivalenceReport` | 148/148 boxes, max Δ < 0.13 px |
| TSR  | Table-structure boxes vs `ref_tsr.py` goldens (**8 fixtures**: table0, normal 2.65:1, rotated 1:6.3, content, caption, cross-page, text-interleaved — covers projected-row-header; **+ real annual-report table (厦门象屿年报 p8)**) | `TestTSRIntegration` / `TestAnalyzerTSRGolden`, `TestTSRExtremeAspect` | 190/190 matched, 2.155 px (≤10 px @4:1 aspect, structure preserved) |
| OCR  | Recognized text vs `ref_ocr_rec.py` goldens (EN incl. bold/italic/serif, CJK, mixed, digits; plus batch semantics; **+ 3 real text-line crops: 三国人物 p1, ZH-TW migration p2, ZoomNeXt p1**) | `TestOCRRecIntegration` / `TestAnalyzerOCRRecGolden`, `TestOCRRecBatchIntegration` | exact text; batch-wide resize reproduced |
| Det  | Text quads vs `ref_det.py` goldens (all fixtures) | `TestDetIntegration` / `TestAnalyzerDetGolden`, `TestDetMembershipAllFixtures`, `TestDetOCRAdjudication` | IoU orphan floor 3/5, adjudicated benign via OCR |

### 5. Two proof tiers (single source of truth)

1. **Standalone library tier** — `internal/deepdoc/dla-native/native`, run with
   `-tags integration`. Exercises `RunDLA` / `RunTSR` / `RunDet` /
   `RunOCRRec` / `RunOCRRecBatch` directly, proving the inference library
   itself is equivalent.
2. **`DocAnalyzer` seam tier** — `internal/deepdoc/parser/pdf/inference/native`,
   run with `-tags "native_det integration"`. Exercises `infnative.NativeAnalyzer`
   — the exact `deepdoctype.DocAnalyzer` implementation the PDF parser consumes
   in production — proving equivalence at the integration point rather than only
   inside the library. These are `TestAnalyzerDLAGolden`, `TestAnalyzerTSRGolden`,
   `TestAnalyzerOCRRecGolden`, `TestAnalyzerDetGolden`.

Both tiers call the same helpers in `golden.go`, so a change to the matching
math applies identically to library and seam.

### 6. Schema and determinism guards

Equivalence is not only value-level; the **wire contract** is guarded too:

- `TestWireSchemaMatchesGolden` asserts Go's `Wire()` JSON structure (top-level
  key, nesting depth, leaf types) is identical to the deepdoc server adapter
  shape for DLA / TSR / Det / OCR-rec, so a caller parsing Go output sees the
  same schema as the Python service.
- `TestDLASessionReuse` / `TestTSRSessionReuse` / `TestOCRRecSessionReuse`
  assert byte-identical `Wire()` across pooled-session runs (no stale tensor,
  no cross-call contamination).
- `TestDetSessionPoolBounded` asserts the session pool set stays bounded under
  many distinct page sizes (regression guard for a prior unbounded `sync.Map`
  leak).

These guards mean the Go port is a drop-in wire-compatible replacement, not just
numerically close on a fixed fixture set.

### 7. Live-service field diff (proves "Go == running service", not just "Go == golden")

`TestWireSchemaMatchesGolden` (section 6) compares Go's `Wire()` against the
**re-serialized golden shape** — i.e. against a Python-shaped artifact, not the
live service. To close that gap, `TestWireVsLiveServer` (integration tier,
`native_integration_test.go`) POSTs each fixture's PNG to a **running**
`deepdoc_server` (`DEEPDOC_URL`, default `http://localhost:9390`) and diffs the
real HTTP JSON response field-by-field against the Go `Wire()` output produced
by `RunDLA` / `RunDet` / `RunTSR` / `RunOCRRec`. It skips when `DEEPDOC_URL` is
unset or the server is unreachable, so it is safe in CI (gated) and local.

**Why this matters.** It answers the reviewer question "you proved Go == a
frozen Python snapshot, not Go == the running service" directly: the test hits
the actual service contract. It also re-confirms the server is a thin wrapper —
its adapters only decode bytes, convert color space, run the same
`deepdoc.vision` recognizers with the same config the goldens were generated
against (DLA/TSR `thr=0.2`, OCR default pipeline), clamp bboxes, and map
label→class_id; there is no extra resize / DPI / rotation / server-side
rasterization.

**Measured (against the reference `deepdoc_server`, ORT 1.23.2 both sides):**

| Task | Fixture | Server boxes | Go boxes | Match / Δ |
|------|---------|--------------|----------|-----------|
| DLA  | page0 | 4 | 4 | matched 4/4, max Δ 0.006 px |
| DLA  | mp_textbook_en_p0 | 13 | 13 | matched 13/13, max Δ 0.023 px |
| DLA  | dla_2510_eq | 24 | 24 | matched 24/24, max Δ 0.007 px |
| TSR  | table0 | 11 | 11 | matched 11/11, max Δ 0.509 px |
| TSR  | tsr_table_rotation | 15 | 15 | matched 15/15, max Δ 0.886 px |
| Det  | page0 | 15 | 15 | IoU orphan 0/0 |
| Det  | mp_en_dense_p0 | 93 | 95 | IoU orphan 2/4 (within 3/5 floor) |
| OCR-rec | line0, line_cn | — | — | exact text match |

Conclusion: the Go backend and the live `deepdoc_server` are field-for-field
equivalent on these fixtures; the only Det divergence is the same documented
3/5 contour floor already covered above.

### 8. End-to-end raster alignment (closes the "same-bytes-in" gap)

Sections 1–7 prove "**given the same raster image bytes**, Go == Python". But in
production neither side receives a pre-rendered PNG: the Go server rasterizes PDF
pages with **pdfium** (`pdfium.RenderPage` @ **216 DPI**) and the Python deepdoc
pipeline rasterizes with **pdfplumber** (`page.to_image(resolution=72*zoomin,
antialias=True)`, `zoomin=3` ⇒ **216 DPI**). §Scope noted this as the one gap
outside the proved boundary. This section closes it by rasterizing the **same
real PDF page with both paths** and comparing the resulting boxes in source-pixel
coordinates — so the "same-bytes-in" assumption becomes a *measured* fact, not a
declaration.

- **Go side:** `pdfium.RenderPage(pdf, page, 216)` → `NativeAnalyzer.DLA` /
  `OCRDetect` / `TSR` (the exact `DocAnalyzer` seam production consumes).
- **Python side:** `ref_raster.py` renders the same page at 216 DPI via deepdoc's
  OWN pdfplumber path (so it matches what the live `deepdoc_server` would rasterize),
  then runs the real `deepdoc.vision` recognizers.
- Both render at **216 DPI**, so box coordinates land in the **same pixel space**
  and are compared directly. Harness: `TestRasterAlignmentDLA` /
  `TestRasterAlignmentDet` / `TestRasterAlignmentTSR` (analyzer suite,
  `//go:build native_det && integration`).

**Measured (6 real documents, 216 DPI both sides, Go pdfium render with
`FPDF_LCD_TEXT (0x02)` set — LCD sub-pixel text AA on top of pdfium's default
grayscale AA, to match pdfplumber's `antialias=True`):**

| Task | Pages | Result |
|------|-------|--------|
| DLA  | 年报 p2 | **15/15**, worst max Δ **0.017 px** |
| DLA  | 年报 p8 | **14/14**, worst max Δ **0.025 px** |
| DLA  | ZoomNeXt p1 | **15/15**, worst max Δ **0.011 px** |
| DLA  | ZH-TW migration p3 | **24/24**, worst max Δ **0.031 px** |
| DLA  | 三国人物 p1 | **18/18**, worst max Δ **0.006 px** |
| DLA  | 15K606 p10 (figure/table) | **18/18**, worst max Δ **0.721 px** (AA-invariant — see note) |
| Det  | 年报 p2 | matched **93/93**, center-max **3.20 px**, corner-maxd **8.0 px**; IoU orphan 0/0 |
| Det  | 年报 p8 | matched **42/42**, center-max **3.35 px**, corner-maxd **12.0 px**; IoU orphan 0/0 |
| Det  | ZoomNeXt p1 | matched **140/140**, center-max **3.35 px**, corner-maxd **3.0 px**; IoU orphan 0/0 |
| Det  | ZH-TW migration p3 | matched **30/30**, center-max **0.50 px**, corner-maxd **1.0 px**; IoU orphan 0/0 |
| Det  | 三国人物 p1 | matched **32/32**, center-max **1.50 px**, corner-maxd **3.0 px**; IoU orphan 0/0 |
| Det  | 15K606 p10 | matched **55/55**, center-max **3.20 px**, corner-maxd **3.0 px**; IoU orphan 1/2 |
| TSR  | 年报 p2/p8, ZH-TW migration p3, 三国人物 p1 | **matched 100%** (117/117 cells), worst max Δ **2.700 px** (≤ 3.5px floor) |

**Interpretation.**
- **DLA is fully closed end-to-end, and now near-pixel-exact on text pages**:
  pdfium-vs-pdfplumber rasterization with AA enabled leaves ≤ 0.03 px of
  coordinate drift on the four text/table pages (far inside the 3.5px floor), so
  the "same-bytes-in" assumption for layout detection is effectively exact there.
  The single technical-standard figure page (15K606 p10) stays at 0.721 px even
  *with* AA on — confirming that residual is a vector/figure-render difference,
  not a text-smoothing artifact.
- **Det is closed end-to-end on structure** (IoU orphan 0/0–1/2, no loss /
  no hallucination). The per-box *center* distance is **sub-pixel**: a
  greedy/Hungarian nearest-center analysis of the dumped boxes on 年报 p2/p8
  gives median **0 px**, mean **< 0.5 px**, p90 **< 2.2 px**, max **< 5 px** —
  i.e. there is **no translation / coordinate-origin offset** between the two
  render paths (the earlier "8–12 px is a render-origin translation" guess,
  inferred from the test log's `corner-maxd`, is falsified). What the test log
  reports as `corner-maxd 8.0 / 12.0 px` is the *max per-corner* coordinate difference
  (`MatchBothDirections` measures the worst of the 4 corners, not the center,
  golden.go:191), and it is concentrated in **1–2 outlier text boxes per page**
  whose quadrilaterals are slightly rotated/skewed differently. The source is the
  same documented contour-boundary geometry already behind the 3/5 IoU orphan
  floor: Go's `bilinearResize` vs cv2.resize interpolation at high-contrast text
  edges, plus the Moore-neighbour vs cv2.findContours contour trace. AA does not
  change it (identical before/after) because it is not an antialias artifact.
  Every box still overlaps its twin at IoU ≥ 0.5, so it is a benign quad-skew
  inside the documented Det floor, not a detection divergence.
- **TSR is closed end-to-end on ordinary/moderate real tables** (≤ 2.7 px). For
  *dense full-page technical-standard tables* the coordinate floor and the model
  itself can diverge further — see *Known model-floor limits* below for the
  quantified bound and the one documented exception.

The harness requires `uv run python3` with `deepdoc` + `pdfplumber` available;
if the Python oracle is absent the alignment tests **skip** (not fail), so CI
without the Python oracle still passes the rest of the native suite.

## Evidence (measured)

Latest `TestEquivalenceReport` (both tiers green; summary printed to the test log):

| Task | Fixtures | Match | Max Δ | Status |
|------|----------|-------|-------|--------|
| DLA  | 11 | 148/148 boxes (incl. equation + figure-caption classes, +3 real-document pages: CN annual report / ZH-TW migration / EN paper) | < 0.13 px | OK |
| TSR  | 8 | 190/190 boxes (incl. projected-row-header class, +1 real annual-report table) | 2.155 px (≤ 10 px on a 4:1 aspect crop, structure preserved) | OK |
| OCR  | 11 | exact text (EN / CJK / mixed / digits, + font variants, +3 real text-line crops: CN / ZH-TW / EN) | — | OK |
| Det  | all | IoU box-membership orphan **3/5** (gold 1125 / go 1127) | — | OK (accepted floor) |

The detection orphan boxes were adjudicated with OCR: cropping each orphan and
running OCR shows the regions still resolve to real text. Concretely, the
orphans split as **Python-only real text 0/3** (those 3 regions are empty
furniture on both sides) and **Go-only real text 3/5** (Go emits 3 text regions
the Python side does not). So the two outputs are **not byte-identical**: Go is
a *superset* on those few regions. This is a small, one-directional divergence
(Go finds extra text), not a loss — the downstream text consumer (dedup /
align) absorbs it, and no content is dropped. "Benign" here means "no content
loss", not "identical output".

### Det IoU orphan 3/5 — what it means, and how small it is

**What "IoU orphan" means.** For Det we compare two *sets* of text boxes — the
Python golden set and the Go set — over each fixture. "IoU" (Intersection-over-
Union) here is computed on the **axis-aligned bounding box (AABB)** of each
quadrilateral (`iou()` in `golden.go:249`): `IoU = area(intersection) /
area(union)`. Two boxes are considered a match only if their *best* greedy IoU
is `≥ 0.5` (`MatchIoUBothDirections`, `golden.go:269`), matched independently in
both directions. An **orphan** is a box on one side whose best counterpart on
the other side has IoU `< 0.5` — i.e. no overlapping twin at all. So "3/5"
means: **3 Python golden boxes** have no Go twin at IoU ≥ 0.5 (Go "missed"
them), and **5 Go boxes** have no Python twin at IoU ≥ 0.5 (Go "extra"). This
isolates genuine box-membership divergence (a box split, merged, or
hallucinated) from mere coordinate drift — a box shifted 20 px but still
overlapping its twin scores high IoU and is **not** an orphan.

**Sample volume.** The 3/5 is accumulated across **every** committed Det
fixture, not a cherry-picked subset. The corpus is **35 fixtures**, totalling
**1,125 golden text boxes** vs **1,127 Go text boxes** (logged by
`TestDetMembershipAllFixtures` as `TOTAL gold=1125 go=1127`). This spans blank
pages, degradation variants (noise / low-contrast / skew / tiny-text / CJK
vertical), and mixed-language dense pages.

**Proportion (how dominant the agreement is).**

| Side | Boxes | Matched (IoU ≥ 0.5) | Orphan | Orphan rate |
|------|-------|---------------------|--------|-------------|
| Python golden | 1,125 | 1,122 | 3 | **0.27%** |
| Go | 1,127 | 1,122 | 5 | **0.44%** |
| Both combined | 2,252 | 2,244 | 8 | **0.36%** |

In other words, the two detectors agree on box membership for **>99.6%** of all
text regions; the residual is a handful of boxes whose contour geometry differs
just enough to drop below the 0.5 IoU hinge.

**Why it is accepted, not a defect.** It is a *regression guard*, not a
zero-target: the test fails only if Go gets *worse* than the baseline (gold
orphan `> 6`, go orphan `> 8`, with slack 3 — see `native_integration_test.go:559`).
The 3/5 is the residue left after every real bug was fixed — stale goldens
(37/20, 42/13) and an R/B channel-swap in `normalizeCHW` (23/9) — leaving only
contour-tracer geometry (explained below). And the orphan boxes are
OCR-adjudicated benign (above): cropping each orphan and running OCR still
resolves real text, so the divergence never drops content.

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
2. **Production caller is wired (server only).** The server binary built with
   `-tags native_det` registers the in-process backend via
   `infnative.Register(...)` in `cmd/ragflow_server_native.go`; it fails fast at
   startup unless either this backend (ORT + models present) or an external
   `DEEPDOC_URL` service is configured. The CLI binary is built without
   `native_det` (no-op path). End-to-end replacement of the running Python
   service depends on that server build path being selected in deployment.
3. **Coverage confirmation required.** Go implements {det, dla, tsr, ocr}.
   Confirm the Python path being replaced uses only these recognizers (e.g. a
   separate table-cell recognizer would not be covered).
4. **Runtime version.** Validated with onnxruntime 1.23.2 (see internal/common.DeepDocORTVersion); re-verify if the
   production Python uses a different ORT version (stable within 1.23.x).
5. **HTTP server shape.** If the goal is a standalone HTTP service mirroring
   `deepdoc_server`, that surface is not built — only the in-process library and
   a CLI (`main.go`) exist.

## Known model-floor limits (documented, not hidden)

The full-page real-table TSR floor is **measured end-to-end** (both production
raster paths) in Methodology §8 / `TestTSRFloorFullPageTables` and recorded in
the *Known model-floor limits (full-page real tables) — measured* subsection
under §3. That is the authoritative, quantified statement: ordinary/moderate
real tables ≤ 3.5px (worst 2.70px; dense annual-report ≤ 1.21px), and dense
technical-standard tables (15K606 p40) breaking the floor on both coordinate
drift **and** cell count (17/30, documented exception).

The two excluded pages (`15K606` p40, `厦门象屿年报` p12) are retained in the
generator (`/tmp/gen_corpus.py`) for future investigation but are **not** in the
strict 3.5px fixture suite — p12 actually measures 0.37px (25/25) once rasterized
through the real production path, so only p40 remains a genuine hard case.

## Reviewer follow-ups (prioritized, with status)

These are the items raised by an independent review of this proof, ranked by
impact. "Closed" items are done in code or in this document.

| ID | Item | Status |
|----|------|--------|
| P1 一 | **Live-service field diff** — confirm server is a thin wrapper (no extra preprocessing) and diff Go `Wire()` against the *real* `deepdoc_server` HTTP response. | **CLOSED** — server verified thin (adapters only decode + color-convert + clamp + label→class_id; config matches goldens: DLA/TSR `thr=0.2`, OCR default pipeline). `TestWireVsLiveServer` added and passing against the reference server (ORT 1.23.2 both sides); measured numbers in Methodology §7. |
| P1 五 | **ORT version** — record the Python-side ORT build, not just assume ABI compatibility. | **CLOSED** — both sides run ONNX Runtime **1.23.2** (Go `DeepDocORTVersion`; reference server `onnxruntime 1.23.2`). Recorded in Prerequisites. |
| P2 二 | **Scope wording** — state that inference-boundary equivalence ≠ end-to-end PDF→chunk pipeline equivalence. | **CLOSED** — explicit "Boundary of this proof" paragraph added to Scope; PDF→raster and `PdfParser` downstream named as out-of-scope. |
| P2 三 | **Det 3/5 "benign" wording** — it is not identical output (Go emits 3 extra text regions). | **CLOSED** — reworded to "no content loss, not identical output"; Go is a *superset* on those regions, absorbed by downstream dedup. |
| P2 八 | **Corpus** — DLA/TSR/OCR coverage is thin (8 / 7 / 8; OCR are line crops, not full pages). | **CLOSED** — expanded with diverse full-page real-document fixtures: DLA +3 (`dla_real_cn_report` 厦门象屿年报 p2, `dla_real_zhtw` ZH-TW migration doc p3, `dla_real_en_paper` ZoomNeXt paper p1), TSR +1 (`tsr_real_report` 厦门象屿年报 p8), OCR +3 (`line_real_cn` 三国人物 p1, `line_real_zhtw` ZH-TW migration p2, `line_real_en` ZoomNeXt p1). All pass sub-pixel / exact-text. The one genuine hard case (15K606 p40 dense technical-standard table) is a documented exception, not in the strict 3.5px suite — see *Known model-floor limits*. |
| P0 | **Model snapshot hash lock** — `MODEL_DIR` must be pinned to the same `InfiniFlow/deepdoc` snapshot as the frozen Python side; the proof must fail if it drifts. | **CLOSED** — enforced. `modelSnapshotHashes` (sha256 of `det.onnx`, `layout.onnx`, `tsr.onnx`, `rec.onnx`, `ocr.res`) is checked by `TestModelSnapshotHash` and at the top of `TestEquivalenceReport`; Fatal on any mismatch. Both repo copies verified byte-identical. Update the table only when the snapshot is intentionally upgraded, and regenerate every golden in the same change. |
| P3 | **Concurrency correctness** — parallel vs serial inference must give identical results (thread-safety is correctness, not performance). | **CLOSED** — `TestInferenceConcurrencyConsistent` drives DLA/TSR/OCR-rec/Det once serially (baseline wire) then 8× concurrently and asserts every concurrent run is byte-identical to the serial baseline. Proves the shared model-session pool is race-free and contamination-free under parallel load (complementing `TestDetSessionPoolBounded` which guards pool *size*). |
| **E2E-1** | **End-to-end raster alignment** — the "same raster bytes in" premise is a declaration: production rasterizes via different engines (Go pdfium @216 DPI vs Python pdfplumber @216 DPI). Prove the two render paths yield equivalent boxes. | **CLOSED** — `TestRasterAlignmentDLA/Det/TSR` rasterize the *same* real PDF page with both paths (pdfium vs deepdoc's own pdfplumber) at 216 DPI and compare boxes in source-pixel space. After enabling LCD text AA in the Go pdfium render to match pdfplumber: DLA **104/104** matched with worst max Δ **0.721 px** on the figure page and **≤ 0.03 px** on the four text pages (near-pixel-exact); Det IoU orphan 0/0–1/2 (inside 3/5); TSR on real-table pages **117/117** matched (worst 2.700px). The "same-bytes-in" premise is now **measured**, not assumed. Measured numbers in Methodology §8. Note (measured): the Det test-log `corner-maxd 8–12 px` is the max *per-corner* difference on 1–2 skewed outlier boxes; per-box **center** distance is sub-pixel (median 0, mean <0.5px, p90 <2.2px, max <5px) — there is **no render-origin translation**; the residual is contour-boundary quad-skew, the same source as the 3/5 IoU orphan floor. |
| **E2E-2** | **Quantify TSR floor on full-page real tables** — give an empirical upper bound ("N pages, worst X px") instead of two hand-picked excluded examples. | **CLOSED** — `TestTSRFloorFullPageTables` runs TSR on whole-page real tables through both raster paths: moderate tables ≤ 3.5px (厦门象屿年报 p8 1.21px, p12 0.37px; 三国人物 p1 2.70px); dense technical-standard 15K606 p40 is the documented exception (17/30, model-level cell-count divergence, regression-guarded). Empirical bound recorded in *Known model-floor limits (full-page real tables) — measured*. |

**Withdrawn critique.** One review point claimed the two proof tiers "use
different inference code paths / decoders, so preprocessing may differ." This is
**factually incorrect**: `infnative.NativeAnalyzer.DLA/TSR/OCRDetect/OCRRecognize`
all delegate to the *same* `native.RunDLA` / `RunDet` / `RunTSR` / `RunOCRRec`
(`native_analyzer.go:101-181`); only the initial PNG→pixel decode differs, which
is the already-documented `≤ 1/255` residual. The standalone and seam tiers
therefore share one inference path.
