# HANDOFF — 架构评审后续 / 接力待办

> **权威 ADR 见 `README.md`**；长会话接力历史见 `HANDOFF.md`（仅保留有效要点）。
> 本文件记录本回合已落地的关键决策与「不要回退」项，避免新会话重复踩坑。

## 1. 本回合已拍板并已落地的决策

- **gocv 双轨收敛 → 纯 Go 单一路径**：删除 `image_gocv.go`/`det_gocv.go`/`dla_gocv.go`，
  `go.mod` 移除 `gocv.io/x/gocv`，CI `go-dla-native-gocv` job 删除，`run.sh`/`build.sh` 的 gocv 分支清理。
  接受 ~3px 地板，换取零 OpenCV/CGO 依赖。
- **det 从生产路径回退，仅作比对工具**：删除 `parser/pdf` 的 `native_det*.go` 四文件，
  移除 `NewParser` 的 `RAGFLOW_NATIVE_DET` opt-in 与 `inferOCRDetect` 原生 seam，删除 CI `go-native-det` job。
  det 现与 DLA/TSR/OCR-rec 一致，只作漂移比对。
- **比对工具内部解码格式无关**：`native.Decode` 改为 `image.Decode`（自动识别 jpeg/png），与 Python(PIL/cv2) 对齐。
- **夹具由 JPG 迁 PNG**：47 个夹具无损转码，像素逐字节等价，golden 不变，对齐生产 `EncodePNG` wire。
- **解码安全上限**：`Decode` 对解码后栅格做尺寸/像素上限校验（`maxImageDim=16384` / `maxImagePixels=100MP`），防解压炸弹。

## 2. 关键事实（避免重复踩坑）

1. **3px 坐标地板**：来自 `bilinearResize` + `box#8` 后处理，与输入格式无关，已接受。
2. **确定性 box 排序**：`dlaPostprocess`/`tsrPostprocess` 对 `Boxes` 按 class→坐标→score 排序，
   消除 Go `map` 迭代非确定性导致的 `Wire()` 抖动（session-reuse 测试依赖此）。**不要回退**。
3. **有界 session 池**：`det` 输入尺寸可变 → 原无界 `sync.Map` 会泄漏，已改为有界 LRU
   （`detMaxShapePools`×`detShapePoolCap`）；`DLA/TSR/OCR-rec` 固定 shape 键塌缩为 modelDir。
4. **dla-native 是嵌套 module**：必须 `cd internal/deepdoc/dla-native` 后 `go build -tags integration ./native/`；
   主模块直接 build 该路径会报 “main module does not contain package”。
5. **`python-drift` job 需 `PYTHONPATH=仓库根`**：`ref_det.py` 做 `from deepdoc.vision.ocr import ...`，
   而 `uv sync` 不把仓库根 `deepdoc` 装进 venv（修复已在分支，main 上仍缺此修复）。

## 3. 不要回退 / 不要重做

- 确定性 box 排序（dla.go/tsr.go）。
- 有界 LRU session 池（det_core.go）。
- `TSR_RAW_DUMP` 调试块清理（tsr.go）。
- 子进程隔离机制（已被证伪）。
- `clipper_offset.go`（与 pyclipper 0.000px 一致）。
- 3px 地板几何核心（除非明确决定做 cv2 逐位浮点仿真）。

## 4. 分支状态

- 分支 `feat/deepdoc-go-port`（与 `origin/feat/deepdoc-go-port` 同步），**只保留在该 fork feature 分支，
  不合并 `origin/main`，不向 upstream 提 PR**（按用户约定）。
- 本工作的提交历史见 `git log`；详细决策记录在 `README.md`。
