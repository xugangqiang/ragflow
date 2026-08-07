# dla-native — DeepDoc 纯 Go 复刻与漂移比对工具

> 本文件是 `dla-native` 的**权威设计说明（ADR 风格）**。长篇会话接力记录见同目录
> `HANDOFF.md` / `HANDOFF_NEXT.md`（会过期，仅作历史上下文，不以它们为准）。

## 1. 定位与边界

`dla-native` 是一个**独立的验证工具（verification harness）**，不是产品路径：

- 把 DeepDoc 的 Python 推理管线（OCR 文本检测 `det` / 版面检测 `DLA` / 表格结构
  `TSR` / 文本识别 `OCR-rec`）用**纯 Go + ONNX Runtime** 复刻，在 CPU 上运行。
- 与 Python 参考实现（`ref_*.py`，import 仓库根的 `deepdoc`）对比，捕获复刻回归。
- **不接生产**：主模块从不 import `dla-native`；生产侧 `internal/deepdoc` 只是远程
  Python 服务的 HTTP 客户端（仅 DLA 有远端 endpoint，OCR/TSR 为 `ErrNoRemoteEndpoint`
  桩），不做本地 Go 推理。两者职责不同，无功能重叠、无耦合。

它是嵌套 Go module（独立 `go.mod`），把 ONNX Runtime 依赖隔离在内部，不污染主模块
构建图。

## 2. 为何是纯 Go（P3 决策）

曾存在 `gocv` / `nogocv` 双构建：gocv 走 cv2 解码+resize 可达 1:1 parity，纯 Go 有
~3px 地板。已**收敛到纯 Go 单一路径**（删除 `image_gocv.go` / `det_gocv.go` /
`dla_gocv.go`，`go.mod` 移除 `gocv.io/x/gocv`，CI `go-dla-native-gocv` job 删除）。

权衡：放弃 cv2 1:1 parity，换取**零 OpenCV / CGO 依赖**。

## 3. 3px 地板的来源（已知硬下限）

复刻与 Python 参考的最大坐标残差稳定在 **~3px**，来自：

- `bilinearResize`（Go 浮点权重）vs cv2 定点 `INTER_LINEAR` 的实现差；
- `box#8` 后处理里 `minAreaRect` 的轮廓最小外接矩形。

该地板**与输入格式无关**（实测 JPG 与 PNG 均为 3.0px，decode 贡献 ~0，因 PNG 无损
包裹 JPEG 解码像素）。除非单独决定做 cv2 逐位浮点 `minAreaRect` 仿真（需重新引入
OpenCV），否则不动几何核心。

## 4. golden 如何生成 / 漂移门能证明什么

- **golden**：`ref_*.py`（Python oracle）在固定夹具上跑出的输出，冻结为
  `testdata/<stem>.<task>.golden.json`。夹具现为 **PNG**（由 JPG 无损转码，逐像素
  等价，已校验 47/47 零差异），与生产的 `EncodePNG` wire 对齐。
- **`python-drift` job**：重跑 Python oracle 比对 golden → 仅当 **Python 逻辑漂移且
  golden 被重新生成并提交**时才报警。它**不独立证明 Python 正确**（Python 是 trust
  anchor / oracle）。
- **`go-dla-native` job**：跑 Go 复刻比对同一 golden → 捕获 **Go 回归**。

> 结论：漂移门是「Go vs 冻结的 Python 快照」比对。它可靠捕获 Go 侧回归，但**不证明
> Python 侧正确**——这是「re-implement-to-verify」模式的固有属性。

## 5. 测试分层与安全

- **unit**（无 tag）：纯几何/后处理单测（`clipper_offset_test.go` 与 pyclipper 对照
  0px、`minAreaRect` 对照、`image_test.go` 解码上限），`go test ./native/` 默认运行。
- **integration**（`//go:build integration`，需 `ORT_LIB` + `MODEL_DIR`，缺失自 skip）：
  Go vs golden 全组件比对。
- **解码安全**：`Decode` 对解码后栅格做尺寸/像素上限校验（`maxImageDim=16384`、
  `maxImagePixels=100MP`），防御解压炸弹。当前仅跑固定夹具、无生产暴露；若将来接
  不可信输入，此上限即生效。

## 6. 比对容差

坐标容差 = **`coordFloor(3.0) + coordTolMargin(0.5)` = 3.5**，由常量计算而非字面量
（`native_integration_test.go`）。`coordFloor` 即 §3 的 3px 硬地板，`coordTolMargin`
把它抬到地板之上，使**越过地板的回归会触发门**、而非藏在地板下。

因容差由 `coordFloor` 派生，将来调整地板时容差**自动跟随**，无需手动同步。门只能捕获
**>3px（地板）** 的回归，对「防大破坏」足够；细微回归（<地板）本身不可分，属工具设计
的灵敏度下限，非缺陷。
