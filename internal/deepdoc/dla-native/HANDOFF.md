# HANDOFF — dla-native（DeepDoc 纯 Go 复刻 / 漂移比对）

> **权威 ADR 见 `README.md`**（定位、边界、纯 Go 决策、3px 地板、golden 与漂移门语义）。
> 本文档为会话接力备忘，**只保留仍有效的要点**；gocv 双轨与 native_det 生产接线均**已移除/回退**，
> 下文不再保留其历史。

## 1. 当前状态（已验证）

- `dla-native` 是**纯 Go 验证工具（verification harness）**，不接生产。
- det / DLA / TSR / OCR-rec 四组件均已完成 Go 复刻，在 CPU 上对齐 Python oracle：
  - DLA / TSR：坐标在容差内匹配（3px 地板 + margin，见 README §3/§6）。
  - OCR-rec：识别文本一致。
  - det：仅作比对工具（`TestDetIntegration` golden），不接 `parser/` 生产路径。
- 3px 坐标为已知硬下限（源自 `bilinearResize` + `box#8` 后处理），**已接受，非缺陷**。

## 2. 关键环境

```bash
export ORT_LIB=<onnxruntime capi/libonnxruntime.so>
export MODEL_DIR=<deepdoc 模型目录: det.onnx/layout.onnx/tsr.onnx/rec.onnx/ocr.res>
export PYTHONPATH=<ragflow 仓库根>   # python-drift 跑 ref_*.py 需要 import deepdoc
```

## 3. 运行 / 对比

```bash
cd internal/deepdoc/dla-native
# 单元（无 tag，不依赖模型）
go test ./native/
# 集成（需 ORT_LIB + MODEL_DIR）
go test -tags integration ./native/
# CLI demo
ORT_LIB=$ORT_LIB MODEL_DIR=$MODEL_DIR go run . -task det -image testdata/page0.png
# Python oracle 比对（drift gate 本地等价调用）
PYTHONPATH=<ragflow 根> <venv>/python check_drift.py
```

## 4. 关键文件

| 文件 | 作用 |
|------|------|
| `main.go` | 入口，flag → `native` 派发 |
| `native/det_core.go` | det 共享核心：`RunDet`、unclip、`minAreaRect`/几何原语、有界 session 池 |
| `native/det.go` | det 几何路径（连通域 + convexHull） |
| `native/dla_preprocess.go` | DLA 预处理（Go decode + bilinearResize + letterbox） |
| `native/dla.go` / `tsr.go` / `tsr_decode.go` / `ocr_rec.go` | DLA/TSR/OCR-rec 管线 |
| `native/session.go` / `session_pool.go` | ORT session 封装与有界复用池 |
| `native/clipper_offset.go` | pyclipper/Clipper1 移植（`clipperOffset`） |
| `ref_det.py` / `ref_dla.py` / `ref_tsr.py` / `ref_ocr_rec.py` | 各组件 Python oracle |
| `check_drift.py` | Python oracle ↔ golden 漂移门 |
| `run.sh` | 四组件一键 Go-vs-Python 对比 |

## 5. 已知约束 / 不要重做

- **不要重建子进程隔离机制**——曾被证伪为诊断方向假象。
- **不要改 `clipper_offset.go`**——已验证与 pyclipper 0.000px 一致。
- **3px 是硬下限**（源自 cv2 自身浮点伪影，非 Go 缺陷）；纯 Go 浮点 `minAreaRect` 已落地，
  若要消除需做 cv2 逐位浮点仿真（极脆弱），除非明确决定否则不要改动几何核心。
- **det 不接生产**：比对能力只在 `dla-native`（`TestDetIntegration` 等）+ `main.go` demo，
  从不被 `parser/` 或 `ingestion/` 引用。
- 回归测试必须用 `//go:build` 分层（unit 无外部服务），不要靠 `t.Skip`+env 软隔离。

## 6. CI 漂移门

`.github/workflows/deepdoc-drift.yml` 两个 job：

1. `python-oracle drift gate`：重跑 Python oracle 比对 pinned golden（`ref_det.py` 需 `PYTHONPATH=仓库根`）。
2. `go DLA/TSR/OCR-rec/det integration`：`go test -tags integration ./native/` 比对同一 golden。

（详见 README §4；golden 重生成步骤见 README §7。）
