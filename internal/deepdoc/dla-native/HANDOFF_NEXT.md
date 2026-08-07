# HANDOFF — 架构评审后续 / 接力待办

> 本文件用于**新会话接力**。它捕获「2026-08-07 这一轮」已完成的修复与 CI 接线，
> 以及 CTO/资深架构师评审后尚未推进的后续项（P0 灰度收尾 / P1 双池合一 / P2 收口暴露面 / P3 决策）。
> 深度移植历史见同目录 `HANDOFF.md`（det/DLA/TSR/OCR-rec 对齐、夹具、命令）。
>
> **重要**：`HANDOFF.md` 中 §8「A2·session 复用」描述的**无界** `detSessionPools sync.Map`
> 已被本轮回合**替换**为有界 LRU（见 §1.1）。新会话勿以旧描述为基准。

## 0. TL;DR（新会话从这里开始）

- 本轮回合已完成：DET 会话池内存泄漏修复（无界 → 有界 LRU）、gocv 复用抖动根因（确定性 box 排序）、
  P5 清理 `TSR_RAW_DUMP`、P4 gocv CI 覆盖、native_det 生产路径 CI 入口。
- 已提交并推送到 `feat/deepdoc-go-port`：**commit `0a0688ff2`**。
- 已派发 `deepdoc-drift`：**https://github.com/xugangqiang/ragflow/actions/runs/31157005421**
  （含 `go-native-det`，非阻断；还含 `python-drift` / `go-dla-native` / `go-dla-native-gocv`）。
- **P0 native_det 已完成**：run `31157005421` 中 `go-native-det` 失败，根因是供给路径 bug（三个
  tarball 被平铺进 `$NL` 根目录，而 run 步骤按 `$NL/<name>/lib/...` 引用 → `.a` 全部打不开；外加
  pdf_oxide 被钉成 `v0.3.73` 而 `build.sh` 规范是 `v0.3.67`）。修复：每个 tarball 解压到具名子目录 +
  pdf_oxide 对齐到 `v0.3.67`。已在本地验证（修正布局链接通过、全部 `TestNativeOCRDetect*` 对 golden 通过），
  提交 `2ed53bec8` 推送后派发 run **https://github.com/xugangqiang/ragflow/actions/runs/31160394410**
  确认 `go-native-det` 变绿，随后提交 `8db3f0c1f` **移除 `continue-on-error`** 使其成为阻断闸门。
- 剩余：见 §3（P1/P2/P3；P0 仅剩 `go-dla-native-gocv` 的收尾，但 gocv job 仍有独立的 pkg-config 供给
  bug，见 §3 P0 注）。**`python-drift` 也失败（独立供给 bug，见 §3 P0 注），与本轮回合代码改动无关。**

## 1. 本轮回合已落地（均本地验证通过）

| 项 | 内容 | 位置 | 验证 |
|----|------|------|------|
| DET 会话池泄漏修复（P0） | 无界 `var detSessionPools sync.Map` → 有界 LRU（`detMaxShapePools=24` 池 × `detShapePoolCap=4` 空闲），带 LRU 淘汰 + `Destroy()` 空闲 session | `native/det_core.go` | `TestDetSessionPoolBounded`（`integration`）：80 个不同合成页尺寸喂 `RunDet`，断言 `len(detPools) <= 24`；两构建均得 24 池。det 漂移不变（15/15 @3px） |
| gocv 复用抖动根因（P0） | Go `map` 迭代非确定性 → `Wire()` 顺序抖动；对 `res.Boxes` 按 class→X0→Y0→X1→Score 确定性排序 | `native/dla.go`（`dlaPostprocess` 末尾）、`native/tsr.go`（按 `tsrClassMap`→X0→Top→X1→Bottom→Score） | 20× gocv + 10× nogocv 复用运行全过；full gocv integration `ok` |
| P5 清理 `TSR_RAW_DUMP` | 删除 `os.Getenv("TSR_RAW_DUMP")` 调试块及连带无用的 `os`/`math` import | `native/tsr.go` | 两构建编译通过；gocv integration `ok` |
| P4 gocv CI 覆盖 | 新增 `go-dla-native-gocv` job：CI 内从源码构建 OpenCV 4.10 到缓存前缀并跑 `go test -tags "integration gocv"`，`continue-on-error` 非阻断 | `deepdoc-drift.yml`（job `go-dla-native-gocv`，约 :162） | YAML 校验通过；本地 gocv 集成 `ok` |
| native_det 生产路径 CI 入口（P0） | 新增 `go-native-det` job：自包含拉取并链接 3 个 PDF 原生静态库（office_oxide v0.1.8 / pdfium chromium/7809 / pdf_oxide v0.3.73），跑 `TestNativeOCRDetect*`，`continue-on-error` 非阻断 | `deepdoc-drift.yml`（job `go-native-det`，约 :259） | `go vet -tags "native_det integration"` 类型检查通过；最终 link 依赖原生库，按设计在真实 CI 验证 |

## 2. 关键事实（避免新会话重复踩坑）

1. **DET 输入尺寸可变**（`detLimitSideLen=960`，round32 保持长宽比）→ 不同页 = 不同 ONNX session 形状
   → 这是无界池泄漏的驱动因素。固定 shape 的 DLA(1024)/TSR(640)/OCR-rec(48×320) 键实质塌缩为 modelDir，不会泄漏。
2. **gocv 复用抖动根因**是 Go `map` 迭代非确定性，不是数值问题。修复是确定性排序，不是改几何核心。
3. **gocv / nogocv 语义一致**：最终推理结果无差异，仅 decode/resize 精度有 ~2.6px 地板
   （gocv 走 cv2 实现 1:1 parity；纯 Go 有浮点 floor）。发行不必二选一。
4. **native_det 是生产真实接线路径**：`internal/deepdoc/parser/pdf/nativeOCRDetect` → dla-native `RunDet`。
   其测试 `TestNativeOCRDetect*`（`//go:build native_det && integration`）位于 `internal/deepdoc/parser/pdf/`，
   **需要链接 PDF 原生静态库**（office_oxide/pdfium/pdf_oxide）；本地无库无法 link，只能在 CI 验证。
   `go vet` 可绕过 link 做类型检查，但真正的 run 必须在有原生库的 CI 中。
5. **dla-native 是嵌套 module**（module path `dla-native`，`internal/deepdoc/dla-native/go.mod`）。
   编译需 `cd internal/deepdoc/dla-native` 后用 `go build -tags integration ./native/`（nogocv）
   或 `-tags "integration gocv"`（gocv）。主模块 `go build ./internal/deepdoc/dla-native/native/` 会报
   “main module does not contain package”。

## 3. 剩余待办（接力重点）

### P0 — 灰度收尾（依赖真实 CI 结果）
- **`go-native-det` 已完成（阻断闸门）**：run `31160394410` 验证变绿，commit `8db3f0c1f` 已移除
  `continue-on-error` → 现在 `go-native-det` 是阻断式闸门。修复内容见 §0（具名子目录解压 + pdf_oxide 0.3.67）。
- **`go-dla-native-gocv` 收尾（未做）**：该 job 在 run `31157005421` 仍失败，但根因是**独立的 pkg-config
  供给 bug**——OpenCV 4.10 从源码编译、`opencv4.pc` 已正确装到 `opencv-4.10/lib/pkgconfig/opencv4.pc`，
  但 run 步骤 `pkg-config` 仍报找不到（`Package opencv4 was not found`）。本地 opencv 是 4.6.0 且在默认路径，
  所以本地 gocv 一直 ok。需先修该 pkg-config 供给问题并验证变绿，**之后**才能移除其 `continue-on-error`
  （约 `:165`）改为阻断。注意：生产路径已是 nogocv（见 P3 决策），gocv 仅作 cv2 1:1 parity 交叉校验。
- **`python-drift`（python-oracle）失败（独立、疑似既存、超出本轮回合范围）**：run `31157005421` 中
  `ref_det.py` 报 `ModuleNotFoundError: No module named 'deepdoc'`（`from deepdoc.vision.ocr import
  TextDetector`），即该 job 没把仓库根加入 `sys.path`。本轮改动未触碰 python 侧/ref_det.py，属独立供给问题，
  需单独确认是否修复（给 run 步骤加 `PYTHONPATH=$GITHUB_WORKSPACE` 之类）。
- 注：若 CI run 中 `go-native-det` 失败，先排查是 (a) 原生库下载/链接问题，还是 (b) 真实回归。
  链接/下载问题是 job 自身供给问题（非阻断期内可迭代），真实回归则需回退对应改动。

### P1 — 合并双会话池
- `native/session_pool.go` 的 `modelSessionPools`（固定 shape，DLA/TSR/OCR-rec）与
  `native/det_core.go` 的 `detPools`（可变 shape DET）是**两套机制**。收敛为一个泛型 pool 原语。
  - 键类型不同（`modelSessKey` vs `detSessKey`），但池语义一致（Get/Put 单 owner、空闲上限、LRU 淘汰）。
    可抽 `genericSessionPool[key any]` 承载两者。
  - 合并时**保留 DET 的 LRU 有界淘汰语义**（cap 24×4）；固定 shape 池键极少无需淘汰，可作为退化情况。
- 验收：双池合一后 `TestDetSessionPoolBounded` / `TestDLASessionReuse` / `TestTSRSessionReuse` /
  `TestOCRRecSessionReuse` 仍通过，且 `go test -tags integration ./native/`（nogocv + gocv）`ok`。

### P2 — 收口暴露面与健壮性
- 降为未导出：`native` 包导出的 `NMS` / `BilinearResize` / `Session` / `Box` 仅为本包/测试使用，应改为未导出。
  （注意：跨 package 的测试若引用需保留或挪到 internal 测试；先 grep 引用面再改。）
- `Session.Run` 补 `len` 长度守卫：当前 `copy(s.in.GetData(), input)` 无长度校验，
  `input` 长度不符会 panic/越界。先校验 `len(input) == len(s.in.GetData())`。
- `Run*` 系列（`RunDet`/`RunDLA`/`RunTSR`/`RunOCRRec`）补 `context.Context` + 超时：
  当前无取消/超时，长页/异常输入无上限。池是单 owner 复用，`ctx` 应透传到 `Session.Run` 的 ORT 调用（若 ORT binding 支持）。
- 验收：上述改动后全量 `integration` + `unit` 测试仍 `ok`，且暴露 API 面收窄（不影响 `internal/deepdoc/parser/pdf` 的调用）。

### P3 — 需用户拍板的决策
- **gocv / nogocv 双轨收敛**：两条构建语义一致，发行不必二选一，但双轨是长期维护债务。选项：
  - (a) 保留双轨（现状，最低风险）；
  - (b) 收敛到 gocv 单一路径（所有构建引入 OpenCV 依赖，换取 cv2 1:1 parity）；
  - (c) 收敛到纯 Go 单一路径（放弃 cv2 1:1，接受 ~2.6px decode/resize 地板，去掉 OpenCV 依赖）。
  - 建议先问用户倾向，再动手。
- **明确 DLA/TSR/OCR-rec 生产接线状态**：漂移门只证明「与 golden 一致」。需确认这些端口是否已接进生产路径。
  - det 已接 `internal/deepdoc/parser/pdf` 原生路径（`nativeOCRDetect`）。
  - DLA/TSR/OCR-rec 是否也被生产消费？查 `internal/deepdoc/parser/...` 或 `internal/ingestion/...`
    是否引用 dla-native 的 `RunDLA`/`RunTSR`/`RunOCRRec`。若未接，需决定是否接线或仅保留为比对工具。

## 4. 环境 / 命令速查

```bash
# 嵌套 module：必须 cd 进去再 build
cd internal/deepdoc/dla-native
go build -tags integration ./native/            # nogocv
go build -tags "integration gocv" ./native/     # gocv（需 OpenCV 4.10 前缀）
go test  -tags integration ./native/            # nogocv 集成
go test  -tags "integration gocv" ./native/     # gocv 集成

# native_det 测试（需 PDF 原生静态库 + ORT）：本地无库时只能 go vet 类型检查
cd /home/shenyushi/codex-workspace/ragflow
go vet -tags "native_det integration" ./internal/deepdoc/parser/pdf/
# 真跑见 CI 的 go-native-det job，或按 build.sh setup_cgo_env 的 CGO flags 本地配库

# 漂移门
cat .github/workflows/deepdoc-drift.yml
gh run view 31157005421 -R xugangqiang/ragflow
```

## 5. 不要重做 / 不要回退
- 不要回退：确定性 box 排序（dla.go/tsr.go）、有界 LRU（det_core.go）、`TSR_RAW_DUMP` 清理（tsr.go）。
- 不要重建子进程隔离机制（已被证伪为诊断方向假象，见 `HANDOFF.md` §4.1）。
- 不要改 `clipper_offset.go`（已验证与 pyclipper 0.000px 一致）。
- box#8 的 3px 是硬下限（源自 cv2 自身浮点伪影，非 Go 缺陷）；除非用户明确决定做 cv2 逐位浮点 `minAreaRect` 仿真，否则不要为消除它改动几何核心。

## 6. 提交 / 分支状态
- 分支：`feat/deepdoc-go-port`（与 `origin/feat/deepdoc-go-port` 同步）。
- 本轮回合提交：`0a0688ff2` — “fix(deepdoc): bound DET session pool, stabilize gocv ordering, add CI jobs”。
- P0 native_det 修复提交：`2ed53bec8` — “fix(ci): provision native_det libs into named subdirs + align pdf_oxide”，
  以及 `8db3f0c1f` — “ci(deepdoc-drift): make go-native-det a blocking gate”。
- 已排除根目录 `client.py`（LitServe 自动生成的本地测试脚本，与本工作无关，未纳入提交）。
- CI run：`31157005421`（含 `go-native-det`，首次失败，根因供给路径 bug）；`31160394410`（修复后验证
  `go-native-det` 变绿，并据此将其改为阻断闸门）。
