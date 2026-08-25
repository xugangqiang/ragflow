# Handoff: DeepDoc 真 batch 接入调用侧 + 移除外部 Python 推理服务依赖

> 状态:**已完成**(截至 2026-08-24)
> 范围:`internal/deepdoc/native`、`internal/deepdoc/parser/pdf`、`internal/deepdoc/parser/pdf/inference/native_analyzer`
> 环境:cgo 依赖齐全(pdfium / office_oxide / pdf_oxide / onnxruntime 均在 `~/ragflow-native-libs/`)

> **后续清理(本 handoff 计划执行后追加完成)**:
> - 真 batch 接入部分按计划完成,已在 `build.sh --test-native` 下验证通过。
> - 移除 PyOracle + 改写对比测试:全部完成。`py_oracle.go`、`inprocess_vs_service_*` 测试、`tmpcheck/`、`ocr_merge_test.go`、`TestWireVsLiveServer` 已删除;`DefaultDLALabels`/`DefaultTSRLabels` 迁至 `cache.go`;12 个测试改用 in-process `NativeAnalyzer`。
> - 进一步清理(本 handoff 原未列、事后确认属同一残留):删除 `pdf_parser_resolver_test.go`;删除 6 处冗余 `t.Setenv("DEEPDOC_URL","")`;清理 `environments.go` 死配置(`GetDeepDocURL`/`GetOSSDeepDocURL` getter、`DeepDocURL`/`OSSDeepDocURL` 字段与 env 读取、`EnvDeepDocURL`/`EnvOSSDeepDocURL` 常量);`resolveDocAnalyzer` 移除死参数 `baseURL`。
> - 文档同步:修正 `AGENTS.md`/`CLAUDE.md`(删除"启动需外部服务"错误描述)、`EQUIVALENCE.md`(删除 `TestWireVsLiveServer`/live-oracle 过时描述)、`docker/.env` 注释。
> - **未处理(需单独决策)**:`docker/docker-compose.yml` 仍定义独立 `deepdoc` 服务容器(端口 9390),`docker/README.md` 仍描述 `DEEPDOC_URL`;Go 代码已不连它,但部署层服务容器是否移除属独立 scope。

---

## 0. TL;DR

1. 已将 OCR 真 batch(`RunOCRRecBatchReal`,单次 ONNX Run 覆盖整页所有 crop)接入 PDF 解析的 OCR 调用侧(`ocrDetectAndRecognize` + `buildTextBoxes`)。
2. 通过 `build.sh`(cgo 环境)端到端验证:底层真 batch 与调用侧 `NativeAnalyzer.OCRRecognizeBatch` 在真实 ORT + 模型下均正确。
3. 发现并修复一个编译 bug(`parser_ocr.go` 未使用变量 `bi`)。
4. 发现 Go 端仍残留外部 Python 推理服务的 HTTP 客户端 `PyOracle`(非测试文件),违反"生产只用 in-process"的架构。已批准计划:**彻底移除 PyOracle + 改写对比测试为 in-process NativeAnalyzer**。该计划已批准,尚未执行。

---

## 1. 已完成的改动(真 batch 接入)

### 1.1 底层真 batch(`internal/deepdoc/native`)
- `ocr_rec.go`:`RunOCRRecBatchReal(ctx, modelDir, []*Image)` —— 把每行的预处理 blob 拼成一个 `{N,3,48,imgW}` 张量,**一次** ONNX Run,逐行 CTC 解码。与逐张 `RunOCRRec` 数值一致(已证明)。
- `image.go`:`FromImages(imgs []image.Image) ([]*Image, error)` —— 批量 `FromImage`。

### 1.2 生产分析器(`internal/deepdoc/parser/pdf/inference/native_analyzer/native_analyzer.go`)
- `OCRRecognizeBatch(ctx, []image.Image) ([][]deepdoctype.OCRText, error)`:
  - `n==0` → nil;`n==1` → 回退单张 `OCRRecognize`(无 batch 宽度拉宽);
  - `n>1` → `native.FromImages` + `native.RunOCRRecBatchReal`(真 batch)。
  - 应用 `dropScore` 空白化契约,与单张路径一致。
- 修复:原文件有**两份重复的 `OCRRecognizeBatch`** 方法(编译错误),已删除第一份,仅保留基于 `FromImages` 的版本。

### 1.3 调用侧(`internal/deepdoc/parser/pdf/parser_concurrency.go` + `parser_ocr.go`)
- `parser_concurrency.go`:
  - 可选接口 `batchRecognizer { OCRRecognizeBatch(ctx, []image.Image) ([][]pdf.OCRText, error) }`。
  - `inferOCRRecognizeBatch(ctx, doc, crops)`:整批占一个 limiter slot,调用 `doc.(batchRecognizer).OCRRecognizeBatch`。
  - `docSupportsBatchOCR(doc)`:是否实现 batch 能力。
  - **未实现该接口的 analyzer(MockDocAnalyzer / DocAnalyzerCache / replay analyzer)自动回退逐 crop 路径**,接口零破坏。
- `parser_ocr.go`:
  - `ocrDetectAndRecognize`:先收集每 box 的 layer-2 旋转候选(矮宽=1 张;高窄=3 张 0/CW90/CCW90),积成一维 `cropAcc`;若 `docSupportsBatchOCR` → 一次 `inferOCRRecognizeBatch`(单次 Run);否则逐 crop 回退。随后按 box 选最高置信度候选并发 `TextBox`。
  - `buildTextBoxes`:OCR 补识别分支同样接入 batch(box 轴对齐,每 box 单候选)。
  - **replay analyzer 不实现 batchRecognizer**,永远走逐 crop 回退路径,`ocrBoxIdxCtxKey` 按 box index 路由的语义保持完整 → 对拍测试不受影响。

### 1.4 新增测试
- `native_analyzer_test.go:TestAnalyzerOCRRecBatchIntegration`(`//go:build cgo && integration`):
  - 验证 `NativeAnalyzer.OCRRecognizeBatch` 逐行文本/置信度 == 底层 `native.RunOCRRecBatchReal`(真 batch oracle);
  - 并断言 batch 语义确实生效(至少部分行文本 != 逐张 `OCRRecognize`,因 batch-max 宽度拉宽)。
  - 注:第一轮断言写错(误以为 batch==单张),后修正为与 `RunOCRRecBatchReal` 对比 —— batch 宽度语义本就使 batch≠single。

---

## 2. 端到端验证结果(本环境,用 `build.sh`)

| 验证项 | 命令 | 结果 |
|---|---|---|
| `parser/pdf` 主包编译链接 | `go build -tags cgo`(build.sh cgo env) | ✅ EXIT=0 |
| `native_analyzer` 包(无模型时) | `build.sh --test-native` | ✅ ok(先抓出 `bi` 未使用 bug 已修) |
| native 真 batch 推理 | `TestOCRRecBatchIntegration`(ORT+模型) | ✅ ok 11.3s |
| `NativeAnalyzer` 整体 | `build.sh --test-native`(ORT+模型) | ✅ ok 387s |
| 调用侧 batch 接口 | `TestAnalyzerOCRRecBatchIntegration`(ORT+模型) | ✅ ok 7.6s |

环境变量(本环境实际可用,非必需——`resolveDeepDocModelDir`/`resolveDeepDocORTLib` 默认值即命中):
```bash
export ORT_LIB="$HOME/ragflow-native-libs/onnxruntime/onnxruntime-linux-x64-1.23.2/lib/libonnxruntime.so"
export MODEL_DIR="$PWD/rag/res/deepdoc"      # 含 det.onnx/rec.onnx/tsr.onnx/layout*.onnx/ocr.res
# 对应 server 用的是:
export DEEPDOC_ORT_LIB="$ORT_LIB"
export DEEPDOC_MODEL_DIR="$MODEL_DIR"
```
native 模块的集成测试读的是 `ORT_LIB` / `MODEL_DIR`(不是 `DEEPDOC_*`)。**注意**:从子目录跑时 `MODEL_DIR` 要用绝对路径,否则会解析到错误位置(曾因 `$PWD/../../rag/res/deepdoc` 多嵌套一层导致 FAIL,非代码问题)。

---

## 3. 待执行:移除外部 Python 推理服务依赖(计划已批准)

### 3.1 背景(已查证)
- 生产路径**已不**依赖外部服务:`internal/parser/parser/pdf_parser_common.go:332` `deepDocAnalyzerFromEnv` 把 `baseURL`(DEEPDOC_URL)用 `_ = baseURL` 显式忽略,只用 in-process `NativeAnalyzer`。
- `GetDeepDocURL()` / `GetOSSDeepDocURL()` getter 全仓库无调用方。
- **残留**:`internal/deepdoc/parser/pdf/inference/py_oracle.go` 是**非测试文件**(无 `_test.go`、无 build tag),却实现完整 HTTP 推理客户端 `PyOracle`(`NewPyOracle`/`DLA`/`TSR`/`OCRDetect`/`OCRRecognize`/`postImage`),只被测试引用。违反 AGENTS.md「删除迁移残留」。

### 3.2 用户决策
1. 改写策略 = 改用 `NativeAnalyzer`(真实推理),相关测试加 `cgo` tag → 成为只在部署 ORT 的 `--test-native` 构建里跑的真实推理回归测试。
2. 纯 Python 对比测试 = 全部删除。

### 3.3 执行清单(任务 #4 / #5 已创建,未执行)

**删除:**
- `internal/deepdoc/parser/pdf/inference/py_oracle.go`
- `internal/deepdoc/parser/pdf/inprocess_vs_service_compare_test.go`(`cgo && cgo`,纯 Go-vs-Python 对比)
- `internal/deepdoc/parser/pdf/inprocess_vs_service_iou_test.go`(`cgo && cgo`,纯对比)
- `internal/deepdoc/parser/pdf/tmpcheck/`(整目录:3 个临时调试对比测试 + `EQUIVALENCE_REPORT.md`)
- `internal/deepdoc/parser/pdf/ocr_merge_test.go`(`cgo && manual`,连 PyOracle 对比)
- `internal/deepdoc/native/native_integration_test.go` 的 `TestWireVsLiveServer`(约 :1466,POST 到 DEEPDOC_URL 做 live 对比) —— 删函数 + 清理其独有 import(`os`/`net/http`/`fmt` 等若不再被用)

**改写(用 NativeAnalyzer,加 cgo tag):**
- `helpers_test.go`:build tag `cgo` → `cgo && cgo`;新增 `infnative`+`native` import;`mustConnectInferenceClient` → `mustConnectInProcessAnalyzer`(返回 `*infnative.NativeAnalyzer`,内部 `native.InitORT` + `infnative.NewAnalyzer`)。
- 调用方 build tag 加 `cgo`:
  - `dla_real_world_test.go`、`dla_tsr_compare_test.go`、`inference_client_integration_test.go`、`parser_parallel_integration_test.go` → `cgo && cgo && integration`
  - `parser_pipeline_integration_test.go`、`parser_pipeline_manual_test.go`、`production_smoke_test.go`、`scan_all_pdfs_test.go` → `cgo && cgo && manual`
  - 调用处函数名改为 `mustConnectInProcessAnalyzer`
- 直接 `inf.NewPyOracle`(不经 helper)的:`batch_smoke_test.go`、`table_crop_ab_manual_test.go`、`table_rotate_integration_test.go`(`cgo && manual`→`cgo && cgo && manual`);`parser_ocr_rotate_integration_test.go`(`integration` 无 cgo → `cgo && cgo && integration`),均改 `mustConnectInProcessAnalyzer`。

**保留不动:**
- `internal/deepdoc/parser/pdf/inference/cache.go`(`DocAnalyzerCache`,in-process Redis 缓存包装,不连外部服务)+ `cache_test.go`。(注:生产代码当前无 `NewDocAnalyzerCache` 调用方,属独立清理项,不在本任务 scope。)
- `MockDocAnalyzer` / `PythonIntermediateDocAnalyzer`(replay)及 parity 测试。
- `pdf_parser_resolver_test.go`(验证 DEEPDOC_URL 被忽略)、ingestion/parser 里 `t.Setenv("DEEPDOC_URL","")` 测试(验证生产不依赖外部服务,有价值)。
- `util/warp_test.go`(仅注释提及)。

**附带清理:**`parser_concurrency.go:226-234` 的 `batchRecognizer` 注释提到 `PyOracle`,改为实际存在的实现(MockDocAnalyzer / DocAnalyzerCache / replay analyzer)。

### 3.4 验证(执行后)
```bash
bash build.sh --test-native   # 编译+真实 ORT 下跑所有 cgo 测试(含改写的)
grep -rn "PyOracle\|NewPyOracle\|mustConnectInferenceClient" --include=*.go internal/ cmd/   # 应只剩 parser_concurrency.go 注释(已更新)
go vet -tags cgo ./internal/deepdoc/parser/pdf/   # 确认不带 cgo 的常规测试(Mock 路径)仍编译
```

---

## 4. 已知约束 / 坑

- **pdfium 实际存在**:之前某轮对话误判"本环境缺 pdfium 无法链接",实际 `~/ragflow-native-libs/pdfium-static/lib/libpdfium.a` 存在。`build.sh` 的 `setup_cgo_env` 能正常导出 cgo flags,可直接编译/测试。**不要再假设无法链接。**
- **`cgo` 强制**:`NativeAnalyzer` 在 `//go:build cgo` 包,import 链拉入 onnxruntime cgo。任何改用它的测试必须带 `cgo` tag,且只能在部署 ORT 的环境(`--test-native`)跑。
- **batch 宽度语义**:`RunOCRRecBatchReal` 把所有行 resize 到 batch-max 宽度,故 batch 文本**不同于**逐张 `OCRRecognize`(单张用固定 recW=320)。这是设计如此,断言时应以 `RunOCRRecBatchReal` 为 oracle,而非单张。
- **`native.InitORT` 进程级全局**,多次调用须幂等(沿用 `analyzerWithModels` 模式)。

---

## 5. 关键文件索引

| 文件 | 角色 |
|---|---|
| `internal/deepdoc/native/ocr_rec.go` | `RunOCRRecBatchReal` / `RunOCRRec` / `RunOCRRecBatch` |
| `internal/deepdoc/native/image.go` | `FromImage` / `FromImages` |
| `internal/deepdoc/parser/pdf/inference/native_analyzer/native_analyzer.go` | `NativeAnalyzer`(含 `OCRRecognizeBatch`) |
| `internal/deepdoc/parser/pdf/parser_concurrency.go` | `batchRecognizer` 接口 / `inferOCRRecognizeBatch` / `docSupportsBatchOCR` |
| `internal/deepdoc/parser/pdf/parser_ocr.go` | `ocrDetectAndRecognize` / `buildTextBoxes`(batch 调用侧) |
| `internal/deepdoc/parser/pdf/inference/py_oracle.go` | **待删**(外部服务残留) |
| `internal/deepdoc/parser/pdf/inference/cache.go` | `DocAnalyzerCache`(保留) |
| `internal/parser/parser/pdf_parser_common.go:332` | `deepDocAnalyzerFromEnv`(忽略 DEEPDOC_URL) |
| `build.sh` | `setup_cgo_env`(cgo flags)、`--test-native` |
| `cmd/ragflow_server_native.go` | `registerNativeDeepDoc`(生产只注册 in-process) |
