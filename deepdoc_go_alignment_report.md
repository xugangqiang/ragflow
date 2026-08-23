# DeepDoc Go in-process 与 Python 推理服务对齐报告

> 面向非技术读者，说明：Go 侧本地推理后端与 Python 推理服务的关系、依赖迁移方式、以及两者的语义一致性结论。

---

## 1. 背景

RAGFlow 的 PDF 解析依赖 DeepDoc 推理（OCR 文本检测 / 版面分析 DLA / 表格结构 TSR / 文本识别 OCR）。原有部署依赖一个独立的 **Python 推理服务**（`deepdoc_server`）。现已在 Go 侧实现**进程内（in-process）推理后端**：

- 用**纯 Go + ONNX Runtime** 复刻了同一批模型和同样的后处理；
- **打包进单个服务端 binary**（`bin/ragflow_server`），无需再单独构建/部署 Python 服务；
- 两者实现同一个 `DocAnalyzer` 接口，上层调用方无感知。

---

## 2. Python 推理服务依赖清单

### 运行时推理（deepdoc/vision 直接使用）

| 包 | 用途 |
|---|---|
| `onnxruntime>=1.20.0` | 跑 det / layout / tsr / rec 四个 ONNX 模型 |
| `opencv-python-headless`（cv2） | 图像解码、resize、findContours、minAreaRect+boxPoints、fillPoly、approxPolyDP、NMS |
| `numpy` | 张量 / 数组运算 |
| `pillow`（PIL） | 服务端图片解码 / 编码 |
| `pyclipper>=1.4.0` | det 框 unclip（多边形膨胀） |
| `shapely` | 仅用于 unclip 的 `Polygon.area/length` 计算 |
| `six` | py2/3 兼容垫片 |

### 服务层（HTTP）

`litserve>=0.2.17`、`python-multipart`

### 栅格化（推理边界外）

`pdfplumber`（PDF → 图片，@216 DPI）

### 模型下载

`huggingface_hub`（`snapshot_download`）

---

## 3. Go 侧迁移方式

| Python 依赖 | Go 迁移 | 状态 / 残差 |
|---|---|---|
| `onnxruntime` | `onnxruntime_go`（cgo，`InitORT` dlopen 加载 `libonnxruntime.so`；`session.go`/`session_pool.go` 有界池） | 同一批 `.onnx` 模型，sha256 锁定 |
| `cv2.resize (INTER_LINEAR)` | 纯 Go `bilinearResize` | 浮点 vs 定点 → **~3px 检测地板**（已接受） |
| `cv2.findContours (RETR_LIST)` | 手写 Moore-neighbour（Suzuki-Abe）`findContours`（`det.go`） | 残余 = 框几何漂移（1:1 框集、mean IoU 0.969） |
| `cv2.minAreaRect + boxPoints` | 纯 Go 旋转卡壳 `minAreaRect`（`geometry.go`） | 对齐 |
| `cv2.fillPoly` | 纯 Go 扫描线 `fillPoly`（box_score_fast） | 与 cv2 位级一致 |
| `cv2.approxPolyDP / arcLength` | 未移植 | 只在 Python `poly` 路径用，quad 路径不走 |
| `cv2 NMS` | `nms.go`（仅 DLA-table 0.45 / TSR 0.2；det 无 NMS） | 对齐 |
| `cv2 imdecode / PIL` | Go `image` 包 `Decode`（格式自动识别 + 尺寸/像素上限） | decode 贡献 ~0 |
| `pyclipper`（unclip, JT_ROUND） | `clipper_offset.go`（Clipper1 移植） | 与 pyclipper 0px 对齐 |
| `shapely Polygon.area/length` | `clipper_offset.go` 内直接算多边形 area/length | 无需依赖 |
| `numpy` | Go `[]float32` | — |
| `pdfplumber`（PDF→栅格 @216DPI） | 生产用 Go `pdfium.RenderPage` @216 DPI + `FPDF_LCD_TEXT` | 实测 DLA ≤0.03px |
| `litserve / python-multipart / six`（HTTP 服务） | 不迁移 | in-process 直接调用；`DEEPDOC_URL` 模式用 Go HTTP client 对接原服务 |
| `huggingface_hub`（模型下载） | 不迁移 | `ragflow_deps/download_deps.py` 拉同一快照，sha256 锁定 |

**一句话总结**：推理相关依赖**全部迁移**——`onnxruntime`→`onnxruntime_go`，`cv2`→纯 Go 几何重实现（resize / findContours / minAreaRect / fillPoly / NMS），`pyclipper`→`clipper_offset.go`，`shapely`/`PIL`/`numpy`→Go 原生。**不迁移**的只有三类：HTTP 服务层（in-process 取代）、模型下载（Python 脚本保留）、`pdfplumber`（换成 `pdfium`，实测对齐）。

---

## 4. 调用方视角：两者语义是否一致？

### 结论：**语义层面对齐；数值层面存在有界差（不影响上层）。**

调用方拿的是 `DocAnalyzer` 接口的四个输出：

| 输出 | 语义一致性 | 实测证据 |
|---|---|---|
| **文本内容**（OCRRecognize） | 一致 | 35-PDF 基线上仅 2/35 有差异，且只是置信度差 0.008，文字本身相同 |
| **版面结构**（DLA） | 一致 | 171/174（98.3%）<1px 匹配；端到端光栅对齐文本页 ≤0.03px |
| **表格结构**（TSR） | 一致 | 85/85（100%）匹配 |
| **文本区域集合**（OCRDetect） | 一致 | 1559 框 1:1 匹配，仅 **1 个孤儿框**（Go 多检） |

调用方看到的"哪块有字、是什么字、版面/表格怎么排"——**语义一致**。`drop_score` 契约也已对齐（两边都在置信度 <0.5 时置空文本但保留真实分数）。

但"完全一致" ≠ 逐字节相同：坐标有数像素抖动（mean IoU 0.969、最差 21px）、置信度有 ~1e-5 浮点噪声、且有那 1 个孤儿框。这不是"每个框偏几像素"的均匀差，而是少数框的几何漂移。

### 35 个 PDF 的中间结果，对上层语义是否一致？

**是，语义一致；有一个可忽略的例外。**

- 最终产出（chunk / table-HTML / markdown）由**文本内容 + 版面/表格结构**决定，而这两者一致 → 上层输出语义一致。
- 坐标抖动不影响上层：阅读顺序/分组按邻近度，几像素不改变分组；表格 HTML 由 TSR 单元（100% 一致）驱动。
- **唯一例外**：`10_numbering_patterns.pdf` 的 1 个 Go-only 孤儿框。若该框被识别并入 chunk，会多出一小段文本跨度——**1/1559 框，可忽略**，但严格说不是 100% 相同。

> 诚实边界：等价性证明的范围是**推理边界**（同图同模型）；`PdfParser` 下游（切分/拼装）本身不在证明范围内。但因为其输入（文本/结构）一致，上层输出语义一致是强推断，非穷举实测。

### 为什么不完全一致（数值层的"为什么"）

残差全部来自**检测几何**，与 `drop_score`、ORT 版本无关（受控实验已证伪 ORT 版本）：

1. **轮廓追踪算法**：Go 手写 Moore-neighbour vs `cv2.findContours` → 边界像素不同 → `minAreaRect` 几何略偏；
2. **resize 插值**：Go 浮点 `bilinearResize` vs cv2 定点 `INTER_LINEAR` → ≤1/255 像素噪声 → seg 图差 634px → 轮廓边界随之变；
3. 两者叠加，在少数接近阈值（0.5 box_thresh）的区域产生"score-flip"孤儿（那 1 个框）。

---

## 5. 结论与建议

- **语义层 = 一致**；**数值层 = 有界差**（坐标漂移 + 1 个孤儿框），对上层可忽略。
- 部署形态：**单个服务端 binary + 两个配套目录**（`libonnxruntime.so` + 模型快照）；不再需要独立的 Python 推理服务。
- 若未来要求"逐框数值完全一致"，只能把轮廓/resize 算法也对齐——已评估为**高成本低收益**，暂不作为目标。
