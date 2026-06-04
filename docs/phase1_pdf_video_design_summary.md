# Phase1 基础上的 PDF 与视频处理设计总结

本文基于 `git log` 中 phase1 之后的相关提交整理，重点覆盖两个新增设计：

1. PDF 转 Markdown
2. 视频预处理，包括稳定帧提取、音频转写和多模态上下文注入

相关核心提交包括：

- `3b76666 feat: pdf转md`
- `4d9aea3 feat: add lightweight PDF context overlay`
- `eeb1e45 feat: 支持视频任务上下文与模型调用配置`
- `d86da35 feat: 视频提取关键帧，以及语音转文字。再塞入上下文`

## 一、Phase1 基线与新增目标

Phase1 的核心运行方式是：每个任务从 `task.json` 和 `context/` 目录读取公开输入，Agent 通过 `list_context`、`read_doc`、`search_doc_text`、`execute_python`、SQL 工具等接口逐步探索数据，并最终写出 `prediction.csv`。

这个模式天然适合 CSV、JSON、SQLite、Markdown、TXT 等文本或结构化文件，但对 PDF 和视频存在两个问题：

- PDF 无法直接被现有文档工具稳定读取，尤其是目录、段落换行和中文硬换行会影响检索。
- 视频不能直接作为普通文件交给工具链处理；即使模型支持多模态，也不适合把原始视频完整塞入上下文。

因此当前设计不是重写 Agent 主流程，而是在任务执行前新增一个“上下文预处理层”，把 PDF 和视频转换为现有 Agent 已经能消费的文本、图片和 manifest。

## 二、总体设计：轻量上下文视图

新增的核心抽象是 `ContextView` 与 `ContextAsset`，定义在 `src/data_agent_baseline/benchmark/schema.py`。

`ContextAsset` 记录一个对 Agent 可见的上下文文件：

- `visible_path`：Agent 和工具看到的相对路径。
- `physical_path`：实际文件路径，可能是原始文件，也可能是预处理生成文件。
- `source_path`：来源文件路径，用于追踪原始资产。
- `action`：资产来源动作，如 `source`、`pdf_to_markdown`、`video_timeline`、`video_stable_frame`。
- `generated`：是否为生成资产。

`ContextView` 则保存：

- 原始 `context/` 目录。
- 生成文件目录 `generated_context/`。
- 当前任务对 Agent 可见的资产列表。

这一层的关键价值是：不必真的复制整个 `context/`，也不必修改所有工具的调用习惯。`list_context_tree`、`read_doc_preview`、`search_doc_text`、`execute_python` 等工具统一通过 `ContextView` 解析可见路径。对 Agent 来说，它仍然是在访问一个普通的任务上下文目录。

执行入口在 `run_single_task()` 中调用：

```text
原始 PublicTask
  -> prepare_task_context(...)
  -> 生成 ContextView / generated_context / context_preprocessing_manifest.json
  -> 使用预处理后的 PublicTask 运行 LangGraphAgent
```

## 三、PDF 转 Markdown 设计

### 1. 设计目标

PDF 处理的目标是把 PDF 转换成 Markdown 文档，让原有文本工具可以直接读取、搜索和按标题定位。

转换后，原始 PDF 不再作为可见资产暴露给 Agent，而是以 `.md` 文件形式出现在上下文文件树中。

### 2. 转换流程

核心实现位于 `src/data_agent_baseline/run/context_preprocessor.py`：

- `pdf_to_markdown(pdf_path)`
- `_extract_pdf_lines(document)`
- `_extract_toc_headings(document)`
- `_lines_with_toc_headings(...)`
- `_coalesce_markdown_lines(...)`

具体流程如下：

1. 使用 PyMuPDF `fitz` 打开 PDF。
2. 从页面文本块中抽取行文本，记录页码、纵坐标和 block index。
3. 读取 PDF 内置 TOC。如果存在目录，则把 TOC 映射成 Markdown 标题。
4. 通过标题文本和页面坐标，把正文中重复出现的标题行替换为 `#` / `##` 等 Markdown 标题。
5. 如果 PDF 没有 TOC，则不主动猜测标题，避免误把普通大字号文本当标题。
6. 对抽取出的行做段落合并，修复英文断行、连字符换行和中文硬换行。

### 3. 命名与冲突处理

PDF 的可见路径默认是同名 `.md`：

```text
doc/report.pdf -> doc/report.md
```

如果原始目录下已经存在同名 Markdown，则生成路径改为：

```text
doc/report.pdf -> doc/report_pdf.md
```

这样可以同时保留原始 Markdown 和 PDF 转换结果，避免覆盖或路径歧义。

### 4. 与工具链集成

PDF 转换后的 Markdown 会作为 `ContextAsset(action="pdf_to_markdown", generated=True)` 加入 `ContextView`。

因此：

- `list_context_tree` 会列出生成的 `.md`，不会列出原始 `.pdf`。
- `read_doc` 可以直接读取 PDF 转换结果。
- `search_doc_text` 可以在 PDF 转换结果中做正则和关键词检索。
- `lookup_doc_outline` 可以利用 TOC 生成的 Markdown 标题做章节定位。
- `TaskContextWorkspace.materialize()` 会物化一个不含 PDF、但含生成 Markdown 的临时工作区，供 Python 工具使用。

### 5. 当前边界

当前 PDF 方案主要面向可抽取文本型 PDF。它没有引入 OCR，也没有处理复杂表格、图片型 PDF 或版面还原。设计取舍是先把 PDF 纳入现有文本工具体系，提升检索和章节定位能力，而不是一次性做完整文档理解。

## 四、视频处理设计

### 1. 设计目标

视频处理的目标是把原始视频转换成三类 Agent 可消费的信息：

- `timeline.md`：按时间组织的视觉片段和语音转写。
- 稳定帧图片：从视频中抽取的代表性画面。
- manifest / transcript 产物：供复盘、调试和审计使用。

原始视频不会直接附加给模型，也不会作为可见上下文文件暴露给 Agent。这样可以控制 token 和请求体大小，也能避免不同模型对视频输入能力不一致的问题。

### 2. 视频文件识别

视频识别逻辑位于 `src/data_agent_baseline/run/video_preprocessor.py`：

```text
.mp4, .m4v, .mov, .webm, .mkv, .avi
```

`prepare_task_context_view()` 在扫描原始 `context/` 时，如果发现视频文件，就进入视频预处理分支。

### 3. 稳定帧提取

稳定帧提取由 `extract_stable_frames()` 完成，使用 OpenCV。

主要步骤：

1. 按 `sample_fps` 对视频采样。
2. 将采样帧缩放到固定宽度，转灰度并高斯模糊。
3. 计算相邻采样帧的 changed pixel ratio。
4. changed pixel ratio 低于 `diff_threshold` 的连续片段被视为稳定视觉片段。
5. 片段持续时间必须达到 `min_stable_duration`。
6. 每个稳定片段选择中间帧作为代表帧。
7. 使用 dHash 和 Hamming distance 去重，避免重复保存相同画面。
8. 将代表帧保存为 `stable_001_t0001.00s.jpg` 这类文件名。

同时会输出：

- `manifest.json`：视频时长、FPS、稳定片段、保存图片数等。
- `frame_diff_scores.csv`：每个采样点的差异分数，便于调试阈值。

### 4. 音频转写

音频转写由 `transcribe_video_audio()` 完成，使用 `faster-whisper`。

配置项包括：

- `asr_model`
- `asr_device`
- `asr_compute_type`

转写结果会保留：

- 识别语言。
- 语言概率。
- 每个语音片段的起止时间。
- 每个片段的文本。
- 拼接后的全文。

另外实现了 `repair_transcript_mojibake()`，用于修复部分中文 ASR 栈中常见的 UTF-8 被 GBK/CP936 错解码后的乱码。

### 5. Timeline Markdown

`render_video_timeline_markdown()` 会把稳定视觉片段和 ASR 结果合并成 Markdown：

```text
# Video Timeline: video/briefing.mp4

Source video: `video/briefing.mp4`
Duration: ...
Detected language: ...
Stable visual segments: ...
Saved stable frames: ...

## Visual And Speech Timeline

### Segment 1: 00:00.000 - 00:03.000
- Representative time: ...
- Stable frame: `video/briefing_stable_frames/stable_001_...jpg`
- Transcript in this visual segment:
  - [00:01.000 - 00:02.000] ...

## Full Transcript
...
```

这个 timeline 是视频题的主要文本入口。Agent 可以通过 `read_doc(video/xxx_timeline.md)` 读取完整转写和时间线，再结合附加的稳定帧图片做视觉验证。

### 6. 多模态模型请求注入

多模态注入逻辑位于 `src/data_agent_baseline/agents/multimodal.py`。

当任务存在 `video_timeline` 或 `video_preprocessing_failed` 资产时，初始用户消息会追加 `<video_context>` 文本，告诉模型：

- 原始视频已经预处理。
- 原始视频不会直接附加。
- 应同时使用 timeline、ASR transcript 和稳定帧。
- 如果 catalog 中没有完整 transcript，应调用 `read_doc` 读取 timeline。
- 附加的稳定帧图片按时间顺序排列。

同时，系统会把前 `max_attached_frames` 张稳定帧以 `image_url` data URL 的形式附加到初始请求中。为了避免 trace 泄露大段 base64，请求摘要中会统计图片数，但不会保留图片原始字节。

如果稳定帧超过上限，`<video_context>` 会说明还有多少张未附加。未附加图片仍然作为上下文文件存在，Agent 可以在需要时通过路径访问。

### 7. 失败降级

如果视频预处理失败，系统不会让整个任务直接失败，而是生成一个 `video_preprocessing_failed` 的 timeline Markdown，说明：

- 源视频路径。
- 失败原因。
- 原始视频没有被附加。
- 应使用其他可用上下文继续作答。

同时失败信息会写入 artifact bundle，便于后续复盘。

## 五、配置设计

视频预处理配置定义在 `VideoPreprocessingConfig`，并接入 YAML 配置文件：

```yaml
video_preprocessing:
  enabled: true
  sample_fps: 2.0
  diff_threshold: 0.025
  pixel_delta: 25
  min_stable_duration: 1.0
  resize_width: 320
  dedup: true
  hash_threshold: 4
  jpg_quality: 95
  max_attached_frames: 16
  asr_model: base
  asr_device: cpu
  asr_compute_type: int8
```

这些配置会进入：

- `prepare_task_context(..., video_config=config.video_preprocessing)`
- `LangGraphAgentConfig(max_attached_video_frames=...)`
- run summary 的 `video_preprocessing` 字段

PDF 转 Markdown 目前是默认启用的预处理行为，没有单独配置开关。

## 六、运行产物

每个任务执行时会在任务输出目录下生成：

```text
task_<id>/
  context_preprocessing_manifest.json
  generated_context/
    doc/report.md
    video/briefing_timeline.md
    video/briefing_stable_frames/*.jpg
  video_preprocessing/
    video__briefing.mp4/
      timeline.md
      transcript.json
      video_preprocessing_manifest.json
      stable_frames/*.jpg
```

`context_preprocessing_manifest.json` 记录每个可见资产与原始文件的对应关系。例如：

- 原始文件保留为 `action="source"`。
- PDF 生成文件为 `action="pdf_to_markdown"`。
- 视频时间线为 `action="video_timeline"`。
- 稳定帧为 `action="video_stable_frame"`。
- 视频失败占位文档为 `action="video_preprocessing_failed"`。

`generated_context/` 面向 Agent 和工具使用；`video_preprocessing/` 面向复盘和调试使用。

## 七、与 Phase1 代码的兼容方式

这两个设计都采用“预处理 + 资产视图”的方式，而不是侵入式改造主 Agent。

兼容点包括：

- `PublicTask.context_dir` 仍然指向原始 context 目录。
- 新增的 `TaskAssets.context_view` 只在工具解析路径时生效。
- 未经过预处理的普通文件仍以 `source` 资产透出。
- 文本工具不需要知道某个 `.md` 是原始文件还是 PDF 生成文件。
- Python 工具通过 `TaskContextWorkspace` 物化 overlay，看到的是预处理后的上下文树。
- Agent 主循环、答案写出、评分流程基本保持 Phase1 的运行形态。

因此当前改动更像是在 Phase1 前面加了一层异构文件适配器：把 PDF 和视频变成文本、图片与结构化 manifest，再交给原有 Agent 体系处理。

## 八、测试覆盖

相关测试主要在 `tests/test_context_preprocessor.py`、`tests/test_runner.py` 和 `tests/test_langgraph_agent.py` 中。

已覆盖的关键行为包括：

- PDF 有 TOC 时生成 Markdown 标题。
- PDF 无 TOC 时不主动推断标题。
- 中文硬换行会被合并。
- PDF 与已有同名 Markdown 冲突时生成 `_pdf.md`。
- `list_context_tree` 只列出预处理后的可见资产。
- overlay 工作区物化后不包含原始 PDF / 视频。
- 合成视频可抽取多个稳定片段。
- 视频文件会替换为 timeline 和 stable frames。
- 视频预处理产物会写入 artifact bundle。
- 初始多模态请求会附加稳定帧，但 trace 中不泄露图片 base64。
- `max_attached_frames` 控制附加图片数量。

## 九、设计边界与后续改进方向

当前 PDF 处理没有 OCR 和复杂版面理解，适合文本型 PDF；如果遇到扫描件、图片型报表或复杂表格，需要补充 OCR、表格抽取或局部页面截图理解。

当前视频处理采用稳定画面 + ASR 的轻量方案，适合幻灯片录屏、数据仪表盘演示、讲解类视频。对于快速运动、短暂细节、细粒度操作过程或强视觉推理视频，可能需要更密集采样、动作片段检测或按问题触发的二次抽帧。

当前 timeline 已把视觉片段和语音片段按时间窗口对齐，但 Agent 仍可能不主动读取完整 timeline。后续可以在 prompt 或流程层增加“视频题必须先读取 timeline，再结合稳定帧验证”的硬约束。

当前视频预处理依赖 OpenCV 和 faster-whisper，运行成本高于纯文本任务。并行跑 benchmark 时，需要关注 CPU、内存、模型加载和任务超时配置。
