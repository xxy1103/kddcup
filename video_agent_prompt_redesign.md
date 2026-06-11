## 视频总结 Agent 提示词重设计方案

### 一、问题诊断

当前 `VIDEO_UNDERSTANDING_SYSTEM_PROMPT`（位于 `video_understanding_agent.py` 第 26-55 行）要求 LLM 产出一份"concise Markdown document"，按主题维度组织（overview → key time ranges → visible clues → exact values → audible criteria → ASR uncertainty）。

这种设计的三个结构性缺陷：

1. **"concise" + "key" = 允许跳过**。LLM 被暗示可以省略它认为不重要的段落，直接导致 stable_001 被完全遗漏。
2. **按主题重组 = 打破时间线**。模型需要自己把段落映射到主题分类中，容易遗漏或合并段落。
3. **"Exact Extracted Values" 只捕获 answer-like 数值**。按钮标签（"干扰信息"、"收尾"）、颜色编码、提示性脚注等非数值元素被系统性忽略。

### 二、改进目标

用户要求的理想文档结构：**以时间线串联，每个 segment 包含语音文字 + 图片完整描述**。

具体规格：
- 严格按 segment 时间顺序，**每一个 stable frame 都必须覆盖，不得跳过**
- 语音转文字（transcript）原样保留，不做改写
- 图片描述必须详尽：所有可见 UI 元素（标题、按钮、标签、徽章、数据行、脚注、颜色编码）全部记录，不得选择性省略
- 保留当前的精确值提取规范（标点、分隔符、原文保留）
- 不确定性单独标注

### 三、代码改动

需要修改 `src/data_agent_baseline/run/video_understanding_agent.py` 中的三处：

#### 改动 1：`VIDEO_UNDERSTANDING_SYSTEM_PROMPT`（第 26-55 行）

```python
VIDEO_UNDERSTANDING_SYSTEM_PROMPT = """
You are a video evidence documentation agent for a data-analysis benchmark.
Your job is to produce an exhaustive, segment-by-segment record of the supplied
video timeline and its stable-frame images. You must NOT solve the user's data
task, infer answers, or draw conclusions. You only document what is seen and heard.

## Output Structure

Produce a single Markdown document. The document MUST be organized strictly in
chronological segment order — one section per stable frame listed in the timeline.
Do NOT group by topic, theme, or importance. Do NOT skip any segment, including
introductory, transitional, or seemingly unimportant ones.

For each segment, include exactly two parts:

### Part A — Transcript

Reproduce ALL transcript lines from the timeline document for this segment's time
window, exactly as written, with original timestamps. Do not paraphrase, summarize,
or omit any transcript text. If the timeline says "none detected" for this segment,
state "No speech in this segment."

### Part B — Complete Visual Description

Examine the stable-frame image for this segment and describe EVERY visible element
exhaustively. This is the most critical part of your work. You MUST cover:

- **Page headers / breadcrumbs / navigation**: all text in header areas, both primary
  and secondary titles.
- **All buttons and interactive elements**: every button, tab, toggle, or link visible
  anywhere on screen — including corners. Record exact label text, position (e.g.,
  "top-right corner"), and visual styling (color, background, border).
- **All informational cards and panels**: titles, body text, and any instructional or
  explanatory text within cards.
- **All data content**: table rows, lists, cards with data — record each entry with
  exact values, colors, and formatting.
- **All footnotes, disclaimers, and helper text**: any small-print text, notes, or
  instructions at the bottom of cards or pages. These are often the most important
  elements for downstream reasoning.
- **Status indicators, badges, and labels**: any tags, counters, progress indicators,
  or status text (e.g., "2/2", "进行中", "已保存").
- **Visual styling cues**: color coding (e.g., orange vs green text), selected/highlighted
  states (e.g., a tab with blue background and border), font weight differences (bold
  vs regular).
- **Annotations**: red boxes, arrows, circles, highlights, or any visual markers that
  draw attention to specific elements.

**Exactness rules:**
- Copy all visible text exactly as displayed, including Chinese text, English text,
  punctuation, separators, spaces, hyphens, parentheses, and line breaks.
- Do not translate, romanize, normalize, or reformat any text.
- If the exact text is unclear (blurry, partially obscured), mark it as [unclear: ...]
  rather than guessing.

If a segment's stable frame is marked as "not saved", "duplicate", or "unavailable",
note this and proceed to the next segment.

## After All Segments

After the segment-by-segment sections, you may add:

1. **Full Transcript Recap**: the complete transcript in chronological order (copied
   from the timeline's "Full Transcript" section), for convenient reading.
2. **Uncertainties**: a section listing any ASR/OCR uncertainties, illegible text,
   or contradictions observed across segments. Be specific: cite the segment number
   and stable-frame path for each uncertainty.

## Rules

- Do NOT solve, answer, or interpret the data-analysis task.
- Do NOT select, rank, or filter segments — cover every one.
- Do NOT group information by topic. Keep chronological segment order.
- Every stable frame image that is attached MUST appear in your output.
- Every button, badge, footnote, and disclaimer MUST be described — even if it seems
  trivial. Downstream agents rely on your completeness.
- State uncertainties explicitly. Do not present guesses as facts.
""".strip()
```

#### 改动 2：`_render_user_content()`（第 118-137 行）

```python
def _render_user_content(
    *,
    timeline_asset: ContextAsset,
    timeline_text: str,
    frame_assets: list[ContextAsset],
) -> list[dict[str, Any]]:
    # Build explicit image-to-index mapping so model cannot skip any
    if frame_assets:
        frame_list = "\n".join(
            f"- Image #{i + 1}: `{asset.visible_path}`"
            for i, asset in enumerate(frame_assets)
        )
    else:
        frame_list = "- none"

    text = (
        "Document the following preprocessed video evidence in exhaustive, "
        "segment-by-segment detail.\n\n"
        f"Source video: `{timeline_asset.source_path or 'unknown'}`\n"
        f"Timeline document path: `{timeline_asset.visible_path}`\n"
        f"Number of stable-frame images: {len(frame_assets)}\n\n"
        "Stable-frame images (attached in chronological order):\n"
        f"{frame_list}\n\n"
        "Timeline document content:\n"
        "```markdown\n"
        f"{timeline_text.strip()}\n"
        "```\n\n"
        f"The {len(frame_assets)} stable-frame image(s) are attached after this text "
        "in the same chronological order as listed above. "
        "You MUST produce a visual description for EACH image — do not skip any.\n\n"
        "For each segment in the timeline, write:\n"
        "1. The exact transcript lines for that segment's time window.\n"
        "2. A complete visual description of the corresponding stable-frame image, "
        "covering every visible UI element (titles, buttons, tabs, data rows, "
        "footnotes, badges, color coding, annotations, etc.)."
    )
    return [{"type": "text", "text": text}, *[_image_part(asset) for asset in frame_assets]]
```

#### 改动 3：`_success_summary_text()`（第 153-170 行）

```python
def _success_summary_text(
    *,
    raw_summary: str,
    timeline_path: str,
    frame_paths: tuple[str, ...],
) -> str:
    del frame_paths
    return (
        "# Video Understanding Summary\n\n"
        "This document was generated by the pre-main video understanding agent. "
        "It contains an exhaustive, segment-by-segment record of the video evidence. "
        "Explicit non-uncertain facts may be used directly as observed video evidence. "
        "Final answers should preserve exact visible values, punctuation, and separators "
        "from the summary's entries; if formatting is unclear, inspect the "
        "referenced stable frame directly with `read_context_image`.\n\n"
        f"- Original timeline: `{timeline_path}`\n\n"
        f"{raw_summary.strip()}\n"
    )
```

### 四、新旧对比

以 task_6 的 stable_001 和 stable_002 为例，展示期望输出的差异：

#### 旧版输出（实际产出）

直接跳过了 stable_001，stable_002 描述为：

```markdown
**From `stable_002_t0019.00s.jpg`:**
> **Header:** 近12个月
> **Range:** 2020-11-01 至 2021-10-31
> **Rows:**
> 1. `丹邦科技` | `2020-11` | `quantum carbon`
> 2. `吉峰科技` | `2020-11` | `repaying interest-bearing liabilities`
> **Footer Note:** `这个快捷范围会混入上一年度记录，不是本题批次。`
```

遗漏了：右上角蓝色"干扰信息"按钮、页面布局结构、标题层级。

#### 新版期望输出

```markdown
## Segment 1: 00:01.000 - 00:14.500

### Transcript

- [00:00.000 - 00:08.160] 好 今天主管 讓我把這個年度批次的定增項目整理一下 導出公司名稱和目自用途
- [00:08.160 - 00:14.840] 系統默認是近12個月 這個口徑不對 我得切到年度批次那邊去

### Visual Description (`video/briefing_stable_frames/stable_001_t0007.50s.jpg`)

**Page Layout:** Light gray background with a centered white card.

**Top-left text:**
- "股票增发审核工作台" — small, gray, non-bold.
- "年度定增审核批次" — large, bold, dark navy.

**Top-right button:**
- "工作台" — rounded rectangle, blue text on white background with subtle border.

**Centered informational card (white background, soft shadow, rounded corners):**
- **Card title:** "确认股票增发的年度受理批次" — bold, dark navy.
- **Card body:** "视频只定义批次年份和日期边界，完整公司及目的列表需要按该口径查询。" — smaller, gray text.

---

## Segment 2: 00:16.000 - 00:22.500

### Transcript

- [00:14.840 - 00:22.480] 你看 這個近12個月把相零批次的幾個項目也帶進來了 這不是我要的範圍
- [00:22.480 - 00:29.040] 批次管理 這裡才是按自然年度歸盪的 我找一下對應的那個批次

### Visual Description (`video/briefing_stable_frames/stable_002_t0019.00s.jpg`)

**Page Layout:** Light gray background with a white content card.

**Top-left text:**
- "股票增发审核工作台" — small, gray.
- "近12个月快捷筛选" — large, bold, dark navy.

**Top-right button:**
- **"干扰信息"** — blue text on white rounded rectangle with subtle border. This is a
  prominent button indicating this page contains interference/distractor information.

**Card content:**
- **Section title:** "近12个月" — bold, dark navy.
- **Date range:** "2020-11-01 至 2021-10-31" — regular gray text.
- **Data rows:**
  1. `丹邦科技` (orange, bold) | `2020-11` (gray) | `quantum carbon` (gray). Separator line below.
  2. `吉峰科技` (orange, bold) | `2020-11` (gray) | `repaying interest-bearing liabilities` (gray). Separator line below.
- **Footer note:** "这个快捷范围会混入上一年度记录，不是本题批次。" — gray, left-aligned.

---
```

### 五、关键改进点总结

| 维度 | 旧版 | 新版 |
|------|------|------|
| 组织结构 | 按主题分类（overview, clues, values...） | 按时间线逐段（Segment 1, 2, 3...） |
| 覆盖要求 | "concise" + "key" = 可跳过 | "MUST cover every segment" = 不得跳过 |
| 图片描述 | 选择性提取"answer-like"值 | 穷尽式描述所有 UI 元素 |
| 按钮/标签 | 系统性遗漏 | 明确要求"every button, badge, footnote" |
| 脚注/提示 | 归入"ASR Uncertainty"分类 | 作为图片描述的一部分，原位记录 |
| 语音文字 | 汇总到"Audible Criteria" | 每段原位保留，不汇总不改写 |
| 精确性规范 | 保留（标点/分隔符/原文） | 保留，并增加 [unclear:...] 标记 |
