# ASR 评测基线：faster-whisper base 模型结果（2026-07-25）

## 结果定位

本文冻结当前项目首版 ASR 生产基线的全量评测结果，后续模型、Prompt 或后处理优化均应以本结果作为对照。

- 模型：`faster-whisper base`
- 运行设备：`CPU`
- 计算类型：`int8`
- 样本范围：项目当前 30 段 `briefing.mp4`
- 参考文本：Azure Fast Transcription `2025-10-15` 原始输出
- Gold ID：`azure-all-20260725`
- Run ID：`base-cpu-int8-all-20260725`

> 本文中的 CER/WER 表示候选转写与冻结 Azure 输出的一致程度，不是相对于人工逐字标注的绝对识别准确率。

## 核心结果

| 语言 | 样本数 | 指标 | Corpus-level | 宏平均 | S | D | I | 参考单元数 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 中文 | 20 | CER | **9.7899%** | 9.8438% | 566 | 13 | 36 | 6282 |
| 英文 | 10 | WER | **1.8822%** | 2.0672% | 26 | 4 | 9 | 2072 |

其他运行指标：

- 候选转写成功率：`30/30（100%）`
- 语言检测准确率：`30/30（100%）`
- 总转写耗时：`173.582 s`
- 平均每条耗时：`5.786 s`

## 高错误率样本

| Task | 语言 | 指标 | 错误率 | S | D | I |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `task_7` | 中文 | CER | 20.1521% | 51 | 2 | 0 |
| `task_39` | 中文 | CER | 18.1237% | 61 | 4 | 20 |
| `task_56` | 中文 | CER | 15.4167% | 29 | 0 | 8 |
| `task_1` | 中文 | CER | 14.7766% | 43 | 0 | 0 |
| `task_57` | 中文 | CER | 13.9535% | 27 | 1 | 2 |
| `task_60` | 英文 | WER | 6.0403% | 7 | 2 | 0 |

## 评测口径

- 双方文本统一执行 Unicode NFKC、英文小写、繁体转简体和标点清理。
- 中文保留汉字、英文字母和数字，以字符计算 CER。
- 英文保留字母和数字，以词计算 WER。
- 主指标使用各语言累计 S/D/I 后计算的 corpus-level CER/WER。
- 中文和英文分别报告，不合并为一个错误率。
- 本次没有候选转写失败；如发生失败，评分系统会按参考文本全部删除计入错误率。

## 原始产物

项目内原始产物位于：

- Gold manifest：`evaluation/asr/gold/azure-all-20260725/manifest.json`
- Run manifest：`artifacts/asr/runs/base-cpu-int8-all-20260725/manifest.json`
- 完整评分：`artifacts/asr/runs/base-cpu-int8-all-20260725/scores/azure-all-20260725/score.json`
- 逐样本结果：`artifacts/asr/runs/base-cpu-int8-all-20260725/scores/azure-all-20260725/per_sample.csv`
- 自动报告：`artifacts/asr/runs/base-cpu-int8-all-20260725/scores/azure-all-20260725/score_report.md`

`score.json` SHA-256：

```text
FC5A7255DBC3233AC02796012F7E8E2AC1C15DD6EB565B661DAC63DC709F1ABA
```

Gold 于 `2026-07-25T14:35:44Z` 创建，最后更新于 `2026-07-25T14:38:24Z`；base run 于 `2026-07-25T14:39:10Z` 创建，最后更新于 `2026-07-25T14:42:06Z`。

## 后续对比要求

后续实验应使用相同的 `azure-all-20260725` gold、相同的标准 WAV 和相同归一化规则，并使用新的 Run ID。只有这样，medium、术语 Prompt、UI 词表或后处理方案的结果才能与本 base 基线直接比较。
