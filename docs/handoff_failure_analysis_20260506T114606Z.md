# Handoff Generation Failure Analysis — Run 20260506T114606Z

**分析日期**: 2026-05-06
**Run ID**: `20260506T114606Z`
**任务数**: 23
**模型**: qwen3.5-35b-a3b
**最终结果**: 23/23 任务成功
**Data Inspector**: hybrid 模式, max_agent_steps=15, max_phase_retries=1

---

## 概述

Handoff 指 DataUnderstandingAgent（Step 2 `understand_and_explore_data`）生成的数据理解简报，包含字段映射、join 路径、answer contract 等信息，传递给主 solver 使用。

| Handoff 状态 | 数量 | 占比 |
|------------|:----:|:----:|
| complete | 19 | 82.6% |
| partial | 3 | 13.0% |
| fallback | 1 | 4.3% |

4 个任务（17.4%）的 handoff 生成存在失败或不完整，全部为 **medium** 难度。

---

## 影响对比

| 指标 | Complete Handoff | 非 Complete Handoff | 差异 |
|------|:---:|:---:|:---:|
| 平均步数 | 13.4 | 19.5 | **+45%** |
| 平均耗时 | 138.5s | 462.8s | **+234%** |

失败的 handoff 显著增加了主 solver 的推理步数和耗时。

---

## 分类 A: 合约验证失败 → fallback

### task_180

| 字段 | 值 |
|---|---|
| 难度 | medium |
| 问题 | For all the people who paid more than 29.00 per unit of product id No.5. Give their consumption status in the August of 2012. |
| Handoff | **fallback** |
| 合约阶段 | phase → rejected, retry → rejected, fallback |
| 步数/耗时 | 18 steps / 768.4s |

**验证错误**:

1. `Contract uses rejected fields: ['db/transactions_1k.db.transactions_1k.Amount', 'db/transactions_1k.db.transactions_1k.Date', 'db/transactions_1k.db.transactions_1k.Price']`
2. `answer_contract.answer_columns is empty despite an explicit requested output.`
3. `answer_contract.join_policy is unknown despite available join paths.`

**根因**: DataUnderstandingAgent 在生成 answer_contract 时引用了 3 个被领域知识标记为应避免使用的字段（Amount、Date、Price）。这些字段在 knowledge 中被标记为不可靠或语义上不适用于此查询。模型在 reasoning 中讨论了 `Price/Amount` 的比例计算，但在合约 JSON 中仍包含了这些字段。重试未能修正。

**结果**: fallback handoff 仍提供了字段映射和 join 路径，主 solver 在 18 步后成功完成任务，但耗时高达 768s（完整 handoff 均值的 5.5 倍）。

---

## 分类 B: 语义歧义 → partial

### task_243

| 字段 | 值 |
|---|---|
| 难度 | medium |
| 问题 | For the user No.24, how many times is the number of his/her posts compared to his/her votes? |
| Handoff | **partial** |
| 合约阶段 | phase → accepted (但有 3 个 remaining uncertainties) |
| 步数/耗时 | 24 steps / 155.7s |

**验证警告**:

1. `What PostTypeId values constitute a 'post' that should be counted? Values 1 and 2 dominate (42912 and 47755 occurrences), but semantics unclear.`
2. `What VoteTypeId values constitute a 'vote' that should be counted? Multiple distinct values exist (2, 5, 1, 3, 16, 15, 8, 9, 11, 10) but meanings unclear.`
3. `Does user 24 have any posts in the dataset? Need to verify existence before computing ratio (division by zero risk).`

**根因**: 数据集中 `PostTypeId` 和 `VoteTypeId` 的语义映射不在 knowledge 中。模型正确识别了歧义（PostTypeId=1 是 Question，2 是 Answer — "post" 应该包含哪种？），但无法在 handoff 阶段自行解决。这些是数据集的语义信息缺口，不是模型错误。

**结果**: 主 solver 拿到 partial handoff 后自行处理了歧义，在 24 步内完成。

---

## 分类 C: Join Policy 缺失 → partial

### task_257

| 字段 | 值 |
|---|---|
| 难度 | medium |
| 问题 | Identify the total views on the post 'Computer Game Datasets'. Name the user who posted it last time. |
| Handoff | **partial** |
| 合约阶段 | phase → rejected, retry → accepted（但 join_policy 仍缺失） |
| 步数/耗时 | 24 steps / 206.1s |

**验证警告**:

1. `answer_contract.join_policy is unknown despite available join paths.`

**根因**: 模型识别出了 3 条 join 路径（posts→users via OwnerUserId, posts→users via LastEditorUserId, posts→postHistory via PostId），但在最终合约中未能指定 `join_policy`（inner/left/outer）。合约在 phase 阶段被拒绝（`accepted=None`），retry 后被接受但 join_policy 仍为 unknown。

**结果**: 这是一个纯模型遗漏错误 — join 路径已确认，只是没有声明 join 类型。主 solver 补充了这部分决策。

---

## 分类 D: 工具基础设施限制 → partial

### task_283

| 字段 | 值 |
|---|---|
| 难度 | medium |
| 问题 | Calculate the percentage of superheroes with blue eyes. |
| Handoff | **partial** |
| 合约阶段 | phase → rejected, retry → accepted（但 2 个 uncertainties 仍存在） |
| 步数/耗时 | 12 steps / 721.0s |

**验证警告**:

1. `join_path between superhero and colour tables returned empty from find_join_paths; implicit FK relationship (eye_colour_id -> colour.id) appears valid but explicit join path not confirmed by tool`
2. `execute_probe_query failed due to table naming convention mismatch in probe layer`

**根因**: 两个工具基础设施问题：
- `find_join_paths` 工具无法发现 `superhero.eye_colour_id → colour.id` 的 join 关系，尽管这是有效的隐式外键
- `execute_probe_query` 因表命名约定不匹配而失败（probe 层的表名约定与实际数据库表名不匹配）

这两个都是工具层面的限制，不是模型推理错误。模型已知正确的关系，但工具无法验证。

**结果**: 虽然只有 12 步，但耗时 721s（几乎全是 DataUnderstandingAgent 在 probe/retry 中消耗的时间）。

---

## 补充发现: Contract Phase 被拒但恢复为 complete

2 个任务的合约阶段初稿被拒，但通过 `repair_or_critique` 修复后 handoff 成功标记为 complete：

| Task | Contract Phase | Repair | 最终 Handoff |
|------|:---:|:---:|:---:|
| task_287 | rejected | 成功修复 | complete |
| task_305 | rejected | 成功修复 | complete |

这属于正常的 contract → repair → complete 流程，不算 handoff 失败。

---

## 根因分类总结

| 根因类别 | 任务数 | 涉及任务 | 严重程度 |
|----------|:---:|------|:---:|
| **合约引用禁用字段** | 1 | task_180 | 高 (fallback) |
| **语义信息缺失** (PostTypeId/VoteTypeId 语义) | 1 | task_243 | 中 (partial) |
| **Join policy 遗漏** | 1 | task_257 | 中 (partial) |
| **工具限制** (find_join_paths / probe query) | 1 | task_283 | 中 (partial) |
| **Contract 初稿被拒但修复成功** | 2 | task_287, task_305 | 低 (正常流程) |

---

## 改善建议

1. **禁用字段的 retry prompt 增强**: task_180 的 retry 未能修正错误。建议在 retry prompt 中显式列出被拒绝的字段名及其替代字段，而非仅给出通用错误信息。

2. **枚举值语义注入**: task_243 的 PostTypeId/VoteTypeId 语义歧义如在 knowledge 中提供映射表（如 `PostTypeId=1 → Question, 2 → Answer`），模型即可在 handoff 阶段解决。

3. **Join policy 默认值**: task_257 的 join_policy 遗漏可设置默认为 `inner`，或在合约验证中接受 `unknown` 作为有效值（因为主 solver 可以在执行时决定）。

4. **find_join_paths 增强**: task_283 的 join 路径发现应支持基于命名约定推断隐式外键关系（如 `eye_colour_id` → `colour.id`），当前工具对此类关系的感知不足。

5. **Probe query 表名约定**: task_283 的探针查询失败说明 probe 层的表命名约定与实际数据结构存在不一致，需对齐。
