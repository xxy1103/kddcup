## Pre-join View 优化方案分析

### 一、现状诊断

通过对 task_1 的完整链路分析（原始数据、semantic_catalog、trace.json），我梳理出以下关键数据：

**数据规模**

- 数据源：11 个 CSV、6 个 JSON、9 张 SQLite 表、5 份文档、1 个视频
- semantic_catalog 中的 schemas：25 张表/文件的完整字段定义（14,181 行 JSON）
- 自动分析出的 relationships：110 条，全部是 `lookup_code` 类型，cardinality 均为 `many_to_one`

**关系的拓扑结构**

110 条关系并非杂乱无章，而是呈现出清晰的星型模式：

| 维度表（target） | 被引用次数 | 行数 | 角色 |
|---|---|---|---|
| lc_exgindustry (行业分类) | 45 | 475 | 行业维度 |
| lc_actualcontroller (实控人) | 40 | 474 | 控制权维度 |
| lc_stockarchives (股票档案) | 25 | 500 | 股票主数据 |

连接键只有两种：`CompanyCode` 和 `SecuCode`。26 个事实数据源（CSV/JSON/SQLite表）通过这些键连接到 3 张维度表。

**Agent 执行链路分析（trace.json，26 步，92.5 秒）**

| 步骤区间 | Agent 在做什么 | 消耗的本质 |
|---|---|---|
| Step 1-5 | 探索数据全貌、看视频截图 | 上下文构建 |
| Step 6-11 | 逐个查看 lc_freefloat、lc_exgindustry 的表结构 | Schema 探索（重复消耗） |
| Step 12-13 | 构造 JOIN SQL 并执行 | 真正的业务逻辑 |
| Step 14-26 | 反复验证结果、repair 循环 | 不确定性导致的冗余操作 |

核心问题：Agent 花了大约 60% 的步骤在"理解表结构"和"搞清楚怎么连表"上，真正做题的时间只占不到 40%。而且 repair 步骤（模型没有调 tool 被系统打断）出现了 3 次，说明模型在多表 join 场景下的决策信心不足。

---

### 二、预建 View 方案的评估

你的思路方向是对的，但需要注意几个关键问题。

**方案的核心价值**

1. 消除 join 认知负担：Agent 不再需要知道 lc_freefloat 怎么连 lc_exgindustry，直接查一个宽表即可
2. 大幅减少上下文消耗：不需要把 110 条 relationships 和 25 张表的完整 schema 都给 Agent
3. 减少 Agent 出错概率：join 是 SQL 错误的高发区（选错连接键、内外连接搞混、字段名冲突）

**但存在三个核心风险**

风险 1：组合爆炸。26 个事实源 × 3 个维度表 = 最多 78 种组合。如果每种组合都建一个 view，管理成本极高，而且大多数 view 对当前题目毫无用处。

风险 2：过度膨胀的宽表。如果把一个事实表同时 join 3 个维度表，加上所有维度表的字段，列数会非常多。而且维度表之间可能有 1:N 关系（比如一个公司可能有多个实控人变更记录），join 后行数可能膨胀。

风险 3：丧失灵活性。如果题目需要某个非常规 join（比如 lc_freefloat join lc_dividend），预建 view 可能覆盖不到，Agent 还是要自己写 join。

---

### 三、推荐设计方案

我建议采用"分层预连接 + 智能上下文裁剪"的方案，而不是暴力全量 join。

#### 第一层：构建 3 张核心维度宽表（Dimension-Enriched Views）

对每张事实表，只预连接最常用的维度信息，且只保留必要字段：

```
对于每个事实表 T，生成：

T_enriched = T
  LEFT JOIN lc_exgindustry ei ON T.CompanyCode = ei.CompanyCode
  LEFT JOIN lc_stockarchives sa ON T.CompanyCode = sa.CompanyCode
```

具体实现方式（以 lc_freefloat 为例）：

```sql
CREATE VIEW v_lc_freefloat_enriched AS
SELECT
    ff.id,
    ff.CompanyCode,
    ff.SecuCode,
    ff.Chiname,
    ff.ChinameAbbr,
    ff.ChangeDate,
    ff.TotalAShare,
    ff.AFloats,
    ff.InactiveFloats,
    ff.AdjFreeFloatRatio,
    ff.AdjFreeFloats,
    -- 来自 lc_exgindustry
    ei.FirstIndustryName,
    ei.SecondIndustryName,
    -- 来自 lc_stockarchives
    sa.AStockCode,
    sa.State,
    sa.City
FROM lc_freefloat ff
LEFT JOIN lc_exgindustry ei ON ff.CompanyCode = ei.CompanyCode
LEFT JOIN lc_stockarchives sa ON ff.CompanyCode = sa.CompanyCode
```

关键设计原则：

1. 只 join 维度表（小表，475-500 行），不 join 其他事实表。维度表是".lookup"角色，事实表之间不应预连接。
2. 只附加高价值维度字段。比如 lc_exgindustry 的 FirstIndustryName、SecondIndustryName 是高频使用的，但 FourthIndustryCode（全为空）就没必要带。
3. 用 LEFT JOIN 而不是 INNER JOIN。确保不丢失事实表数据。

#### 第二层：智能上下文注入（Context Pruning）

当前 semantic_catalog 有 14,181 行，全量注入会浪费大量 token。建议改为分层注入：

**Level 0（始终注入，约 500 token）**：
- 所有 view 的名字 + 一句话描述 + 关键字段列表（不含 distinct_values、cardinality 等统计信息）
- 3 张维度表的简要说明

```yaml
views:
  - name: v_lc_freefloat_enriched
    description: "流通股数据（已含行业分类和股票主数据）"
    key_fields: [CompanyCode, SecuCode, ChangeDate, AFloats, TotalAShare, SecondIndustryName]
    
  - name: v_lc_dividend_enriched
    description: "分红数据（已含行业分类和股票主数据）"
    key_fields: [CompanyCode, SecuCode, DividendImplementDate, BonusShareRatio, SecondIndustryName]
```

**Level 1（按需注入，Agent 通过 tool 请求）**：
- 某张 view 的完整 schema（含 cardinality、distinct_values 等）
- 原始表的完整 schema（当预建 view 不够用时）

这样 Agent 在大多数情况下只看 Level 0 的概要就能决定查哪张 view，不需要遍历 25 张表的完整 schema。

#### 第三层：保留原始表和 relationships 作为 fallback

预建 view 不可能覆盖所有场景。保留以下能力：

1. Agent 可以查询原始表（通过 execute_probe_query）
2. Agent 可以调用 get_table_relationships 查看特定表的连接关系
3. semantic_catalog 中保留完整的 relationships 数据，但只在 Agent 主动搜索时才返回相关片段

---

### 四、具体实现步骤

1. **在 semantic_catalog 生成后、Agent 运行前**：
   - 解析 relationships，识别出维度表和事实表
   - 对所有事实表（CSV + SQLite 表）生成 enriched view 的 SQL
   - 将所有数据加载到统一的 SQLite 内存数据库中
   - 执行 CREATE VIEW 语句

2. **修改 Agent 的 system prompt / context injection**：
   - 用 view 的概要描述替代完整的 schemas + relationships
   - 告知 Agent "优先使用预建 view，它们已经包含了行业分类等维度信息"

3. **修改 Agent 的 tool 行为**：
   - get_table_profile：如果请求的是已有 view 的基表，返回 view 的信息
   - execute_probe_query：自动识别查询中涉及的表，如果有对应 view 则建议使用
   - 新增 list_available_views tool：列出所有预建 view 及其概要

---

### 五、对 task_1 的效果预估

如果采用此方案，task_1 的执行链路会变成：

| 步骤 | Agent 行为 | 对比原方案 |
|---|---|---|
| 1 | 看 Level 0 context，直接发现 v_lc_freefloat_enriched 包含 SecondIndustryName | 原：需要全量看 schema |
| 2 | 看视频截图，提取筛选条件 | 相同 |
| 3 | 直接写 SQL：SELECT SecondIndustryName, COUNT(DISTINCT CompanyCode) FROM v_lc_freefloat_enriched WHERE AFloats > 10000000000 AND ... GROUP BY SecondIndustryName | 原：需要先 get_table_profile × 2，再构造 JOIN |
| 4 | 执行并验证 | 相同 |

预计步骤从 26 步减少到 10-12 步，时间从 92 秒减少到 40-50 秒，上下文消耗减少约 60%。

---

### 六、需要注意的边界情况

1. **维度表一对多问题**：lc_actualcontroller 虽然是 474 行对应 474 个公司（1:1），但某些公司可能有历史变更记录导致 1:N。建 view 时需要做去重（如取最新记录或聚合）。

2. **跨事实表 join**：如果题目需要 lc_freefloat join lc_dividend（两者都是事实表），预建 view 无法覆盖。这种情况下仍需 Agent 自己写 join，但可以通过 relationships 信息辅助。

3. **JSON 数据源的 join**：JSON 文件需要先解析成表结构才能 join。建议在预处理阶段将 JSON 展开为 SQLite 表，然后再建 view。

4. **字段命名冲突**：事实表和维度表都有 ChiName、ChiNameAbbr、SecuCode 等同名字段，view 中需要重命名（如 `ei_ChiName` 或只保留事实表的版本）。

5. **数据量考虑**：当前数据量较小（几千行），join 后不会膨胀太多。但如果未来数据量增大，需要考虑物化 view（materialized view）而不是每次实时 join。
