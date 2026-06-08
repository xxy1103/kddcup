# Dimension-Enriched View 预建优化方案 — 技术设计文档

## 1  背景与问题陈述

### 1.1 当前系统架构

当前 data-agent 处理一个 task 的流程分为两个阶段。第一阶段是预处理：程序扫描 task 的 context 目录，对所有 CSV、JSON、SQLite 数据源进行 schema 分析，自动推断表间的 join 关系，生成 `semantic_catalog.json`。第二阶段是 Agent 运行：将 semantic_catalog 作为上下文注入 LLM，Agent 通过 tool call 探索数据、编写 SQL、执行查询，最终输出答案。

以 task_1 为例，这个流程的开销分布如下：

| 产出物 | 规模 | 说明 |
|--------|------|------|
| semantic_catalog.json | 14,181 行 | 25 个 schema 条目（18 个数据源 + 7 个文档），其中数据源展开后包含 26 张数据表（11 CSV + 6 JSON + SQLite 内 9 张子表）+ 110 条 relationships |
| global_data_profile.json | ~1,200 行 | 精简版 schema 概要，注入 Agent 初始上下文 |
| Agent trace | 26 步 / 92.5 秒 | 从开始到提交最终答案的完整执行链路 |

### 1.2 核心问题

**Agent 60% 的执行步骤消耗在"理解表结构"和"搞清楚怎么连表"上，而不是解决业务问题本身。**

通过 task_1 的 trace 分析，Agent 的行为可以划分为四个阶段：

1. **上下文构建**（Step 1-5）：浏览全局数据概要、看视频截图，理解题目含义。
2. **Schema 探索**（Step 6-11）：依次调用 `get_table_profile` 查看 lc_freefloat 和 lc_exgindustry 的完整字段定义，理解每张表有哪些列、数据类型、值分布。这些调用每次返回 3,000-4,000 token 的详细信息。
3. **构造查询**（Step 12-13）：编写 JOIN SQL 并执行。这一步才是真正解决问题的核心逻辑。
4. **结果验证**（Step 14-26）：由于对 join 的正确性缺乏信心，Agent 反复执行验证查询（检查数据分布、换用 ROW_NUMBER 窗口函数取最新记录、对比不同写法的结果），期间触发了 3 次 repair（模型没有调用 tool 被系统打断）。

如果 Agent 一开始就能看到一张已经包含行业分类字段的宽表（比如 `v_lc_freefloat_enriched`，自带 `SecondIndustryName` 列），那么阶段 2 可以完全跳过，阶段 4 的验证需求也会大幅减少——因为 Agent 不再需要担心 join 写对了没有。

### 1.3 问题的本质

问题的本质是 **信息组织方式与 Agent 的认知方式不匹配**。

当前的 semantic_catalog 是按照"原始数据源"组织的：每张表独立描述自己的 schema，表与表之间的关系作为独立的 metadata 条目（relationships 数组）存储。这种组织方式对人类数据库工程师来说很自然，但对 LLM Agent 来说却增加了认知负担——它需要先在脑中构建出"哪些表能通过什么键连起来"的图模型，然后才能写出正确的 SQL。

优化方向是：**把 Agent 最需要的信息组织方式前置——直接提供已经连好的宽表，让 Agent 看到的就是它能直接用的。**

---

## 2  数据拓扑分析

### 2.1 关系结构的星型模式

对 task_1 的 110 条 relationships 进行拓扑分析，发现它们呈现出高度规整的星型模式：

```
                         ┌──────────────────────┐
   事实表 (23个)          │   维度表 (3个)         │
                         │                      │
   lc_freefloat ─────────┤                      │
   lc_dividend ──────────┤  lc_exgindustry ◄────┤ 被引用 45 次
   lc_ashareipobid ──────┤  (行业分类, 475行)     │
   lc_sharefp ───────────┤                      │
   qt_dailyquote ────────┤                      │
   ...                   │                      │
                         │                      │
   lc_freefloat ─────────┤                      │
   lc_dividend ──────────┤  lc_actualcontroller ◄┤ 被引用 40 次
   ...                   │  (实控人, 474行)       │
                         │                      │
                         │                      │
   lc_freefloat ─────────┤                      │
   lc_dividend ──────────┤  lc_stockarchives ◄───┤ 被引用 25 次
   ...                   │  (股票档案, 500行)     │
                         └──────────────────────┘
```

所有 110 条关系的方向一致：事实表（多端，source）指向维度表（一端，target），cardinality 均为 `many_to_one`，连接键只有两种：`CompanyCode` 和 `SecuCode`。

### 2.2 维度表与事实表的区分依据

维度表（Dimension Table）是描述实体属性的参考数据——行数少、变化缓慢、被大量其他表引用。事实表（Fact Table）是记录业务事件的流水数据——行数多、持续增长、引用维度表获取补充信息。

在 task_1 中：

| 表 | 行数 | 被引用次数 | 连接键唯一性 | 角色 |
|----|------|-----------|-------------|------|
| lc_exgindustry | 475 | 45 | CompanyCode 唯一 | 维度表（行业） |
| lc_actualcontroller | 474 | 40 | CompanyCode 唯一 | 维度表（实控人） |
| lc_stockarchives | 500 | 25 | CompanyCode 唯一 | 维度表（股票主数据） |
| lc_freefloat | 3,119 | 0 | CompanyCode 100个不同值 / 3,119行 | 事实表 |
| lc_dividend | 2,685 | 0 | CompanyCode 434个不同值 / 2,685行 | 事实表 |
| lc_ashareipobid | 7,652 | 0 | CompanyCode 3个不同值 / 7,652行 | 事实表 |

### 2.3 跨领域的拓扑差异

60 个 task 横跨 4 个数据领域，各领域的拓扑特征不同：

| 领域 | 任务数 | 维度表清晰度 | 说明 |
|------|--------|-------------|------|
| A股 (Stock) | ~15 | 高 | 典型星型模式，3 张小维度表被大量引用 |
| 公募基金 (Fund) | ~24 | 中 | 存在维度表但引用模式更分散 |
| 宏观经济 (Macro) | ~14 | 低 | 时间序列为主，维度/事实边界模糊 |
| 医疗 (MIMIC) | ~7 | 高 | 明确的维度表（patients, admissions），但可能存在链式 join |

这一差异意味着自动分类算法不能硬编码规则，而需要基于可量化的信号进行打分。

---

## 3  设计目标与约束

### 3.1 设计目标

| 目标 | 量化指标 | 说明 |
|------|---------|------|
| 减少 Agent 执行步骤 | 步骤数减少 ≥ 30% | Agent 不再需要逐步探索表结构和构造 join |
| 减少上下文 token 消耗 | 初始 context 减少 ≥ 50% | 用精简的 view 概要替代完整的 schema + relationships |
| 提高 Agent 答题准确率 | 准确率提升 ≥ 10% | 减少因 join 错误导致的错误答案 |
| 零人工干预 | 全自动运行 | 程序自动判断哪些表该连、怎么连，无需人工标注 |

### 3.2 设计约束

1. **不丢失信息**：原始表必须保留可访问，预建 view 是增量能力而非替代。
2. **不引入数据错误**：view 构建过程必须有安全校验，不能因为 join 导致数据膨胀或丢失。
3. **跨领域通用**：同一套代码处理 A股、基金、宏观经济、医疗四个领域，不做领域特定硬编码。
4. **性能可控**：view 构建时间不超过 30 秒，view 查询响应时间与原表查询相当。

---

## 4  整体架构

### 4.1 在现有 Pipeline 中的位置

```
┌──────────────────────────────────────────────────────────────┐
│                    当前 Pipeline                              │
│                                                              │
│  原始数据 ──► Schema 分析 ──► Relationship 推断               │
│                                    │                         │
│                                    ▼                         │
│                             semantic_catalog.json            │
│                                    │                         │
│                                    ▼                         │
│                        global_data_profile 生成              │
│                                    │                         │
│                                    ▼                         │
│                           Agent 启动 & 运行                  │
│                                    │                         │
│                                    ▼                         │
│                              答案输出 & 校验                  │
└──────────────────────────────────────────────────────────────┘
```

```
┌──────────────────────────────────────────────────────────────┐
│                    优化后 Pipeline                            │
│                                                              │
│  原始数据 ──► Schema 分析 ──► Relationship 推断               │
│                                    │                         │
│                                    ▼                         │
│                             semantic_catalog.json            │
│                                    │                         │
│                          ┌─────────┴──────────┐              │
│                          ▼                    ▼              │
│                  【新增】维度表/事实表     【新增】数据统一      │
│                    自动分类打分            加载到 SQLite       │
│                          │                    │              │
│                          ▼                    ▼              │
│                  【新增】View 生成 & 安全校验                  │
│                          │                                   │
│                          ▼                                   │
│                  【新增】精简 Context 生成                     │
│                  (view_summary 替代全量 schema)               │
│                          │                                   │
│                          ▼                                   │
│                   Agent 启动 & 运行                          │
│                   (使用 view 直接查询)                        │
│                          │                                   │
│                          ▼                                   │
│                    答案输出 & 校验                             │
└──────────────────────────────────────────────────────────────┘
```

核心变化是在 semantic_catalog 生成之后、Agent 启动之前，插入了四个新环节。

### 4.2 模块划分

新增四个模块，按执行顺序排列：

| 模块 | 输入 | 输出 | 职责 |
|------|------|------|------|
| **TableClassifier** | semantic_catalog.relationships + schemas | 维度表列表 + 事实表列表 | 基于打分自动区分维度表和事实表 |
| **ViewBuilder** | 分类结果 + 原始数据文件 | SQLite 内存数据库中的 view 定义 | 生成 enriched view 并执行安全校验 |
| **ContextPruner** | semantic_catalog + view 定义 | 精简后的 context injection | 生成 view_summary 替代完整 schema |
| **ToolAdapter** | Agent 的 tool call 请求 | 路由到 view 或原始表 | 让 Agent 的 tool 透明地使用 view |

---

## 5  模块一：TableClassifier — 自动维度表识别

### 5.1 设计思路

TableClassifier 的目标是：给定一组表和它们之间的 relationships，自动判断哪些表是维度表（适合被 join 进来），哪些表是事实表（应该作为主表被查询）。

不做硬性的二分类，而是为每张表计算一个**维度性得分（dimension_score）**，得分越高越像维度表。最终根据阈值和排名决定分类。

### 5.2 打分信号与权重

从 semantic_catalog 已有的数据中可以提取五个信号：

#### 信号 1：被引用频次（target_ref_count）— 权重 0.40

**含义**：在 relationships 数组中，该表作为 target 出现的次数。

**原理**：维度表的本质特征是被大量其他表引用以获取描述性信息。被引用次数越多，越可能是维度表。

**数据获取**：遍历 `semantic_catalog.relationships`，对每条 relationship 的 `target.asset_path + target.table` 计数。

**task_1 实证**：lc_exgindustry 被引用 45 次，lc_actualcontroller 被引用 40 次，lc_stockarchives 被引用 25 次。所有事实表被引用次数为 0。

**归一化**：

```
norm_ref = target_ref_count / max(all_target_ref_counts)
```

#### 信号 2：行数反比（1 - norm_row_count）— 权重 0.20

**含义**：表的行数越少，越可能是维度表。

**原理**：维度表描述的是有限的实体集合（公司、行业、地区），行数通常在几百到几千之间。事实表记录事件流水，行数通常远大于维度表。

**数据获取**：从 `semantic_catalog.schemas` 中读取每张表的 `row_count`。

**归一化**：

```
norm_row = row_count / max(all_row_counts)
signal_row = 1 - norm_row
```

**注意**：这个信号单独使用不可靠（小表也可能是事实表的子集），所以权重较低。

#### 信号 3：连接键唯一性（join_key_uniqueness）— 权重 0.20

**含义**：该表在 relationship 中作为 target 时，连接键字段的唯一性比例。

**原理**：维度表的连接键通常是主键或准主键（每个公司只有一条记录），唯一性接近 1.0。事实表的连接键会有大量重复（同一个公司有很多条事件记录）。

**数据获取**：从 `relationship.evidence.target_uniqueness_ratio` 直接读取。如果一张表在多条 relationship 中作为 target，取平均值。

**task_1 实证**：三张维度表的 `target_uniqueness_ratio` 均为 1.0。

#### 信号 4：时间字段密度反比（1 - time_density）— 权重 0.10

**含义**：表中高基数时间字段的占比越低，越可能是维度表。

**原理**：事实表通常包含精确到日甚至到秒的时间字段（如 ChangeDate、TradeDate），且不同值的数量接近行数。维度表通常没有时间字段，或只有 InfoPublDate 之类的低基数时间字段。

**数据获取**：遍历表的字段，检测 type 为 TIMESTAMP 且 cardinality / row_count > 0.5 的字段数量。

```
time_density = count(high_cardinality_timestamp_fields) / total_fields
```

#### 信号 5：数值字段比例反比（1 - numeric_ratio）— 权重 0.10

**含义**：表中 DOUBLE/REAL 类型字段的占比越低，越可能是维度表。

**原理**：事实表包含大量度量字段（金额、数量、比率），维度表以描述性文本字段为主（名称、分类、编码）。

**数据获取**：统计 type 为 DOUBLE、REAL、BIGINT（排除 id 和连接键）的字段占比。

```
numeric_ratio = count(numeric_measure_fields) / total_fields
```

### 5.3 综合打分公式

```python
def compute_dimension_score(table, relationships, schemas):
    """
    计算一张表的维度性得分。
    返回 0.0 ~ 1.0 之间的浮点数，越高越像维度表。
    """
    # 信号 1: 被引用频次
    ref_count = count_as_target(table, relationships)
    max_ref = max(count_as_target(t, relationships) for t in all_tables)
    norm_ref = ref_count / max_ref if max_ref > 0 else 0

    # 信号 2: 行数反比
    row_count = get_row_count(table, schemas)
    max_rows = max(get_row_count(t, schemas) for t in all_tables)
    norm_row = 1 - (row_count / max_rows) if max_rows > 0 else 0

    # 信号 3: 连接键唯一性
    uniqueness = avg_target_uniqueness(table, relationships)

    # 信号 4: 时间字段密度反比
    time_density = compute_time_density(table, schemas)

    # 信号 5: 数值字段比例反比
    numeric_ratio = compute_numeric_ratio(table, schemas)

    # 加权求和
    score = (
        0.40 * norm_ref +
        0.20 * norm_row +
        0.20 * uniqueness +
        0.10 * (1 - time_density) +
        0.10 * (1 - numeric_ratio)
    )

    return score
```

### 5.4 分类决策

```python
DIMENSION_THRESHOLD = 0.50  # 可调参数

dimension_tables = []
fact_tables = []

for table in all_tables:
    score = compute_dimension_score(table, relationships, schemas)
    if score >= DIMENSION_THRESHOLD:
        dimension_tables.append((table, score))
    else:
        fact_tables.append((table, score))

# 额外约束: 维度表行数不超过事实表中位数的 2 倍
# （防止把行数很大的事实表误判为维度表）
median_fact_rows = median(get_row_count(t) for t, _ in fact_tables)
dimension_tables = [
    (t, s) for t, s in dimension_tables
    if get_row_count(t) <= median_fact_rows * 2
]
```

### 5.5 task_1 的预期打分结果

> 注：以下数值为近似估算，实际值取决于 `max_row_count` 的取值范围（含/不含 SQLite 内部表）以及各信号的精确计算。

| 表 | norm_ref | norm_row | uniqueness | 1-time | 1-numeric | **总分** | 分类 |
|----|----------|----------|------------|--------|-----------|----------|------|
| lc_exgindustry | 1.00 | ~0.94 | 1.00 | ~0.88 | ~0.78 | **~0.95** | 维度表 |
| lc_actualcontroller | 0.89 | ~0.94 | 1.00 | ~1.00 | ~0.86 | **~0.93** | 维度表 |
| lc_stockarchives | 0.56 | ~0.94 | 1.00 | ~1.00 | ~0.89 | **~0.80** | 维度表 |
| lc_ashareipobid | 0.00 | ~0.00 | — | ~0.25 | ~0.20 | **~0.05** | 事实表 |
| lc_freefloat | 0.00 | ~0.60 | — | ~0.25 | ~0.22 | **~0.17** | 事实表 |
| lc_dividend | 0.00 | ~0.65 | — | ~0.38 | ~0.25 | **~0.19** | 事实表 |

三张维度表得分均在 0.80 以上，所有事实表得分均在 0.20 以下，区分度非常好。以 DIMENSION_THRESHOLD = 0.50 为阈值可以完美区分。

---

## 6  模块二：ViewBuilder — 预建 Enriched View

### 6.1 设计思路

ViewBuilder 的职责是：对每张事实表，把它和所有维度表通过已有的 relationship 连接起来，生成一张"enriched"宽表视图。

核心设计原则是**只连维度表，不连事实表**。事实表之间的 join 留给 Agent 按需自行构造，因为事实表 × 事实表的 join 组合不可预测，且不同题目的需求完全不同。

### 6.2 数据统一加载

当前数据分散在三种存储中：CSV 文件、JSON 文件、SQLite 数据库。要建 view，首先需要将它们统一到同一个 SQLite 引擎中。

```python
import sqlite3
import pandas as pd
import json

def create_unified_db(task_context_dir, schemas):
    """
    将所有数据源加载到一个 SQLite 内存数据库中。
    返回 sqlite3.Connection 对象。
    """
    conn = sqlite3.connect(':memory:')

    for schema in schemas:
        asset_path = schema['asset_path']
        kind = schema['kind']
        table_name = sanitize_table_name(asset_path)  # 如 "csv__lc_freefloat"

        if kind == 'csv':
            df = pd.read_csv(f"{task_context_dir}/{asset_path}")
            df.to_sql(table_name, conn, index=False)

        elif kind == 'json':
            with open(f"{task_context_dir}/{asset_path}", encoding='utf-8') as f:
                data = json.load(f)
            # JSON 结构通常是 {"records": [...]} 或直接是 [...]
            records = data.get('records', data) if isinstance(data, dict) else data
            df = pd.json_normalize(records)
            df.to_sql(table_name, conn, index=False)

        elif kind == 'sqlite':
            # 从原始 SQLite 文件中逐表复制到内存数据库
            src_conn = sqlite3.connect(f"{task_context_dir}/{asset_path}")
            for (tbl_name,) in src_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall():
                df = pd.read_sql(f'SELECT * FROM "{tbl_name}"', src_conn)
                df.to_sql(tbl_name, conn, index=False)
            src_conn.close()

    return conn
```

### 6.3 View 生成算法

对每张事实表，算法需要决定：join 哪些维度表、用什么键、选哪些字段。

```python
def generate_enriched_views(conn, fact_tables, dimension_tables, relationships):
    """
    为每张事实表生成 enriched view 的 CREATE VIEW SQL。
    返回 {view_name: sql} 字典。
    """
    views = {}

    for fact_table in fact_tables:
        fact_name = fact_table['name']

        # 找出该事实表与哪些维度表存在 relationship
        relevant_rels = [
            r for r in relationships
            if get_source_table(r) == fact_name
               and get_target_table(r) in [d['name'] for d in dimension_tables]
        ]

        if not relevant_rels:
            # 没有可连接的维度表，跳过
            continue

        # 按维度表去重（一个事实表可能通过 CompanyCode 和 SecuCode
        # 两个键连接同一个维度表，只取 cardinality 更高的那个）
        dim_join_map = deduplicate_joins(relevant_rels)

        # 构建 SELECT 字段列表
        select_cols = []
        from_clause = f'"{fact_name}" f'

        # 1. 事实表自身的所有字段
        fact_columns = get_column_names(conn, fact_name)
        for col in fact_columns:
            # 跳过纯自增 id 列（减少噪音）
            if col.lower() == 'id':
                continue
            select_cols.append(f'f."{col}"')

        # 2. 每张维度表的附加字段
        join_clauses = []
        for idx, (dim_table, join_key, rel) in enumerate(dim_join_map):
            alias = f'd{idx}'
            dim_columns = get_column_names(conn, dim_table)

            # 只选取高价值字段，跳过：id、连接键、与事实表重名的描述字段
            fact_col_lower = {c.lower() for c in fact_columns}
            skip_cols = {'id', join_key.lower()}
            # 跳过维度表中与事实表重名的字段（如 ChiName、SecuCode 等），避免语义混淆
            for c in dim_columns:
                if c.lower() in fact_col_lower and c.lower() != join_key.lower():
                    skip_cols.add(c.lower())

            for col in dim_columns:
                if col.lower() not in skip_cols:
                    select_cols.append(f'{alias}."{col}" AS "{dim_table}_{col}"')

            join_clauses.append(
                f'LEFT JOIN "{dim_table}" {alias} '
                f'ON f."{join_key}" = {alias}."{join_key}"'
            )

        # 组装完整 SQL
        view_name = f'v_{fact_name}_enriched'
        sql = (
            f'CREATE VIEW "{view_name}" AS\n'
            f'SELECT\n  {",\n  ".join(select_cols)}\n'
            f'FROM {from_clause}\n'
            f'{" ".join(join_clauses)}'
        )

        views[view_name] = sql

    return views
```

### 6.4 以 task_1 的 lc_freefloat 为例

生成的 SQL 会是：

```sql
CREATE VIEW "v_lc_freefloat_enriched" AS
SELECT
  f."CompanyCode",
  f."ChiName",
  f."ChiNameAbbr",
  f."SecuCode",
  f."ChangeDate",
  f."TotalAShare",
  f."AFloats",
  f."InactiveFloats",
  f."AdjFreeFloatRatio",
  f."AdjFreeFloats",
  d0."FirstIndustryName"  AS "lc_exgindustry_FirstIndustryName",
  d0."SecondIndustryName" AS "lc_exgindustry_SecondIndustryName",
  d0."InfoPublDate"       AS "lc_exgindustry_InfoPublDate",
  d1."AStockCode"         AS "lc_stockarchives_AStockCode",
  d1."State"              AS "lc_stockarchives_State",
  d1."City"               AS "lc_stockarchives_City",
  d1."LegalRepr"          AS "lc_stockarchives_LegalRepr",
  d2."ControllerName"     AS "lc_actualcontroller_ControllerName",
  d2."NationalityDesc"    AS "lc_actualcontroller_NationalityDesc"
FROM "lc_freefloat" f
LEFT JOIN "lc_exgindustry" d0 ON f."CompanyCode" = d0."CompanyCode"
LEFT JOIN "lc_stockarchives" d1 ON f."CompanyCode" = d1."CompanyCode"
LEFT JOIN "lc_actualcontroller" d2 ON f."CompanyCode" = d2."CompanyCode"
```

有了这张 view，task_1 的查询变成了：

```sql
SELECT
  lc_exgindustry_SecondIndustryName AS "二级行业",
  COUNT(DISTINCT CompanyCode) AS "公司数目"
FROM v_lc_freefloat_enriched
WHERE AFloats > 10000000000
  AND strftime('%Y', ChangeDate) = '2019'
GROUP BY lc_exgindustry_SecondIndustryName
```

Agent 不再需要知道 lc_exgindustry 的存在，也不需要写 JOIN。

### 6.5 安全校验

View 创建后，必须执行以下校验以确保数据质量：

```python
def validate_view(conn, view_name, fact_table_name):
    """
    校验 enriched view 的数据完整性。
    """
    issues = []

    # 检查 1: 行数不膨胀
    fact_count = conn.execute(
        f'SELECT COUNT(*) FROM "{fact_table_name}"'
    ).fetchone()[0]
    view_count = conn.execute(
        f'SELECT COUNT(*) FROM "{view_name}"'
    ).fetchone()[0]

    inflation_ratio = view_count / fact_count if fact_count > 0 else float('inf')

    if inflation_ratio > 1.05:
        # 行数膨胀超过 5%，说明维度表连接键不唯一（1:N 问题）
        issues.append(
            f"Row inflation: {fact_count} -> {view_count} "
            f"({inflation_ratio:.1%} increase). "
            f"Dimension table may have non-unique join keys."
        )

    # 检查 2: 关键维度字段非空率
    view_cols = get_column_names(conn, view_name)
    # 动态检测维度表字段前缀（字段名格式: {dim_table}_{original_col}）
    dim_prefixes = {f"{d['name']}_" for d in dimension_tables}
    dim_cols = [
        c for c in view_cols
        if any(c.startswith(prefix) for prefix in dim_prefixes)
    ]

    for col in dim_cols:
        non_null = conn.execute(
            f'SELECT COUNT("{col}") FROM "{view_name}" WHERE "{col}" IS NOT NULL'
        ).fetchone()[0]
        fill_rate = non_null / view_count if view_count > 0 else 0

        if fill_rate < 0.50:
            # 维度字段填充率低于 50%，说明 join 匹配度差
            issues.append(
                f"Low fill rate for {col}: {fill_rate:.0%}. "
                f"Join key may have poor coverage."
            )

    return issues
```

如果校验不通过，策略是：

- **行数膨胀**：对该维度表的连接键做去重（取 GROUP BY 后出现频次最高的记录，或取最新一条），然后重建 view。
- **填充率过低**：保留 view 但在 context 中标注该维度字段的 join 覆盖率，让 Agent 知晓。

### 6.6 维度表 1:N 膨胀问题的处理

某些维度表可能存在 1:N 的情况。比如 lc_actualcontroller 中，一家公司可能在不同时间有不同的实控人。处理方式是在 join 前先对维度表做去重：

```python
def deduplicate_dimension(conn, dim_table, join_key):
    """
    对维度表按连接键去重，保留每个键值的最新记录。
    创建一个去重后的临时表。
    """
    # 检查是否有时间字段可用于排序
    columns = get_column_names(conn, dim_table)
    time_cols = [c for c in columns if 'date' in c.lower() or 'time' in c.lower()]

    dedup_table = f"_dedup_{dim_table}"

    if time_cols:
        # 有时间字段：取每个 join_key 的最新记录
        order_col = time_cols[0]
        conn.execute(f"""
            CREATE TEMP TABLE "{dedup_table}" AS
            SELECT * FROM "{dim_table}"
            WHERE rowid IN (
                SELECT MAX(rowid) FROM "{dim_table}"
                GROUP BY "{join_key}"
            )
        """)
    else:
        # 无时间字段：取每个 join_key 的第一条记录
        conn.execute(f"""
            CREATE TEMP TABLE "{dedup_table}" AS
            SELECT * FROM "{dim_table}"
            WHERE rowid IN (
                SELECT MIN(rowid) FROM "{dim_table}"
                GROUP BY "{join_key}"
            )
        """)

    return dedup_table
```

---

## 7  模块三：ContextPruner — 精简上下文注入

### 7.1 设计思路

当前 Agent 启动时接收的 `global_data_profile.json` 包含所有表的完整 schema（字段名、类型、distinct_values 列表等），总量约 46,000 token。加上 semantic_catalog 中的 relationships 信息，初始上下文可能超过 60,000 token。

ContextPruner 的目标是：用 view 的精简概要替代原始的全量 schema，将初始上下文压缩到 5,000-10,000 token 以内。

### 7.2 分层注入策略

将信息分为三个层级，Agent 在初始上下文中只看到 Level 0，更详细的信息通过 tool call 按需获取。

#### Level 0：View 概览（始终注入，约 1,500-3,000 token）

注入内容：

```yaml
# ═══════════════════════════════════════════
# 可用数据视图（已预连接行业/地域等维度信息）
# ═══════════════════════════════════════════

enriched_views:
  - name: v_lc_freefloat_enriched
    description: "流通股变动数据，已含行业分类(FirstIndustryName, SecondIndustryName)、股票档案(AStockCode, State, City)、实控人(ControllerName)"
    base_table: lc_freefloat
    row_count: 3119
    fields:
      - CompanyCode (BIGINT) — 公司编码，join key
      - SecuCode (VARCHAR) — 证券代码
      - ChangeDate (TIMESTAMP) — 变更日期
      - TotalAShare (DOUBLE) — A股总股本
      - AFloats (DOUBLE) — 流通A股股本
      - InactiveFloats (DOUBLE) — 非流通A股
      - AdjFreeFloatRatio (DOUBLE) — 自由流通比例
      - AdjFreeFloats (DOUBLE) — 自由流通股本
      - lc_exgindustry_FirstIndustryName (VARCHAR) — 一级行业
      - lc_exgindustry_SecondIndustryName (VARCHAR) — 二级行业
      - lc_stockarchives_AStockCode (VARCHAR) — A股代码
      - lc_stockarchives_State (VARCHAR) — 省份
      - lc_actualcontroller_ControllerName (VARCHAR) — 实控人名称

  - name: v_lc_dividend_enriched
    description: "分红数据，已含行业分类、股票档案、实控人信息"
    base_table: lc_dividend
    row_count: 2685
    fields:
      - CompanyCode (BIGINT)
      - SecuCode (VARCHAR)
      - DividendImplementDate (TIMESTAMP)
      - BonusShareRatio (DOUBLE)
      - ...（同上模式）

  # ... 其他 enriched views

# 原始维度表（如需独立查询）
dimension_tables:
  - lc_exgindustry (475 rows) — 行业分类
  - lc_actualcontroller (474 rows) — 实际控制人
  - lc_stockarchives (500 rows) — 股票档案

# 未建 view 的独立表
standalone_tables:
  - lc_ashareipobid (7652 rows) — IPO申购明细
  - qt_dailyquote (行数较大) — 日行情数据
```

**关键设计点**：

1. 每个 view 只列出**与业务相关的核心字段**（跳过 id、ChiName、ChiNameAbbr 等描述性冗余字段），并附带类型和中文语义说明。
2. 不注入 distinct_values、cardinality、min/max 等统计信息——这些信息通过 `get_table_profile` tool 按需获取。
3. 标注 `base_table`，让 Agent 知道这个 view 的数据来源。
4. 列出 `standalone_tables`（没有建 view 的表），让 Agent 知道还有哪些数据可用。

#### Level 1：按需详细 Schema（通过 tool call 获取）

Agent 调用 `get_table_profile("v_lc_freefloat_enriched")` 时返回完整字段信息，包括 cardinality、distinct_values、min/max 等。

Agent 调用 `get_column_distinct_values("v_lc_freefloat_enriched", "lc_exgindustry_SecondIndustryName")` 时返回该列的所有不同值及计数。

#### Level 2：关系追溯（通过 tool call 获取）

Agent 调用 `get_table_relationships("lc_freefloat")` 时返回该表与哪些表有 join 关系。这是 fallback 能力，用于 Agent 需要跨事实表 join 的场景。

### 7.3 对比：优化前后的上下文消耗

| 项目 | 优化前 | 优化后 |
|------|--------|--------|
| 初始注入的 schema 信息 | ~46,000 token（25 张表完整 schema） | ~2,500 token（view 概览） |
| Relationships 信息 | ~8,000 token（110 条关系） | 0 token（不注入，按需获取） |
| Agent 主动探索 schema 的 tool call | 通常 3-5 次，每次 ~4,000 token | 通常 0-1 次 |
| **总上下文消耗（估算）** | **~70,000 token** | **~10,000 token** |

---

## 8  模块四：ToolAdapter — Agent 工具层适配

### 8.1 设计思路

ToolAdapter 让 Agent 现有的 tool 能够透明地使用 enriched view，而不需要修改 Agent 的 tool call 逻辑。

### 8.2 各 Tool 的适配方式

#### execute_probe_query / execute_context_sql

**当前行为**：Agent 写 SQL，直接查原始表。

**适配后行为**：SQL 仍然直接执行。但如果 Agent 查询的表有对应的 enriched view，系统在返回结果前附加一条提示：

```json
{
  "ok": true,
  "results": [...],
  "hint": "Note: 'lc_freefloat' has an enriched view 'v_lc_freefloat_enriched' that includes SecondIndustryName, FirstIndustryName, State, City, ControllerName. Consider using the view to avoid manual JOINs."
}
```

这是一种**渐进式引导**而非强制替换——Agent 可以继续查原始表，但会被提示有更便捷的选项。

#### get_table_profile

**当前行为**：返回某张表的完整 schema。

**适配后行为**：如果请求的表有对应的 enriched view，优先返回 view 的 schema，并标注"这是预连接了维度信息的宽表视图"。如果 Agent 明确请求原始表（通过原始表名），仍然返回原始表的 schema。

#### 新增 Tool：list_enriched_views

```
Tool name: list_enriched_views
Description: List all pre-built enriched views with their descriptions and field summaries.
Arguments: (none)
Returns: Array of {view_name, description, base_table, row_count, key_fields}
```

这个 tool 让 Agent 在不确定该查哪张表时，先浏览所有可用的 view。

### 8.3 System Prompt 补充

在 Agent 的 system prompt 中增加一段指引：

```
## Enriched Views (预建宽表)

The data environment includes pre-built "enriched views" that already JOIN
common dimension tables (industry classification, stock archives, etc.).

Guidelines:
1. ALWAYS check if an enriched view exists for your target table before
   writing a JOIN query. Use list_enriched_views to browse available views.
2. Enriched views have dimension fields prefixed with the source table name,
   e.g., "lc_exgindustry_SecondIndustryName" for industry classification.
3. If an enriched view doesn't cover your needs (e.g., you need to JOIN
   two fact tables), you can still write custom JOINs using the original
   tables. Use get_table_relationships to find join paths.
```

---

## 9  跨领域适配策略

### 9.1 四个领域的差异分析

| 领域 | 维度表特征 | 适配注意事项 |
|------|-----------|-------------|
| **A股 (Stock)** | 3 张小维度表（~500行），被引用 25-45 次，打分区分度极好 | 标准流程即可，无需特殊处理 |
| **公募基金 (Fund)** | 维度表更多（可能 5-8 张），引用模式分散 | 可能需要降低 DIMENSION_THRESHOLD（如 0.40），或增加 max_dimension_tables 限制 |
| **宏观经济 (Macro)** | 时间序列为主，维度/事实边界模糊，打分区分度可能较差 | 如果打分结果没有明显的高分表，跳过 view 构建，回退到原始模式 |
| **医疗 (MIMIC)** | 维度表明确（patients, admissions），但存在链式 join | 可能需要支持 2 级 join：facts → dim_A → dim_B |

### 9.2 自适应降级机制

```python
def build_views_with_fallback(classification_result, relationships):
    """
    根据分类结果的质量，决定是否执行 view 构建。
    """
    dim_tables = classification_result.dimension_tables
    fact_tables = classification_result.fact_tables

    # 质量检查 1: 至少识别出 1 张维度表
    if len(dim_tables) == 0:
        return FallbackDecision.SKIP_VIEW_BUILD
        # 原因: 没有可靠的维度表，无法构建 view

    # 质量检查 2: 维度表的最高得分 >= 0.65
    max_score = max(s for _, s in dim_tables)
    if max_score < 0.65:
        return FallbackDecision.SKIP_VIEW_BUILD
        # 原因: 打分结果不够明确，维度表不可靠

    # 质量检查 3: 维度表行数不能太大
    for table, score in dim_tables:
        if get_row_count(table) > 50000:
            dim_tables.remove((table, score))
            # 原因: 行数太大的"维度表" join 后会导致性能问题

    # 质量检查 4: 至少存在 1 条 fact→dimension 的 relationship
    has_valid_join = any(
        get_source_table(r) in [t for t, _ in fact_tables]
        and get_target_table(r) in [t for t, _ in dim_tables]
        for r in relationships
    )
    if not has_valid_join:
        return FallbackDecision.SKIP_VIEW_BUILD

    return FallbackDecision.BUILD_VIEWS(dim_tables, fact_tables)
```

### 9.3 链式 Join 支持（医疗领域）

医疗领域的 MIMIC 数据可能存在链式 join 需求。例如：

```
diagnoses_icd (事实表)
  → admissions (维度表/中间表) — 通过 hadm_id
    → patients (二级维度表) — 通过 subject_id
```

这种情况下，如果 admissions 被分类为维度表，它自身也可以 join patients 的字段：

```python
def build_chained_views(dim_tables, relationships):
    """
    检测维度表之间是否存在 join 关系，如果有则构建链式 enriched view。
    """
    dim_names = [t for t, _ in dim_tables]
    dim_to_dim_rels = [
        r for r in relationships
        if get_source_table(r) in dim_names
           and get_target_table(r) in dim_names
    ]

    # 如果维度表之间存在关系，构建维度表自身的 enriched view
    # （在事实表 join 维度表时，使用 enriched 版本的维度表）
    chained_dims = {}
    for rel in dim_to_dim_rels:
        src = get_source_table(rel)
        tgt = get_target_table(rel)
        key = rel['source']['fields'][0]
        chained_dims[src] = (tgt, key)

    return chained_dims
```

这个功能作为可选扩展，在检测到维度表间的 relationship 时自动启用。

---

## 10  完整执行流程

将所有模块串联起来，一个 task 的完整处理流程如下：

```
Step 1: Schema 分析（已有）
  ├─ 扫描 CSV / JSON / SQLite
  ├─ 提取每张表的字段、类型、统计量
  └─ 输出: schemas[]

Step 2: Relationship 推断（已有）
  ├─ 分析表间的连接键和基数
  └─ 输出: relationships[]

Step 3: 维度表/事实表分类（新增 — TableClassifier）
  ├─ 对每张表计算 dimension_score
  ├─ 按阈值分类
  ├─ 输出: dimension_tables[], fact_tables[]
  └─ 耗时: < 0.1 秒

Step 4: 数据统一加载（新增 — ViewBuilder 前半部分）
  ├─ CSV → SQLite 内存表
  ├─ JSON → SQLite 内存表
  ├─ SQLite 原始表 → 复制到内存
  ├─ 维度表去重（如有 1:N 问题）
  └─ 耗时: 5-15 秒

Step 5: View 生成与校验（新增 — ViewBuilder 后半部分）
  ├─ 对每张事实表，生成 enriched view SQL
  ├─ 执行 CREATE VIEW
  ├─ 校验行数膨胀率和字段填充率
  ├─ 如有问题则修复或跳过
  └─ 耗时: 1-3 秒

Step 6: 精简 Context 生成（新增 — ContextPruner）
  ├─ 生成 view_summary（Level 0）
  ├─ 生成 standalone_tables 列表
  ├─ 注入 Agent system prompt
  └─ 耗时: < 0.1 秒

Step 7: Agent 运行（已有，Tool 层适配）
  ├─ Agent 查看 Level 0 context
  ├─ 按需调用 tool 获取 Level 1/2 信息
  ├─ 直接使用 enriched view 查询
  └─ 输出答案

新增总耗时: 约 6-18 秒（主要消耗在数据加载上）
```

---

## 11  效果预估与验证方案

### 11.1 task_1 的效果预估

**优化前的 Agent 执行链路（26 步 / 92.5 秒）**：

```
Step 1-5:  数据探索 + 视频截图理解        (5 步, ~12 秒)
Step 6-11: 逐个查看表结构                 (6 步, ~15 秒)
Step 12-13: 构造 JOIN SQL 并执行          (2 步, ~5 秒)
Step 14-26: 反复验证 + repair 循环         (13 步, ~60 秒)
```

**优化后的预期链路（8-12 步 / 30-45 秒）**：

```
Step 1-2:  视频截图理解                    (2 步, ~6 秒)
Step 3:    浏览 view_summary，定位目标 view (1 步, ~2 秒)
Step 4:    (可选) get_table_profile 查看详细 schema (0-1 步, ~3 秒)
Step 5-6:  直接写单表查询并执行             (2 步, ~5 秒)
Step 7-8:  简单验证                         (2 步, ~5 秒)
```

### 11.2 全局验证方案

建议在 60 个 task 上按以下步骤验证：

1. **A/B 测试**：对每个 task，分别跑"有 view"和"无 view"两个版本，对比步骤数、耗时、token 消耗、答案准确率。
2. **分领域统计**：按 A股、基金、宏观经济、医疗四个领域分别统计改善幅度，观察哪些领域受益最大。
3. **失败案例分析**：对于"有 view"反而变差的 task（如果存在），分析原因——可能是 view 的字段命名让 Agent 困惑，或者该 task 根本不需要 join。
4. **降级机制验证**：确认在维度表打分不明确的 task 上（如宏观经济），系统能正确跳过 view 构建而非产出低质量的 view。

---

## 12  边界情况与风险

### 12.1 已识别的边界情况

| 场景 | 风险 | 应对策略 |
|------|------|---------|
| 维度表连接键 1:N | view 行数膨胀 | 去重后重建 view，校验不通过则跳过该维度表 |
| 字段名冲突 | 事实表和维度表有同名字段 | 维度表字段统一加前缀 `{dim_table}_{field}` |
| JSON 嵌套结构 | json_normalize 可能产生极宽的表 | 限制展开深度（max_level=2），过深的嵌套不纳入 view |
| 维度表过大（>50,000 行） | JOIN 后内存和性能问题 | 不构建该维度表的 view，保留为 fallback |
| 跨事实表 JOIN | 预建 view 无法覆盖 | Agent 仍可查原始表 + 使用 relationships 信息 |
| 打分边界模糊 | 维度/事实无法清晰区分 | 自适应降级：跳过 view 构建，回退到原始模式 |

### 12.2 数据量增长后的演进

当前数据量在几千行级别，SQLite 内存 view 的性能完全足够。如果未来数据量增长到百万行级别，需要考虑：

1. 将 `CREATE VIEW` 改为 `CREATE TABLE ... AS SELECT ...`（物化 view），避免每次查询时重新计算 join。
2. 在连接键上建立索引，加速 join 计算。
3. 对大事实表考虑采样策略：只对有代表性的子集建 view，Agent 先探索 view 了解结构，再去原始表做精确查询。

---

## 13  工程化补充

### 13.1 错误处理与容错

每个模块都需要明确的异常处理策略，确保 view 构建失败时不会阻断 Agent 运行：

| 模块 | 可能的异常 | 处理策略 |
|------|-----------|---------|
| **TableClassifier** | relationships 为空（无法打分） | 直接返回空维度表列表，跳过 view 构建 |
| **ViewBuilder** | CSV 编码异常 / JSON 格式错误 | try-except 包裹，单个文件加载失败不影响其他文件 |
| **ViewBuilder** | CREATE VIEW SQL 语法错误（特殊字符字段名） | catch 异常，跳过该 view，记录日志 |
| **ViewBuilder** | 行数膨胀校验失败 | 尝试去重修复；修复失败则跳过该维度表的 join |
| **ToolAdapter** | Agent 引用了不存在的 view 字段 | 返回错误提示并建议 Agent 使用 get_table_profile 获取正确的字段列表 |
| **全局** | 内存不足（大数据量） | 切换到文件模式 SQLite（`/tmp/enriched_{task_id}.db`） |

核心原则是 **"尽力而为，优雅降级"**——任何一个环节的失败都不应该阻止 Agent 启动。最坏情况下，系统回退到不使用 view 的原始模式。

### 13.2 日志与可观测性

为支持调优和问题排查，以下数据点应持久化记录：

```python
# TableClassifier 决策日志
{
    "task_id": "task_1",
    "classification_results": [
        {
            "table": "lc_exgindustry",
            "scores": {
                "norm_ref": 1.00,
                "norm_row": 0.94,
                "uniqueness": 1.00,
                "time_density_inv": 0.88,
                "numeric_ratio_inv": 0.78
            },
            "final_score": 0.954,
            "classified_as": "dimension"
        },
        ...
    ],
    "threshold": 0.50,
    "dimension_tables": ["lc_exgindustry", "lc_actualcontroller", "lc_stockarchives"],
    "fact_tables": [...]
}

# ViewBuilder 校验日志
{
    "task_id": "task_1",
    "views_created": ["v_lc_freefloat_enriched", "v_lc_dividend_enriched", ...],
    "views_skipped": [],
    "validation_results": [
        {
            "view": "v_lc_freefloat_enriched",
            "fact_rows": 3119,
            "view_rows": 3119,
            "inflation_ratio": 1.00,
            "dim_fill_rates": {
                "lc_exgindustry_SecondIndustryName": 0.97,
                "lc_stockarchives_State": 0.95,
                "lc_actualcontroller_ControllerName": 0.89
            },
            "status": "pass"
        }
    ]
}

# Agent 使用行为日志（用于评估优化效果）
{
    "task_id": "task_1",
    "total_queries": 5,
    "queries_using_view": 4,
    "queries_using_raw_table": 1,
    "view_usage_rate": 0.80
}
```

### 13.3 Feature Flag 与回滚

建议在系统层面增加一个 feature flag 来控制 view 优化的开关：

```python
ENRICHED_VIEW_ENABLED = True  # 全局开关

# 在 pipeline 入口处判断
if ENRICHED_VIEW_ENABLED:
    classification = TableClassifier.run(semantic_catalog)
    if classification.is_viable():
        views = ViewBuilder.build(classification)
        context = ContextPruner.prune(semantic_catalog, views)
    else:
        context = original_context(semantic_catalog)  # 回退
else:
    context = original_context(semantic_catalog)  # 完全跳过
```

这样在 A/B 测试期间可以快速切换，也可以在发现问题时立即回滚而不需要代码变更。

### 13.4 类型兼容性处理

CSV、JSON、SQLite 三种来源的数据合并到同一个引擎时，连接键的类型可能不一致（如 CSV 中 CompanyCode 被推断为 TEXT "301027"，SQLite 中为 INTEGER 301027）。`LEFT JOIN ON f."CompanyCode" = d0."CompanyCode"` 可能因类型不匹配而连接失败。

处理方式：在数据加载阶段，对所有连接键字段统一做类型转换：

```python
def normalize_join_key_types(conn, table_name, join_keys, target_type='INTEGER'):
    """
    将连接键字段统一转换为指定类型，确保跨表 JOIN 时类型匹配。
    """
    for key in join_keys:
        try:
            conn.execute(f"""
                UPDATE "{table_name}"
                SET "{key}" = CAST("{key}" AS {target_type})
                WHERE "{key}" IS NOT NULL
            """)
        except Exception:
            # 如果 CAST 失败（如字段包含非数字字符），保持原样
            pass
```

### 13.5 维度表历史快照的局限性

第 6.6 节的去重策略（保留最新记录）会丢失维度表中的历史变更信息。例如，如果 lc_actualcontroller 记录了某公司在不同年份的实控人变更，去重后只保留最新的实控人。如果题目问的是"2019 年的实控人是谁"，Agent 使用 enriched view 会得到错误答案。

为缓解此问题，在 Level 0 context 的 view 描述中标注局限性：

```yaml
- name: v_lc_freefloat_enriched
  description: "流通股变动数据（已含行业分类、股票档案、实控人快照）"
  dimension_note: "维度信息为最新快照。如需历史维度数据（如某年份的实控人），请查询原始维度表。"
```
