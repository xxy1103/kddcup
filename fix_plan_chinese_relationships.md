## semantic_catalog.py 中文字段名关系推断修复方案

### 问题概述

`semantic_catalog.py` 中的 `infer_schema_relationships` 函数通过三层启发式过滤器来发现隐式外键关系：source key 检测 → target key 检测 → 名称评分。这三层过滤器全部依赖 `_field_tokens` 函数将字段名拆分为英文 token 集合，再与 `_ID_TOKENS`（id/key/code）等英文关键词做交集运算。

当数据集的字段名、表名、文件名全部为中文时，`_field_tokens` 使用的正则 `[^A-Za-z0-9]+` 将所有中文字符视为分隔符丢弃，导致 token 集合为空集。三层过滤器全部失效，`relationships` 始终返回空数组。

### 修复策略

修复分为 5 个变更点，全部位于 `src/data_agent_baseline/inspectors/semantic_catalog.py`。核心思路是：

1. 让 `_field_tokens` 保留 CJK 字符段作为完整 token
2. 新增 CJK 关键词集合
3. 用子串匹配（而非精确匹配）检测 CJK token 中是否包含关键词
4. 在名称评分中增加"同名字段"快捷路径

这些变更对纯英文数据集完全向后兼容——ASCII token 仍然使用精确匹配逻辑，CJK 子串匹配仅在 token 本身是 CJK 字符串时才生效。

---

### 变更 1：修复 `_field_tokens` 以保留 CJK 字符段

**位置**：第 123-125 行

**当前代码**：

```python
def _field_tokens(name: str) -> set[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return {token for token in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if token}
```

**修改为**：

```python
_CJK_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff"
    r"\U00020000-\U0002a6df\U0002a700-\U0002b73f]+",
    re.UNICODE,
)


def _field_tokens(name: str) -> set[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    # Split by non-alphanumeric (ASCII) characters, preserving CJK segments
    parts = _CJK_RE.split(" " + spaced + " ")
    tokens: set[str] = set()
    for part in parts:
        for token in re.split(r"[^A-Za-z0-9]+", part.lower()):
            if token:
                tokens.add(token)
    # Also add full CJK segments as individual tokens
    for match in _CJK_RE.finditer(spaced):
        tokens.add(match.group())
    return tokens
```

**说明**：先用 `_CJK_RE.split` 把 CJK 段从字符串中剥离出来，对剩余部分按原逻辑做 ASCII token 拆分。然后通过 `_CJK_RE.finditer` 将每个 CJK 连续字符段作为一个完整 token 加入集合。

**效果示例**：

| 字段名 | 修改前 tokens | 修改后 tokens |
|--------|-------------|-------------|
| `公司代码` | `set()` | `{'公司代码'}` |
| `所属基金/股票代码` | `set()` | `{'所属基金', '股票代码'}` |
| `A股证券代码` | `{'a'}` | `{'a', '股证券代码'}` |
| `user_id` | `{'user', 'id'}` | `{'user', 'id'}`（不变） |
| `companyCode` | `{'company', 'code'}` | `{'company', 'code'}`（不变） |

---

### 变更 2：新增 CJK 关键词集合与子串匹配辅助函数

**位置**：第 32-75 行（`_ROLE_TOKENS`、`_ID_TOKENS`、`_METRIC_TOKENS` 定义处）

**当前代码**：

```python
_ROLE_TOKENS = {"owner", "author", ...}
_ID_TOKENS = {"id", "key", "code"}
_METRIC_TOKENS = {"age", "amount", ...}
```

**修改为**：

```python
_ROLE_TOKENS = {
    "owner", "author", "creator", "created", "editor",
    "last", "parent", "child", "source", "target", "from", "to", "by",
}
_ID_TOKENS = {"id", "key", "code", "代码", "编号"}
_METRIC_TOKENS = {
    # English metric tokens (unchanged)
    "age", "amount", "average", "avg", "body", "count", "date",
    "description", "downvotes", "favorite", "favorites", "name",
    "number", "price", "quantity", "score", "sum", "text", "time",
    "timestamp", "title", "total", "upvotes", "value", "view", "views",
    # Chinese metric tokens
    "日期", "时间", "名称", "缩写", "比例", "金额", "股数", "总数",
}
```

**新增辅助函数**（放在 `_field_tokens` 之后）：

```python
def _is_cjk_token(token: str) -> bool:
    """Return True if the token consists entirely of CJK characters."""
    return bool(_CJK_RE.fullmatch(token))


def _has_token_match(tokens: set[str], keyword: str) -> bool:
    """Check if any token matches the keyword.

    For ASCII tokens: exact match only.
    For CJK tokens: substring match (e.g., '代码' in '公司代码').
    """
    for token in tokens:
        if token == keyword:
            return True
        if _is_cjk_token(token) and keyword in token:
            return True
    return False


def _tokens_intersect(tokens: set[str], keyword_set: set[str]) -> bool:
    """Check if any keyword from keyword_set matches any token."""
    return any(_has_token_match(tokens, kw) for kw in keyword_set)
```

**说明**：CJK 字段名是复合词（如 `公司代码` = `公司` + `代码`），无法像英文那样通过空格/符号拆分为独立 token。因此对 CJK token 使用子串匹配：只要 CJK token 中包含关键词（如 `代码`），就视为匹配。ASCII token 仍然使用精确匹配，确保向后兼容。

**关键设计决策**：

- `_ID_TOKENS` 选择 `代码` 和 `编号` 而不是 `号`：`号` 太短太泛，会匹配到 `序号`（行号）等无关字段
- `_METRIC_TOKENS` 中的中文词选择经过验证：`名称`/`缩写`/`日期`/`时间`/`比例`/`金额`/`股数`/`总数` 都是度量或描述性概念，不会与标识符字段混淆
- 子串匹配的方向是 `keyword in token`（关键词是否是 token 的子串），而非 `token in keyword`。这确保 `公司代码` 能匹配 `代码`，但 `代码` 不会匹配到更短的无关关键词

---

### 变更 3：将 `_looks_like_source_key` 和 `_looks_like_target_key` 切换为子串匹配

**位置**：第 595-616 行

**`_looks_like_source_key` 当前代码**：

```python
def _looks_like_source_key(ref: FieldRef) -> bool:
    tokens = _field_name_tokens(ref)
    if not (tokens & _ID_TOKENS):
        return False
    if tokens <= {"id"}:
        return False
    if tokens & _METRIC_TOKENS:
        return False
    return True
```

**修改为**：

```python
def _looks_like_source_key(ref: FieldRef) -> bool:
    tokens = _field_name_tokens(ref)
    if not _tokens_intersect(tokens, _ID_TOKENS):
        return False
    if tokens <= {"id"}:
        return False
    if _tokens_intersect(tokens, _METRIC_TOKENS):
        return False
    return True
```

**`_looks_like_target_key` 当前代码**：

```python
def _looks_like_target_key(ref: FieldRef) -> bool:
    tokens = _field_name_tokens(ref)
    if ref.is_primary_key:
        return True
    if tokens in ({"id"}, {"key"}, {"code"}):
        return True
    if tokens & _METRIC_TOKENS:
        return False
    if tokens & _ID_TOKENS and ref.cardinality is not None and ref.row_count:
        return ref.cardinality / max(ref.row_count, 1) >= 0.80
    return False
```

**修改为**：

```python
def _looks_like_target_key(ref: FieldRef) -> bool:
    tokens = _field_name_tokens(ref)
    if ref.is_primary_key:
        return True
    if tokens in ({"id"}, {"key"}, {"code"}, {"代码"}, {"编号"}):
        return True
    if _tokens_intersect(tokens, _METRIC_TOKENS):
        return False
    if _tokens_intersect(tokens, _ID_TOKENS) and ref.cardinality is not None and ref.row_count:
        return ref.cardinality / max(ref.row_count, 1) >= 0.80
    return False
```

**变更说明**：

- 将 `tokens & _ID_TOKENS`（精确交集）替换为 `_tokens_intersect(tokens, _ID_TOKENS)`（子串匹配）
- 将 `tokens & _METRIC_TOKENS` 替换为 `_tokens_intersect(tokens, _METRIC_TOKENS)`
- `_looks_like_target_key` 的精确匹配集合中增加 `{"代码"}, {"编号"}`（处理字段名恰好就是一个 CJK ID 词的情况）

**验证结果（以 task_4 字段为例）**：

| 字段名 | source_key | target_key | 理由 |
|--------|-----------|-----------|------|
| `公司代码` | True | True | 含 `代码`，无 metric 词 |
| `所属基金/股票代码` | True | True | 含 `代码` |
| `A股证券代码` | True | True | 含 `代码` |
| `中文名称缩写` | False | False | 含 `名称`+`缩写`（metric） |
| `公司中文名称` | False | False | 含 `名称`（metric） |
| `序号` | False | True(PK) | 不含 ID 词，但 is_primary_key=True |
| `截止日期` | False | False | 含 `日期`（metric） |
| `送股比例(10送X)` | False | False | 含 `比例`（metric） |
| `股东名称` | False | False | 含 `名称`（metric） |

---

### 变更 4：修复 `_relationship_name_score` 和 `_relationship_type`

**位置**：第 619-645 行

**`_relationship_name_score` 当前代码（关键段）**：

```python
def _relationship_name_score(source: FieldRef, target: FieldRef) -> tuple[float, str | None]:
    source_tokens = _field_name_tokens(source)
    source_refs = _reference_tokens(source)
    target_entity = _entity_tokens(target)
    target_field = _field_name_tokens(target)

    if "parent" in source_tokens and source.asset_path == target.asset_path and source.table == target.table:
        if target_field <= {"id"} or target.is_primary_key:
            return 0.90, "parent self-reference field targets the same entity key"

    if source_refs and source_refs <= (target_entity | target_field):
        return 0.86, "source key tokens match target entity/key tokens"

    if source.field.lower() == target.field.lower() and _looks_like_target_key(target):
        return 0.78, "source and target key fields share the same name"

    return 0.0, None
```

**修改为**：

```python
def _relationship_name_score(source: FieldRef, target: FieldRef) -> tuple[float, str | None]:
    source_tokens = _field_name_tokens(source)
    source_refs = _reference_tokens(source)
    target_entity = _entity_tokens(target)
    target_field = _field_name_tokens(target)

    if "parent" in source_tokens and source.asset_path == target.asset_path and source.table == target.table:
        if target_field <= {"id"} or target.is_primary_key:
            return 0.90, "parent self-reference field targets the same entity key"

    if source_refs and source_refs <= (target_entity | target_field):
        return 0.86, "source key tokens match target entity/key tokens"

    if source.field.lower() == target.field.lower() and _looks_like_target_key(target):
        return 0.78, "source and target key fields share the same name"

    # CJK same-name fallback: exact field name match across different assets
    if (
        source.field == target.field
        and source.field  # non-empty
        and source.asset_path != target.asset_path
        and _looks_like_target_key(target)
    ):
        return 0.75, "CJK same-name field across different assets"

    return 0.0, None
```

**说明**：原有的第 632 行同名匹配 `source.field.lower() == target.field.lower() and _looks_like_target_key(target)` 在变更 3 之后已经能处理中文字段（因为 `_looks_like_target_key` 现在对中文也能返回 True）。但这里额外增加一条 CJK 专用的跨资产同名匹配路径，作为兜底保障，分数略低（0.75 vs 0.78）以区分优先级。

**`_relationship_type` 修改**（第 638-645 行）：

```python
def _relationship_type(source: FieldRef, target: FieldRef) -> str:
    if source.asset_path == target.asset_path and source.table == target.table:
        return "self_reference"
    tokens = _field_name_tokens(source) | _field_name_tokens(target)
    if "code" in tokens or _tokens_intersect(tokens, {"代码"}):
        return "lookup_code"
    if source.field.lower() == target.field.lower():
        return "same_key"
    return "foreign_key"
```

**说明**：原来硬编码检查 `"code" in tokens`，对中文字段 `公司代码` 无法匹配。修改后使用 `_tokens_intersect` 同时检查 `code` 和 `代码`。这确保中文数据集的关系能被正确标记为 `lookup_code` 类型，下游 `build_derived_views` 才能识别。

---

### 变更 5：修复 `_reference_tokens` 中的 CJK 过滤

**位置**：第 569-571 行

**当前代码**：

```python
def _reference_tokens(ref: FieldRef) -> set[str]:
    tokens = _field_name_tokens(ref)
    return {token for token in tokens if token not in _ID_TOKENS and token not in _ROLE_TOKENS}
```

**修改为**：

```python
def _reference_tokens(ref: FieldRef) -> set[str]:
    tokens = _field_name_tokens(ref)
    return {
        token for token in tokens
        if not _has_token_match(tokens, token)  # keep token if it's not a keyword
        or not (_tokens_intersect({token}, _ID_TOKENS) or _tokens_intersect({token}, _ROLE_TOKENS))
    }
```

更简洁的等价写法：

```python
def _reference_tokens(ref: FieldRef) -> set[str]:
    tokens = _field_name_tokens(ref)
    result = set()
    for token in tokens:
        if _tokens_intersect({token}, _ID_TOKENS) or _tokens_intersect({token}, _ROLE_TOKENS):
            continue
        result.add(token)
    return result
```

**说明**：原来的 `token not in _ID_TOKENS` 是精确匹配，CJK token `公司代码` 不在 `_ID_TOKENS` 中所以会被保留。修改后使用子串匹配，`公司代码` 包含 `代码`（属于 `_ID_TOKENS`），因此会被正确过滤掉。这确保 `_reference_tokens` 对 CJK 字段只保留真正的"实体描述"部分（如 `公司`），而不是保留 ID 类 token。

---

### 变更总览

| 变更 | 函数 | 改动类型 | 影响范围 |
|------|------|---------|---------|
| 1 | `_field_tokens` | 正则修改 | 全局基础，所有下游函数 |
| 2 | `_ID_TOKENS` / `_METRIC_TOKENS` + 新增辅助函数 | 关键词扩展 | `_looks_like_*`, `_reference_tokens` |
| 3 | `_looks_like_source_key` / `_looks_like_target_key` | 匹配逻辑替换 | `infer_schema_relationships` 主循环 |
| 4 | `_relationship_name_score` / `_relationship_type` | 增加 CJK 兜底 + 修复 lookup_code | 关系评分和分类 |
| 5 | `_reference_tokens` | 过滤逻辑适配 | 名称评分的 token 比较 |

---

### 对 task_4 的预期修复效果

修复后，`infer_schema_relationships` 将能检测到以下关系：

**通过 `公司代码` 字段**：所有包含此字段的表（6 CSV + 12 SQLite 表 + 6 JSON 文件 = 24 张表）之间都能建立 N×(N-1) 对多关系。每对关系都会进入 `_validate_relationship` 做值域匹配验证，只保留匹配率 ≥ 95% 的高置信度关系。

**通过 `所属基金/股票代码` 字段**：同理，跨表同名字段匹配。

**最终输出数量估算**：由于值域匹配会过滤掉值域不重叠的表对（比如 SQLite 中 `A股询价明细` 只有 3 家公司，其 `公司代码` 值域与其他表重叠很少），实际保留的关系数量会远少于理论上限。这是正确的行为。

---

### 测试计划

**新增单元测试**（在 `tests/test_data_inspector.py` 中）：

```python
def test_relationship_inference_detects_chinese_same_name_keys(tmp_path: Path) -> None:
    """Verify that fields with identical Chinese names are detected as relationships."""
    # Create two CSV files sharing the field '公司代码'
    companies = tmp_path / "companies.csv"
    companies.write_text(
        "公司代码,公司中文名称\n"
        "100,测试公司A\n200,测试公司B\n300,测试公司C\n",
        encoding="utf-8",
    )
    orders = tmp_path / "orders.csv"
    orders.write_text(
        "公司代码,订单金额\n"
        "100,5000\n200,3000\n100,2000\n",
        encoding="utf-8",
    )
    # Run build_semantic_catalog and verify relationships are found
    ...

def test_relationship_inference_rejects_chinese_metric_fields(tmp_path: Path) -> None:
    """Verify that fields like '截止日期' or '中文名称' are NOT detected as keys."""
    ...

def test_field_tokens_preserves_cjk_segments() -> None:
    """Unit test for _field_tokens with CJK input."""
    from data_agent_baseline.inspectors.semantic_catalog import _field_tokens
    assert "公司代码" in _field_tokens("公司代码")
    assert "股票代码" in _field_tokens("所属基金/股票代码")
    assert "user" in _field_tokens("user_id")  # backward compat
```

**回归测试**：确保现有的 4 个 relationship 测试仍然通过：

- `test_relationship_inference_validates_csv_key_matches_with_data`
- `test_relationship_inference_avoids_low_match_and_type_id_false_positive`
- `test_relationship_inference_matches_json_field_to_sqlite_key`
- `test_relationship_inference_reports_sqlite_explicit_foreign_key`

---

### 风险评估

**向后兼容风险：低**。ASCII token 的处理路径完全不变（精确匹配），CJK 子串匹配仅在 token 是 CJK 字符串时触发。英文数据集不会进入 CJK 分支。

**误报风险：中**。中文字段名的子串匹配可能引入一些边界情况。例如如果未来有字段叫 `反代码审查表`，其中包含 `代码` 子串，会被误判为 ID 字段。但在当前 A 股数据集的字段命名规范下，这种风险极低。可以通过增加 `_METRIC_TOKENS` 的覆盖范围来缓解。

**性能影响：可忽略**。`_tokens_intersect` 的子串匹配开销极小（CJK token 数量通常 ≤ 5 个，keyword 集合 ≤ 30 个），且值域验证（`_validate_relationship`）仍然是主要的性能瓶颈，未受影响。
