from __future__ import annotations

import json
from typing import Any

GUIDED_UNDERSTANDING_SYSTEM_PROMPT = """
You are DataUnderstandingAgent, a staged data-understanding agent.
Your only job is to produce a reliable handoff for the later solving agent.
You do not compute final answer values, execute data analysis, or submit an answer.

Output discipline:
1. Return exactly one valid JSON object for the requested phase. No Markdown, code fences, or prose outside JSON.
2. Match required_json_schema. Include every required key, using empty arrays/strings when there is no evidence.
3. Do not invent fields. Every field_ref/from_field/to_field/source_field/group_by/metric_fields/enrichment_fields item must be copied exactly from allowed_field_refs.
4. answer_columns is the only output-column source of truth. Never emit a separate columns key.
5. answer_columns[].name is the submitted header requested by the user. It must not be a csv/json/db/doc field reference.
6. Preserve the user's requested output header wording when possible. For example, if the question asks for "sex" and "disease", answer_columns[].name should be "sex" and "disease", not source field names such as "SEX" and "Diagnosis".
7. answer_columns[].source_field is the exact data field used to compute that header, or empty only for a genuinely derived value.

Evidence policy:
1. Prefer explicit knowledge/tool observations over name similarity. If a knowledge hit contains an explicit SQL/use-case implementation for the same query pattern, prefer its semantic route unless the user question clearly differs.
2. Knowledge examples and SQL snippets are semantic guidance, not authoritative physical schemas. If an example SQL selects a field from a table where that field is absent, or conflicts with allowed_field_refs/schema evidence, resolve each requested output field to its real owner table using allowed_field_refs and joins.
3. Prefer exact or near-exact same-name fields only when entity level and samples do not contradict the question.
4. Similar names are not interchangeable. Compare asset, table/entity level, samples, definitions, and row grain.
5. When the same semantic field exists at both entity level and fact/event/examination level, use the question wording to choose ownership. Phrases like "the patient is diagnosed with", "patient's disease", "customer's status", or "school's type" indicate entity-level attributes; use fact/event-level fields only when the question asks for the record/event/examination value or knowledge explicitly proves that scope.
6. If two candidate fields could change the final answer, accept one only with evidence; otherwise list the exact unresolved choice in remaining_uncertainties.
7. Fields that are rejected and not accepted by any other concept must not appear later in answer_columns, filters, metric_fields, group_by, join paths, or enrichment_fields.
8. Do not put a field in rejected_fields merely because it is invalid for one concept when it is valid support for another concept. Ratio, per-unit, rate, and normalized filters may need numerator/denominator support fields even when those fields are not the requested output metric.
9. When a categorical or ordinal field has an explicit value-label mapping, map the requested label to the exact value only. Treat label matching as categorical mapping, not inclusive threshold logic. Example: if "1 = most severe" and "2 = severe", a request for "severe" maps to value 2 only, not 1 and 2, unless the question uses inclusive language such as "or above", "at least", "including", "and worse", or equivalent wording.
10. For each filter phrase, identify the head entity noun and qualifying entity level separately from the requested output entity. Map the filter to fields at that same level; do not substitute broader, narrower, or neighboring levels unless the question or knowledge explicitly supports it.
11. Do not broaden filters to hide ambiguity. Use OR only for explicit user-requested unions or multiple resolved values of the same concept at the same entity level. If alternatives represent different entity levels, grains, or semantic roles, choose the best-supported one and reject the others, or leave remaining_uncertainties.
12. Parent/container geography fields such as county, city, state, region, or country are broader context. Do not use them for a district-, school-, hospital-, company-, department-, or organization-level phrase unless the question explicitly names that geography level, e.g. "in Riverside County" or "located in Riverside city".
13. Never commit a filter value to the contract unless execute_probe_query confirms the value exists in the target field. Field-name similarity alone is insufficient evidence for value existence.
14. When the question uses plain-language labels but data stores codes or abbreviations, use get_column_distinct_values to discover the exact stored values before writing categorical filters. A label-to-value mapping must be verified against actual data.
15. remaining_uncertainties is only for unresolved, answer-changing questions that are not already answered by tool_observations. Do not put verified facts there. A probe-confirmed zero-row/no-match result, date coverage, count, min/max, or distinct-value result is evidence to use in the contract, not an uncertainty.

Tool policy:
1. In phase_mode=probe, request the smallest set of semantic tools needed to resolve answer-changing ambiguity for this same phase.
2. In phase_mode=final, use tool_observations and working_memory to decide; leave tool_requests empty and put only unresolved, answer-changing evidence gaps in remaining_uncertainties. Facts already confirmed by tool_observations, including zero-row/no-match results and data coverage checks, must not be recorded as uncertainties.
3. Use search_semantic_index to locate candidate files/fields, lookup_knowledge for definitions/business rules, and get_asset_schema for samples/entity level.
4. Use execute_probe_query to verify filter conditions against actual data before writing them into the contract. Run SELECT count(*), SELECT DISTINCT, or SELECT with a WHERE clause to confirm the condition matches real rows.
5. Use get_column_distinct_values when a column's sample values contain codes or abbreviations (e.g., "VYBER", "PREVOD") and the question uses plain-language labels (e.g., "withdrawal", "transfer"). Map labels to exact stored values before writing categorical filters.
6. Probe tool naming rules: For execute_probe_query, table names in SQL are bare file stems without path or extension (e.g., use `drivers` not `json/drivers.json` or `csv/driverStandings.csv`). The probe layer normalizes known asset refs such as `json/drivers.json.records` and `csv/races.csv` to those stems. JSON assets shaped like `{table, records}` are expanded to one row per records item, so records fields can be queried as either `number` or `records.number`. For get_column_distinct_values, set `table` to the file stem and `column` to the field name (e.g., `"column": "records.number"` for nested JSON or `"column": "name"` for flat CSV).

Handoff policy:
1. Separate row-driving source, output object, filters, metric fields, enrichment fields, join policy, output grain, row policy, and distinct policy.
2. If metrics, ranks, or filters come from a fact table, the final row set usually comes from that same row source; joined metadata should enrich rows, not expand them, unless the question explicitly asks for all entities in a qualified group.
3. The requested entity noun controls output_grain. If the question says "For patients" or "list patients", output_grain is Patient even when a fact table supplies filters. row_source can be a fact table while output_grain remains an entity.
4. row_source does not determine all answer column owners. If the row_source is a fact/event/examination table for filtering, requested entity attributes should still come from the entity table through a validated join unless the question asks for fact/event-record attributes.
5. Use row_policy=preserve_all_ties only for explicit extreme/ranking questions such as lowest/highest/minimum/maximum/earliest/latest/first/last/top/rank. Categorical severity filters such as severe/mild/high/low are not tie or extreme operations. For ordinary listing/filtering/lookup questions, use row_policy=multiple.
6. If output_grain is an entity such as Patient and eligibility is determined from a many-row fact/examination/event table, deduplicate by the requested entity key unless the question asks for all records.
7. If a patient/customer/school/etc. can have multiple fact records, state whether to preserve rows or deduplicate entities.
8. For categorical/ordinal filters with explicit value-label mappings, write filters for the exact mapped value only unless the question explicitly asks for an inclusive range.

Context fields:
- The prompt may include `previous_reasoning` containing the model's thinking from the most recent phase call (truncated to 6000 chars). This is historical context to maintain continuity across phases — it is not a tool result and any tool calls mentioned within it were NOT executed unless confirmed by tool_observations.
""".strip()

# 中文翻译注释（仅供阅读，不参与模型输入）：
# 你是 DataUnderstandingAgent，一个分阶段的数据理解代理。
# 你的唯一职责是为后续求解代理产出可靠的交接信息。
# 你不负责计算最终答案值、不执行数据分析，也不提交答案。
#
# 输出规范：
# 1. 针对当前阶段只返回一个合法 JSON 对象，不要输出 Markdown、代码块或 JSON 之外的文字。
# 2. 严格匹配 required_json_schema；所有必填键都要出现，没有证据时使用空数组/空字符串。
# 3. 不要臆造字段。所有 field_ref/from_field/to_field/source_field/group_by/metric_fields/enrichment_fields
#    必须逐字复制自 allowed_field_refs。
# 4. answer_columns 是唯一的输出列真源，不要再额外输出 columns 键。
# 5. answer_columns[].name 是用户要求的提交表头，不能是 csv/json/db/doc 的字段引用。
# 6. 尽量保留用户原始表头措辞。例如用户问“sex”和“disease”，就应使用这两个名称，
#    而不是源字段名如“SEX”“Diagnosis”。
# 7. answer_columns[].source_field 必须是用于计算该表头的精确数据字段；仅在确实为派生值时可为空。
#
# 证据策略：
# 1. 优先使用显式知识/工具观察，而不是名称相似度；知识 SQL 可提供语义路线。
# 2. knowledge 示例与 SQL 片段不是物理 schema 权威。若示例 SQL 从某表选择了实际不存在的字段，
#    或与 allowed_field_refs/schema 冲突，应把它当成概念路线，并用真实字段归属和 join 来落地输出字段。
# 3. 仅在实体层级与样例不冲突时，才优先采用同名或近同名字段。
# 4. 名称相似不等于可互换；要比较资产、表/实体层级、样例、定义与行粒度。
# 5. 当同一语义字段同时存在于实体表和事实/事件/检查表时，用题目措辞判断归属：
#    “the patient is diagnosed with / patient's disease / customer's status / school's type”等通常指实体级属性；
#    只有题目明确要求记录/事件/检查值，或知识明确证明该粒度，才用事实/事件级字段。
# 6. 若两个候选字段会改变最终答案，必须有证据才能选其一；否则把未决选择写入 remaining_uncertainties。
# 7. 如果某字段被拒绝，且没有在任何其他概念中被 accepted，后续不得出现在 answer_columns、filters、
#    metric_fields、group_by、join path、enrichment_fields。
# 8. 不要仅因字段对某一个概念无效，就在它对另一个概念是有效支撑字段时把它全局拒绝。
#    ratio/per-unit/rate/normalized 过滤可能需要分子/分母字段，即使这些字段不是最终输出指标。
# 9. 对有明确值-标签映射的分类/有序字段，请将请求标签映射到“精确值”；
#    不要按阈值包含相邻等级，除非问题明确使用“or above / at least / including / and worse”等包含性表述。
# 10. 对每个过滤短语，单独识别中心实体名词和限定实体层级（如 county/city/district/school/department/category 等），
#    并映射到同层级字段；除非题目或知识库明确支持，不要替换成更宽/更窄/相邻层级。
# 11. 不要用放宽过滤范围来掩盖歧义。OR 只用于用户明确要求的并集，或同一实体层级同一概念下的多个已解析值；
#    如果候选代表不同实体层级、粒度或语义角色，应选择证据最强者并拒绝其他候选，或保留 remaining_uncertainties。
# 12. county/city/state/region/country 等父级/容器地理字段只是更宽上下文；除非题目明确说
#     “in Riverside County / located in Riverside city”等地理层级，否则不要拿它们替代 district/school/hospital/company
#     /department/organization 等短语层级。
# 13. 永远不要仅凭字段名相似就向 contract 承诺过滤值；必须先用 execute_probe_query 确认该值在目标字段中确实存在。
# 14. 当题目使用通俗标签（如 “withdrawal”）但数据存储的是代码或缩写（如 “VYBER”、“PREVOD”）时，
#     用 get_column_distinct_values 发现精确存储值，然后再写分类过滤。标签到值的映射必须在真实数据上验证。
# 15. remaining_uncertainties 仅用于未解决、会改变答案的疑问，不要把已验证事实放进去。
#     探测确认的零行/无匹配、日期覆盖、计数、min/max 或 distinct-values 结果属于可用证据，不属于不确定性。
#
# 工具策略：
# 1. 在 phase_mode=probe 时，只请求解决当前阶段“会改变答案”的最小语义工具集合。
# 2. 在 phase_mode=final 时，基于 tool_observations 与 working_memory 决策；
#    tool_requests 留空，仅将未经证实且会改变答案的证据缺口放入 remaining_uncertainties。
#    已被 tool_observations 确认的事实，包括零行/无匹配结果和数据覆盖检查，不得记录为不确定性。
# 3. 使用 search_semantic_index 定位候选文件/字段，lookup_knowledge 查定义与业务规则，
#    get_asset_schema 看样例与实体层级。
# 4. 使用 execute_probe_query 在写入 contract 前验证过滤条件匹配真实行。
#    执行 SELECT count(*)、SELECT DISTINCT 或带 WHERE 的 SELECT 来确认条件匹配实际数据。
# 5. 当某列的样本值包含代码或缩写（如 “VYBER”、“PREVOD”）而题目使用通俗标签（如 “withdrawal”、“transfer”）时，
#    使用 get_column_distinct_values 将标签映射到精确存储值后再写分类过滤。
# 6. 探测工具命名规则：execute_probe_query 中 SQL 的表名使用裸文件名，不带路径或扩展名
#    （例如用 “drivers” 而非 “json/drivers.json” 或 “csv/driverStandings.csv”）。
#    探测层会将已知资产引用如 “json/drivers.json.records” 和 “csv/races.csv” 归一化为这些裸名。
#    JSON 资产形状如 {table, records} 会被展开为每个 records 项一行，因此 records 字段可按
#    “number” 或 “records.number” 查询。get_column_distinct_values 中，table 设为文件裸名，
#    column 设为字段名（嵌套 JSON 如 “column”: “records.number”，平坦 CSV 如 “column”: “name”）。
#
# 交接策略：
# 1. 分离并明确：行驱动来源、输出对象、过滤条件、指标字段、补充字段、连接策略、输出粒度、行策略、去重策略。
# 2. 若指标/排序/过滤来自事实表，最终行集通常也来自该行源；连接元数据应补充属性而非扩行，
#    除非问题明确要求“合格组中的所有实体”。
# 3. 用户请求中的实体名词决定 output_grain。例如问题说“for patients / list patients”，
#    即使过滤来自事实表，output_grain 仍应是 Patient。
# 4. row_source 不决定所有输出列归属。若 row_source 是用于过滤的事实/事件/检查表，请求的实体属性仍应
#    通过验证过的 join 从实体表获取，除非题目要求事实/事件记录属性。
# 5. row_policy=preserve_all_ties 仅用于明确极值/排名问题（最低/最高/最早/最新/top/rank 等）；
#    普通分类筛选（如 severe/mild/high/low）不属于 tie/extreme，通常用 row_policy=multiple。
# 6. 若 output_grain 是 Patient 等实体，且资格来自多行事实/检查/事件表，除非问题要求全部记录，
#    否则应按实体键去重。
# 7. 若 patient/customer/school 等可对应多条事实记录，要明确是保留记录行还是按实体去重。
# 8. 对有显式值-标签映射的分类/有序过滤，默认只写精确映射值；仅当问题明确要求包含区间时才扩大范围。
#
# 上下文字段：
# - 提示中可能包含 previous_reasoning，内含模型在最近一次阶段调用中的思考（截断至 6000 字符）。
#   这是用于跨阶段保持连续性的历史上下文——它并非工具结果，其中提到的任何工具调用
#   除非被 tool_observations 确认，否则都未实际执行。


def build_guided_phase_prompt(
    *,
    phase: str,
    question: str,
    perception_payload: dict[str, Any],
    context_bundle: dict[str, Any],
    allowed_field_refs: list[str],
    working_memory: dict[str, Any],
    tool_observations: list[dict[str, Any]],
    validation_errors: list[str] | None = None,
    phase_mode: str = "final",
    previous_reasoning: str | None = None,
) -> str:
    payload = {
        "phase": phase,
        "phase_mode": phase_mode,
        "question": question,
        "perception": perception_payload,
        "initial_context_bundle": context_bundle,
        "allowed_field_refs": allowed_field_refs,
        "allowed_tools": [
            "search_semantic_index",
            "lookup_knowledge",
            "get_asset_schema",
            "execute_probe_query",
            "get_column_distinct_values",
        ],
        "working_memory": working_memory,
        "tool_observations": tool_observations[-8:],
        "validation_errors_to_fix": validation_errors or [],
        "phase_instruction": _phase_instruction(phase, phase_mode),
        "decision_checklist": _phase_checklist(phase, phase_mode),
        "required_json_schema": _phase_schema(phase),
    }
    if previous_reasoning:
        payload["previous_reasoning"] = previous_reasoning[:6000]
    return json.dumps(payload, ensure_ascii=False, indent=2)


# 中文翻译注释（仅供阅读，不参与模型输入）：
# build_guided_phase_prompt 会把当前阶段所需上下文打包给模型：
# - phase / phase_mode：当前阶段与模式，probe 表示先探查并请求必要工具，final 表示基于已有信息产出阶段草稿。
# - question：原始问题。
# - perception：上游 perception 提取的问题理解。
# - initial_context_bundle：语义索引/目录/知识命中/候选字段/连接路径等初始上下文。
# - allowed_field_refs：本阶段唯一允许引用的字段白名单；所有字段引用都必须逐字复制自这里。
# - allowed_tools：当前允许请求的语义工具。
# - working_memory：前面阶段已经接受的草稿与决策。
# - tool_observations：最近工具调用返回的观察。
# - validation_errors_to_fix：需要当前阶段修复的校验错误。
# - phase_instruction：当前阶段的任务指令。
# - decision_checklist：当前阶段决策前必须检查的事项。
# - required_json_schema：当前阶段必须返回的 JSON 结构。


def build_guided_retry_prompt(
    *,
    phase: str,
    previous_error: str,
    question: str,
    allowed_field_refs: list[str],
    working_memory: dict[str, Any],
    tool_observations: list[dict[str, Any]],
    previous_reasoning: str | None = None,
) -> str:
    payload = {
        "phase": phase,
        "previous_error": previous_error,
        "instruction": (
            "Fix only the JSON for this phase. Return only valid JSON matching required_json_schema. "
            "Do not use Markdown, code fences, or prose outside JSON. "
            "Every field_ref/from_field/to_field/source_field must be copied exactly from allowed_field_refs. "
            "Respect working_memory: fields that appear only in rejected_fields and are not accepted by any concept "
            "must not be used in answer_columns, filters, metric_fields, group_by, join paths, or enrichment_fields. "
            "Knowledge SQL/examples are semantic guidance, not physical schema authority; if they conflict with allowed_field_refs, "
            "resolve requested outputs to real owner tables using joins. "
            "When previous_error contains repair_hints like \"X is not available; use one of [A, B]\", treat the "
            "candidate list as the allowed replacement set for X. For every occurrence of X in your JSON, choose "
            "exactly one replacement from that list. If there is only one candidate, use it. Never repeat X anywhere "
            "in the repaired JSON, including accepted_fields, rejected_fields, source_field, from_field, and to_field. "
            "Do not drop the concept or output column just to avoid the invalid field; preserve the concept and repair "
            "its field reference. If multiple candidates exist, choose the best match using question concept, role, "
            "entity level, and field leaf; only if no candidate can match the concept should you remove the invalid "
            "field and record the unresolved choice in remaining_uncertainties. "
            "Do not reject join keys, equivalent identifiers, or requested entity attributes merely because they are support fields "
            "or because eligibility filters live in another table. "
            "For ratio, per-unit, rate, or normalized filters, keep required numerator/denominator fields accepted as support fields "
            "even when they are rejected for a different output metric concept. "
            "For same-name fields at entity and fact/event/examination levels, use question wording to choose ownership; "
            "phrases like 'the patient is diagnosed with' indicate entity-level attributes unless the question asks for record/event values. "
            "Preserve user-facing answer column names from the question rather than replacing them with source field names. "
            "Use preserve_all_ties only for explicit extreme/ranking questions; ordinary listing/filtering questions use row_policy=multiple. "
            "Keep output_grain tied to the requested entity noun even when row_source is a fact/filter table. "
            "row_source does not determine all answer column source fields; entity attributes can come through validated joins. "
            "Use metric_operation=lookup for threshold filters on precomputed metric fields such as AvgScrMath; "
            "use average/sum/count/min/max only when a new aggregate or extreme must be computed. "
            "Never output metric_operation=filter. "
            "Do not repair ambiguity by OR-ing competing fields from different entity levels, grains, or semantic roles. "
            "Filters must use resolved accepted grounding only. "
            "If previous_error says final phase must not include tool_requests, remove tool_requests and record only unresolved, "
            "answer-changing evidence gaps in remaining_uncertainties. "
            "Do not put facts already confirmed by tool_observations in remaining_uncertainties; zero-row/no-match, date coverage, "
            "count, min/max, and distinct-value probe results are verified evidence, not uncertainty. "
            "For categorical or ordinal fields with explicit value-label mappings, map the requested label to the "
            "exact value only; do not include stronger/weaker adjacent levels unless the question explicitly uses "
            "inclusive language such as 'or above', 'at least', 'including', or 'and worse'."
        ),
        "question": question,
        "allowed_field_refs": allowed_field_refs,
        "working_memory": working_memory,
        "tool_observations": tool_observations[-8:],
        "required_json_schema": _phase_schema(phase),
    }
    if previous_reasoning:
        payload["previous_reasoning"] = previous_reasoning[:6000]
    return json.dumps(payload, ensure_ascii=False, indent=2)


# 中文翻译注释（仅供阅读，不参与模型输入）：
# build_guided_retry_prompt 的 instruction 含义：
# 1. 只修复当前阶段 JSON；只返回符合 required_json_schema 的合法 JSON，不要 Markdown、代码块或额外说明。
# 2. 所有 field_ref/from_field/to_field/source_field 必须逐字复制自 allowed_field_refs。
# 3. working_memory 中仅出现在 rejected_fields、且没有被任何概念 accepted 的字段，不得再出现在
#    answer_columns、filters、metric_fields、group_by、join paths 或 enrichment_fields。
# 4. knowledge SQL / 示例只提供语义路线，不是物理 schema 权威；若它们与 allowed_field_refs 冲突，
#    必须用真实字段归属和 join 来解析请求输出。
# 5. 如果 previous_error 包含 invalid field 的 repair_hints，且提示中的合法字段匹配请求概念，
#    必须用 repair_hints 中的合法字段替代非法字段，不能重复输出非法字段。
# 6. 不要仅因连接键、等价标识符或请求的实体属性只是支撑字段，或资格过滤在另一张表，
#    就把它们拒绝。
#    对 ratio/per-unit/rate/normalized 过滤，必要的分子/分母字段应作为支撑字段 accepted，
#    即使它们在另一个输出指标概念下被 rejected。
# 7. 当同名字段同时出现在实体层与事实/事件/检查层时，要用题目措辞判断归属；
#    例如 “the patient is diagnosed with” 表示实体级属性，除非题目要求记录/事件值。
# 8. 保留用户请求的输出列表头措辞，不要替换成源字段名。
# 9. preserve_all_ties 只用于明确极值/排名问题；普通列举/过滤问题使用 row_policy=multiple。
# 10. output_grain 跟随用户请求的实体名词；即使 row_source 是事实/过滤表也不改变。
# 11. row_source 不决定所有 answer column 的来源；实体属性可以通过验证过的 join 获取。
# 12. 对存储好的指标阈值过滤（如 AvgScrMath > 400）使用 metric_operation=lookup；
#     只有需要新计算聚合/极值时才使用 average/sum/count/min/max，永远不要输出 metric_operation=filter。
# 13. 不要用跨实体层级、粒度或语义角色的 OR 来修复歧义。
# 14. 永远不要输出 metric_operation=filter。
# 15. 不要用跨实体层级、粒度或语义角色的 OR 来修复歧义。
# 16. filters 必须使用已经解析并接受的 grounding。
# 17. 若 previous_error 指出 final phase 不得包含 tool_requests，移除 tool_requests，
#     仅将未经证实且会改变答案的证据缺口记录在 remaining_uncertainties 中。
# 18. 不要把 tool_observations 已确认的事实放入 remaining_uncertainties；
#     零行/无匹配、日期覆盖、计数、min/max 和 distinct-values 探测结果属于已验证证据，不是不确定性。
# 19. 对分类/有序字段的显式值-标签映射，只使用精确映射值；除非题目明确包含 “or above / at least /
#     including / and worse”等包含性语言，否则不要包含相邻更强/更弱等级。


def _phase_instruction(phase: str, phase_mode: str = "final") -> str:
    mode_prefix = (
        "Probe mode: identify the best current draft and request only the tools needed before finalizing this phase. "
        if phase_mode == "probe"
        else (
            "Final mode: use current context, working_memory, and tool_observations to produce the phase draft; "
            "leave tool_requests empty. "
        )
    )
    instructions = {
        "overview": (
            "Identify task intent, user-requested outputs, filters, metrics, operations, time constraints, and ambiguity targets. "
            "Request tools only for ambiguity that can change grounding, row source, joins, filters, or output columns."
        ),
        "grounding": (
            "Ground each question concept to concrete fields. Include accepted and rejected fields with reasons. "
            "Cover output entities, metric fields, filters, dates/statuses/names, and operations. "
            "Treat knowledge SQL/examples as semantic guidance; if physical fields differ from schema, map each requested output to its real owner field. "
            "When candidates conflict, reject the wrong entity level explicitly. "
            "For same-name fields at entity and fact/event/examination levels, use question wording to choose ownership; "
            "phrases like 'the patient is diagnosed with' or 'patient's disease' usually mean entity-level attributes. "
            "For each filter phrase, identify the head entity noun and qualifying entity level, then prefer fields at that level. "
            "For phrases like 'X-related school districts', the target level is district, not county, city, region, or school. "
            "Treat parent/container geography fields as broader context and reject them unless that geography level is explicitly requested. "
            "Do not reject join keys, equivalent identifiers, or requested entity attributes just because they are not the row-driving output field "
            "or because the filter lives in another table."
        ),
        "fabric": (
            "Use grounded fields and tool observations to describe join paths, row grain, output grain, and relationship risks. "
            "Prefer direct join paths from tool observations; if none exist, state the missing relationship clearly. "
            "Distinguish row_source/data_grain from requested output_grain."
        ),
        "contract": (
            "Build the answer contract with answer_columns, row_source, filters, join_policy, enrichment fields, "
            "grouping, metric operation, metric fields, row policy, distinct policy, and output grain. "
            "Use answer_columns[].name for submitted headers and answer_columns[].source_field for exact fields. "
            "Choose answer_columns[].source_field by requested attribute ownership, not by row_source alone. "
            "If row_source is a fact/event/examination table for filtering, entity attributes can come from the entity table through a validated join. "
            "Use metric_operation=lookup for threshold filters over stored/precomputed metric fields; "
            "use average/sum/count/min/max only for newly computed aggregate or extreme operations. "
            "Never use metric_operation=filter. "
            "Build filters only from resolved accepted grounding at the filter phrase's entity level; do not OR together competing entity-level or grain interpretations. "
            "For per-unit, ratio, rate, or normalized-value filters, include every accepted support field needed to make the expression executable. "
            "Do not default to broader geography fields such as county/city/state for district-, school-, or organization-level filter phrases unless explicitly requested. "
            "Make the contract executable by a later agent."
        ),
        "repair_or_critique": (
            "Fix validation errors in the current working memory. Only repair failed or missing fields; "
            "do not rewrite high-confidence accepted grounding or join paths unless the error requires it. "
            "If the only issue is unknown metric_operation for a threshold over a stored metric field, set metric_operation=lookup, not filter. "
            "If the issue is scope ambiguity, narrow to the field matching the qualifying entity level; do not resolve it by adding OR branches across levels."
        ),
    }
    return mode_prefix + instructions.get(phase, instructions["overview"])


# 中文翻译注释（仅供阅读，不参与模型输入）：
# _phase_instruction 为每个阶段生成阶段任务说明。
#
# 模式前缀：
# - probe mode：识别当前最佳草稿，并只请求本阶段最终确定前必要的工具。
# - final mode：使用当前上下文、working_memory 和 tool_observations 产出阶段草稿。
#
# overview：
# - 识别任务意图、用户请求的输出、过滤、指标、操作、时间约束和歧义目标。
# - 只为会改变 grounding、row source、join、filter 或输出列的歧义请求工具。
#
# grounding：
# - 将问题概念落地到具体字段，给出 accepted/rejected 字段及原因。
# - 覆盖输出实体、指标字段、过滤、日期/状态/名称和操作。
# - knowledge SQL / 示例只作为语义线索；若物理字段与 schema 不一致，要把每个输出映射到真实 owner 字段。
# - 同名字段跨实体层与事实/事件/检查层时，用题目措辞判断字段归属。
# - 对每个 filter phrase 识别中心实体名词和限定实体层级，并优先接受同层级字段。
# - “X-related school districts” 这类短语的目标层级是 district，不是 county/city/region/school。
# - 父级/容器地理字段只作为上下文；除非题目明确请求该地理层级，否则应拒绝。
# - 不要因为连接键、等价标识符或请求实体属性不是 row-driving 字段，或过滤在另一表，就拒绝它们。
#
# fabric：
# - 使用已落地字段和工具观察描述 join path、行粒度、输出粒度和关系风险。
# - 优先使用工具观察中的直接连接路径；如果没有，要清楚说明缺失关系。
# - 区分 row_source/data_grain 和 requested output_grain。
#
# contract：
# - 构建 answer contract，包括 answer_columns、row_source、filters、join_policy、enrichment_fields、
#   grouping、metric_operation、metric_fields、row_policy、distinct_policy 和 output_grain。
# - answer_columns[].name 是提交表头；answer_columns[].source_field 是精确源字段。
# - source_field 按请求属性的归属选择，而不是简单跟随 row_source。
# - 如果 row_source 是用于过滤的事实/事件/检查表，实体属性可以通过验证过的 join 从实体表获取。
# - 存储指标阈值过滤使用 metric_operation=lookup；新计算聚合/极值才用 average/sum/count/min/max。
# - 永远不要使用 metric_operation=filter。
# - filters 只能来自同层级的 resolved accepted grounding，不要 OR 竞争层级/粒度解释。
# - per-unit/ratio/rate/normalized-value 过滤必须包含表达式可执行所需的所有 accepted 支撑字段。
# - 对 district/school/organization 等层级短语，不要默认用 county/city/state 等更宽地理字段。
#
# repair_or_critique：
# - 修复当前 working_memory 中的校验错误。
# - 只修复失败或缺失字段；除非错误要求，否则不要重写高置信度 grounding 或 join path。
# - 存储指标阈值导致 metric_operation unknown 时，设为 lookup，不要设为 filter。
# - scope ambiguity 修复时，选择匹配限定实体层级的字段，不要通过跨层级 OR 放宽范围。


def _phase_checklist(phase: str, phase_mode: str = "final") -> list[str]:
    common = [
        "Use exact allowed_field_refs for every field reference.",
        "Do not use a field that is only rejected and not accepted by any concept.",
        "Do not reject join keys or equivalent identifiers merely because they are support fields.",
        "Preserve unresolved answer-changing uncertainty explicitly rather than hiding it.",
    ]
    mode_items = (
        [
            "If evidence is insufficient, request at most four targeted tools.",
            "Do not request tools for facts already present in tool_observations.",
        ]
        if phase_mode == "probe"
        else [
            "Resolve choices using tool_observations before adding remaining_uncertainties.",
            "Do not put facts already confirmed by tool_observations in remaining_uncertainties; zero-row/no-match, date coverage, count, min/max, and distinct-value probe results are verified evidence.",
            "Leave tool_requests empty; record only unresolved answer-changing evidence gaps in remaining_uncertainties.",
        ]
    )
    phase_items = {
        "overview": [
            "List all requested output concepts, not just filters or metrics.",
            "Identify same-name field ambiguity and row-source ambiguity early.",
            "Identify the requested output entity noun, such as Patient, school, event, or record.",
        ],
        "grounding": [
            "Accepted fields should map natural-language terms to concrete fields.",
            "Rejected fields should explain entity-level, metric-level, or scope mismatch.",
            "Do not treat knowledge SQL/examples as authoritative physical schema when allowed_field_refs show a different owner table.",
            "For same-name fields on entity and fact/event/examination tables, prefer the entity table for entity-possessive phrases such as patient's disease or customer's status.",
            "Do not reject entity-table IDs or attributes only because the filter field is in a fact/event/examination table.",
            "For each filter phrase, name the head entity noun and qualifying entity level, then accept fields at that level.",
            "If candidate fields are broader, narrower, or neighboring levels, reject them unless evidence explicitly supports that mapping.",
            "For 'X-related <entity plural>' phrases, bind X to the named entity level, not to a parent/container geography level.",
            "Reject county/city/state/region/country fields for district-, school-, hospital-, company-, department-, or organization-level phrases unless the question explicitly names that geography level.",
            "For patient-level phrases such as 'the patient is diagnosed with', compare patient-level fields against examination/event-level fields.",
            "Before accepting a field for a filter concept, verify the filter value actually exists in that column using execute_probe_query.",
        ],
        "fabric": [
            "Join paths must connect exact field refs and state their purpose.",
            "State whether joins enrich rows, filter rows, or may duplicate rows.",
            "Do not let fact-table row_source automatically change requested output_grain.",
        ],
        "contract": [
            "Use answer_columns[].name from the user's requested headers, not source field names.",
            "Set answer_columns[].source_field from the field that owns the requested attribute, even when row_source is a different filter/fact table.",
            "For entity-possessive output phrases, prefer entity-table attributes through joins unless the question asks for fact/event/record attributes.",
            "Set row_source to the table/asset that drives final rows.",
            "Set output_grain from the requested entity noun, not automatically from row_source.",
            "Put all eligibility conditions and value predicates in filters.",
            "Filters must come from resolved accepted grounding at the filter phrase's entity level; do not include unresolved alternatives.",
            "For per-unit, ratio, rate, or normalized-value filters, include all accepted numerator/denominator support fields.",
            "If grounding contains competing levels, choose the accepted same-level field and do not default to a broader geographic/container field.",
            "Do not use county/city/state/region/country filters for a district-, school-, hospital-, company-, department-, or organization-level phrase unless explicitly requested.",
            "Use OR only for explicit user-requested unions or multiple resolved values of the same concept at the same entity level.",
            "Never OR together competing fields from different entity levels, grains, or semantic roles.",
            "For stored/precomputed metric fields used in filters, such as AvgScrMath > 400, set metric_operation=lookup.",
            "Use metric_operation=average/sum/count/min/max only when the later agent must compute that operation from rows.",
            "Never set metric_operation to filter; filter is not an allowed operation.",
            "For mapped categorical/ordinal filters, use only the exact mapped value unless inclusive range language is present.",
            "Use preserve_all_ties only for explicit extreme/ranking questions; otherwise use row_policy=multiple for listing/filtering tasks.",
            "If output_grain is an entity and row_source is a many-row fact table, set distinct_policy=deduplicate unless all records are requested.",
            "Each filter condition must be verified against real data: use execute_probe_query to confirm at least one row matches.",
            "If a filter uses a categorical label, verify the exact stored value with get_column_distinct_values before writing the filter.",
        ],
        "repair_or_critique": [
            "Fix only the validation errors unless evidence proves a broader issue.",
            "Keep unchanged high-confidence fields stable.",
            "When repairing metric_operation, choose only min|max|sum|count|average|lookup|unknown. Use lookup for stored metric threshold filters; never use filter.",
            "When repairing scope ambiguity, choose the field matching the filter phrase's qualifying entity level; do not broaden with cross-level OR conditions.",
        ],
    }
    return [*common, *mode_items, *phase_items.get(phase, [])]


# 中文翻译注释（仅供阅读，不参与模型输入）：
# _phase_checklist 为每个阶段生成决策检查清单。
#
# 通用检查项：
# - 所有字段引用都必须使用精确 allowed_field_refs。
# - 只被 rejected、没有被任何概念 accepted 的字段不得继续使用。
# - 不要仅因连接键/等价标识符是支撑字段就拒绝它们。
# - 明确保留会改变答案的不确定性，不要用模糊选择隐藏它。
#
# probe 模式：
# - 证据不足时，最多请求四个有针对性的工具。
# - 不要为 tool_observations 中已经有的信息重复请求工具。
#
# final 模式：
# - 先用 tool_observations 解析选择，再添加 remaining_uncertainties。
# - 除非确实还需要有针对性的查询，否则 tool_requests 留空。
#
# overview 检查项：
# - 列出所有请求输出概念，不只列过滤或指标。
# - 早识别同名字段歧义和 row-source 歧义。
# - 识别请求的输出实体名词，如 Patient、school、event、record。
#
# grounding 检查项：
# - accepted_fields 要把自然语言术语映射到具体字段。
# - rejected_fields 要解释实体层级、指标层级或范围不匹配。
# - allowed_field_refs 显示真实 owner table 不同时，不要把 knowledge SQL / 示例当物理 schema 权威。
# - 同名字段出现在实体表和事实/事件/检查表时，对 “patient's disease / customer's status”
#   等实体所属短语优先使用实体表。
# - 不要仅因过滤字段在事实/事件/检查表，就拒绝实体表 ID 或属性。
# - 对每个 filter phrase，命名中心实体名词和限定实体层级，并接受同层级字段。
# - 更宽/更窄/相邻层级候选字段，除非有显式证据支持，否则应拒绝。
# - “X-related <entity plural>” 要把 X 绑定到该 entity 层级，而不是父级/容器地理层级。
# - district/school/hospital/company/department/organization 层级短语，不应使用 county/city/state/
#   region/country 字段，除非题目明确命名该地理层级。
# - 对 “the patient is diagnosed with” 等 patient-level 短语，要比较 patient-level 字段与
#   examination/event-level 字段。
#
# fabric 检查项：
# - join path 必须连接精确 field ref，并说明用途。
# - 说明 join 是补充属性、过滤行，还是可能造成重复行。
# - 不要让事实表 row_source 自动改变请求的 output_grain。
#
# contract 检查项：
# - answer_columns[].name 使用用户请求的表头，而不是源字段名。
# - answer_columns[].source_field 使用拥有该请求属性的字段，即使 row_source 是不同的过滤/事实表。
# - 对实体所属输出短语，除非题目要求事实/事件/记录属性，否则优先通过 join 使用实体表属性。
# - row_source 设置为驱动最终行的表/资产。
# - output_grain 来自请求实体名词，不自动来自 row_source。
# - 所有资格条件和值谓词都放入 filters。
# - filters 必须来自同层级的 resolved accepted grounding，不包含 unresolved alternatives。
# - per-unit/ratio/rate/normalized-value 过滤要包含所有 accepted 的分子/分母支撑字段。
# - grounding 中存在竞争层级时，选择同层级 accepted 字段，不要默认使用更宽地理/容器字段。
# - district/school/hospital/company/department/organization 短语，不要用 county/city/state/region/
#   country 过滤，除非明确请求。
# - OR 只用于用户明确要求的并集，或同一实体层级同一概念的多个已解析值。
# - 不要 OR 不同实体层级、粒度或语义角色的竞争字段。
# - 存储/预计算指标字段用于过滤时，如 AvgScrMath > 400，metric_operation=lookup。
# - 只有后续 agent 需要从行中计算时，才用 average/sum/count/min/max。
# - 永远不要把 metric_operation 设为 filter。
# - 映射型分类/有序过滤，默认只使用精确映射值，除非出现包含性范围语言。
# - preserve_all_ties 只用于明确极值/排名问题；普通列举/过滤任务使用 row_policy=multiple。
# - 若 output_grain 是实体且 row_source 是多行事实表，除非要求所有记录，否则 distinct_policy=deduplicate。
#
# repair_or_critique 检查项：
# - 除非证据证明有更大问题，只修复 validation errors。
# - 保持未受影响的高置信字段稳定。
# - 修复 metric_operation 时，只能选择 min|max|sum|count|average|lookup|unknown；
#   存储指标阈值过滤用 lookup，永远不要用 filter。
# - 修复 scope ambiguity 时，选择匹配 filter phrase 限定实体层级的字段，不要用跨层级 OR 扩大范围。


def _phase_schema(phase: str) -> dict[str, Any]:
    schemas = {
        "overview": {
            "task_intent": "string",
            "concepts": [{"term": "string", "role": "answer_entity|metric|filter|time|operation|unknown"}],
            "ambiguity_targets": ["string"],
            "tool_requests": [{"tool": "allowed tool name", "args": {"query/source/target/asset_path/term/sql/table/column": "string"}}],
        },
        "grounding": {
            "grounded_concepts": [
                {
                    "term": "string",
                    "role": "answer_entity|metric|filter|time|operation|unknown",
                    "accepted_fields": [
                        {"field_ref": "exact allowed_field_refs item", "confidence": "high|medium|low", "reason": "string"}
                    ],
                    "rejected_fields": [{"field_ref": "exact allowed_field_refs item", "reason": "string"}],
                }
            ],
            "remaining_uncertainties": ["string"],
            "tool_requests": [{"tool": "allowed tool name", "args": {"query/source/target/asset_path/term/sql/table/column": "string"}}],
        },
        "fabric": {
            "join_paths": [
                {
                    "purpose": "string",
                    "path": [
                        {"from_field": "exact allowed_field_refs item", "to_field": "exact allowed_field_refs item"}
                    ],
                    "confidence": "high|medium|low",
                }
            ],
            "data_grain": "string",
            "relationship_risks": ["string"],
            "tool_requests": [{"tool": "allowed tool name", "args": {"query/source/target/asset_path/term/sql/table/column": "string"}}],
        },
        "contract": {
            "answer_columns": [
                {
                    "name": "final submitted header, not a field_ref",
                    "source_field": "exact allowed_field_refs item or empty if derived",
                    "reason": "string",
                }
            ],
            "filters": ["field_ref/operator/value in compact text"],
            "group_by": ["exact allowed_field_refs item"],
            "metric_operation": "min|max|sum|count|average|lookup|unknown; use lookup for stored metric threshold filters; never use filter",
            "metric_fields": ["exact allowed_field_refs item"],
            "row_policy": "single|multiple|preserve_all_ties|unknown",
            "distinct_policy": "preserve|deduplicate|unknown",
            "output_grain": "string",
            "row_source": "asset/table/ref that defines final answer rows",
            "join_policy": "inner|left|preserve_left|unknown",
            "enrichment_fields": ["exact allowed_field_refs item used only to add attributes"],
            "remaining_uncertainties": ["string"],
        },
        "repair_or_critique": {
            "grounded_concepts": "optional same shape as grounding",
            "join_paths": "optional same shape as fabric",
            "contract_patch": "optional same shape as contract",
            "remaining_uncertainties": ["string"],
        },
    }
    return schemas.get(phase, schemas["overview"])


# 中文翻译注释（仅供阅读，不参与模型输入）：
# _phase_schema 定义每个阶段要求模型返回的 JSON 形状。
#
# overview schema：
# - task_intent：任务意图。
# - concepts：问题中的概念列表，每个概念包含 term 与 role。
# - ambiguity_targets：会影响答案的不确定点。
# - tool_requests：本阶段想请求的工具调用。
#
# grounding schema：
# - grounded_concepts：每个自然语言概念的字段落地结果。
#   - term：概念名称。
#   - role：概念角色，如 answer_entity / metric / filter / time / operation / unknown。
#   - accepted_fields：接受的精确字段引用及置信度、理由。
#   - rejected_fields：拒绝的精确字段引用及理由。
# - remaining_uncertainties：仍未解决的不确定性。
# - tool_requests：需要的进一步工具请求。
#
# fabric schema：
# - join_paths：连接路径，每条路径说明 purpose、path 和 confidence。
#   - path 中 from_field/to_field 必须是精确 allowed_field_refs。
# - data_grain：数据/行粒度说明。
# - relationship_risks：关系风险，如 join 会扩行、缺失关系、格式不一致等。
# - tool_requests：需要的进一步工具请求。
#
# contract schema：
# - answer_columns：最终提交列，每列包含 name、source_field、reason。
# - filters：紧凑文本形式的字段/操作符/值谓词。
# - group_by：分组字段。
# - metric_operation：min|max|sum|count|average|lookup|unknown；存储指标阈值过滤用 lookup，禁止 filter。
# - metric_fields：指标字段。
# - row_policy：single|multiple|preserve_all_ties|unknown。
# - distinct_policy：preserve|deduplicate|unknown。
# - output_grain：输出粒度。
# - row_source：驱动最终答案行的资产/表/引用。
# - join_policy：inner|left|preserve_left|unknown。
# - enrichment_fields：只用于补充属性的精确字段。
# - remaining_uncertainties：仍未解决的不确定性。
#
# repair_or_critique schema：
# - grounded_concepts：可选 grounding 形状补丁。
# - join_paths：可选 fabric 形状补丁。
# - contract_patch：可选 contract 形状补丁。
# - remaining_uncertainties：修复后仍存在的不确定性。
