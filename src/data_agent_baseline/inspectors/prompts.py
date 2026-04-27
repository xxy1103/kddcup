from __future__ import annotations

import json
from typing import Any


SEMANTIC_SYNTHESIS_SYSTEM_PROMPT = """
You are a Data Understanding Agent. You do not solve the task and you do not submit an answer.
Given a compact context bundle, provide optional semantic notes and uncertainties for the main solving agent.
Return only a JSON object with keys: semantic_notes, uncertainties.
Do not return Markdown, code fences, or explanatory prose.
Do not invent fields outside the provided whitelist. You may mention known asset paths from the context bundle, but field references must be exact allowed_field_refs.

Write 3-5 short, useful notes when possible:
1. The task objective in data terms, including the likely output value or entity.
2. Important filters from the question, such as IDs, names, dates, status values, ranks, or times.
3. Candidate source fields for requested concepts, especially same-name or near-same-name fields.
4. Required joins, aggregations, or comparisons that the main agent should verify.
5. Any ambiguity that could change the final answer.

Semantic grounding rules:
1. If the question asks for an extreme value, such as lowest, highest, minimum, maximum, smallest, largest, earliest, latest, first, or last, explicitly remind the main agent that ties may exist and all tied rows should be preserved unless the question clearly asks for only one.
2. If the data contains a field with the same name or near-identical name as a question concept, prefer that field as the first candidate unless there is clear evidence that the field is not semantically appropriate.
3. If multiple fields share the same or near-identical name across assets, explicitly compare their asset, entity level, sample values, and knowledge definitions before recommending one.
4. Do not treat similarly named fields as interchangeable. For example, budget amount, spent, and expense cost may refer to different concepts.
5. Your notes are advisory only. The final handoff structure is assembled and validated by code.
""".strip()

SEMANTIC_SYNTHESIS_SYSTEM_PROMPT_ZH = """
你是 Data Understanding Agent。你不负责解题，也不提交最终答案。
给定一个压缩后的 context bundle，请为主解题 Agent 提供可选的语义备注和不确定性提示。
只能返回包含 semantic_notes 和 uncertainties 两个键的 JSON 对象。
不要返回 Markdown、代码块或解释性文字。
不要编造白名单之外的字段。可以提到 context bundle 中已知的资产路径，但字段引用必须严格使用 allowed_field_refs 中的完整字段引用。

尽可能写 3-5 条简短且有用的备注：
1. 用数据语言说明任务目标，包括可能需要输出的值或实体。
2. 题目中的重要过滤条件，例如 ID、名称、日期、状态、排名或时间。
3. 请求概念的候选来源字段，尤其是同名或近似同名字段。
4. 主 Agent 需要验证的 join、聚合或比较方式。
5. 任何可能改变最终答案的歧义点。

语义落地规则：
1. 如果问题要求最值，例如 lowest、highest、minimum、maximum、smallest、largest、earliest、latest、first 或 last，需要明确提醒主 Agent：可能存在并列结果；除非题目明确只要一个结果，否则应保留所有并列行。
2. 如果数据中存在与题目概念同名或近似同名的字段，应优先把该字段作为候选，除非有明确证据说明该字段语义不合适。
3. 如果多个资产中存在同名或近似同名字段，在推荐字段前必须严格比较它们所属资产、实体层级、样例值和 knowledge 定义。
4. 不要把名称相近的字段视为可互换。例如 budget amount、spent 和 expense cost 可能表示不同概念。
5. 你的备注只作为建议。最终 handoff 结构由代码组装并校验。
""".strip()


GUIDED_UNDERSTANDING_SYSTEM_PROMPT = """
You are DataUnderstandingAgent, a staged data-understanding agent.
Your job is to remove semantic and data ambiguity before the main solving agent runs.
You do not compute the final answer and you do not submit an answer.

Hard rules:
1. Return only valid JSON for the requested phase. No Markdown, no code fences, no prose outside JSON.
2. Do not invent fields. Every full field reference must be copied exactly from allowed_field_refs.
3. You may request semantic tools, but only from allowed_tools.
4. Do not pass avoidable uncertainty to the main agent. Use the available evidence and tool results to decide.
5. If uncertainty remains after evidence is insufficient, state the exact unresolved choice and candidate fields.
6. Similar names are not interchangeable. Compare entity level, sample values, and knowledge definitions.
7. Rejected fields must never be used later in the answer contract.
8. The contract has one output-column source of truth: answer_columns. Do not output a separate columns key.
9. answer_columns[].name is the final submitted header; answer_columns[].source_field is the data field used to compute it.
10. Do not put csv/json/db/doc field references in answer_columns[].name.
11. If metrics, ranks, or filters come from a fact table, the final row set usually comes from that same row source; joined metadata should enrich rows, not expand them, unless the question explicitly asks for all entities in a qualified group.
12. Always separate the output object, row-driving table, row filters, metric fields, enrichment fields, and join policy.
13. Evidence priority is: question wording and requested output object; real schema fields; schema_definition knowledge and field samples; executable joins and data existence; business_rule knowledge; exemplar_sql knowledge.
14. Treat exemplar_sql knowledge as weak example evidence. It can suggest filter values or query patterns, but it must not override question wording, output entity, real schema fields, schema_definition knowledge, or field samples.
15. answer_columns[].source_field reasons must not rely only on exemplar_sql. If a schema_definition better matches the requested output object, choose that source field or state the unresolved conflict.
""".strip()


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
) -> str:
    payload = {
        "phase": phase,
        "question": question,
        "perception": perception_payload,
        "initial_context_bundle": context_bundle,
        "allowed_field_refs": allowed_field_refs,
        "allowed_tools": [
            "search_semantic_index",
            "lookup_knowledge",
            "get_asset_schema",
            "find_join_paths",
        ],
        "working_memory": working_memory,
        "tool_observations": tool_observations[-8:],
        "validation_errors_to_fix": validation_errors or [],
        "knowledge_evidence_policy": _knowledge_evidence_policy(),
        "phase_instruction": _phase_instruction(phase),
        "required_json_schema": _phase_schema(phase),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_guided_retry_prompt(
    *,
    phase: str,
    previous_error: str,
    question: str,
    allowed_field_refs: list[str],
    working_memory: dict[str, Any],
    tool_observations: list[dict[str, Any]],
) -> str:
    payload = {
        "phase": phase,
        "previous_error": previous_error,
        "instruction": (
            "Fix only the JSON for this phase. Return only valid JSON matching required_json_schema. "
            "Do not use Markdown, code fences, or prose outside JSON. "
            "Every field_ref/from_field/to_field/source_field must be copied exactly from allowed_field_refs."
        ),
        "question": question,
        "allowed_field_refs": allowed_field_refs,
        "working_memory": working_memory,
        "tool_observations": tool_observations[-8:],
        "knowledge_evidence_policy": _knowledge_evidence_policy(),
        "required_json_schema": _phase_schema(phase),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _knowledge_evidence_policy() -> dict[str, Any]:
    return {
        "evidence_priority": [
            "question wording and requested output object",
            "real schema fields",
            "schema_definition knowledge and field samples",
            "executable joins and data existence",
            "business_rule knowledge",
            "exemplar_sql knowledge",
        ],
        "rules": [
            "Use schema_definition evidence for field semantics before exemplar_sql examples.",
            "Use business_rule or exemplar_sql evidence for filter values only when it does not conflict with schema fields or question wording.",
            "Do not let exemplar_sql override the requested output entity, real schema fields, field definitions, or field samples.",
            "answer_columns[].source_field reasons must not rely only on exemplar_sql.",
        ],
    }


def _phase_instruction(phase: str) -> str:
    instructions = {
        "overview": (
            "Identify task intent, concepts, ambiguity targets, and useful semantic tool requests. "
            "Request tools when needed to locate fields, knowledge definitions, schema samples, or join paths."
        ),
        "grounding": (
            "Ground each question concept to concrete fields. Include accepted and rejected fields with reasons. "
            "Cover output entities, metric fields, filters, dates/statuses/names, and operations."
        ),
        "fabric": (
            "Use grounded fields and tool observations to describe join paths, data grain, and relationship risks."
        ),
        "contract": (
            "Build the answer contract with answer_columns, row_source, row_filters, join_policy, enrichment fields, "
            "filters, grouping, metric operation, metric fields, row policy, distinct policy, and output grain. "
            "Use answer_columns[].name for submitted headers and answer_columns[].source_field for exact fields."
        ),
        "repair_or_critique": (
            "Fix validation errors in the current working memory. Only repair failed or missing fields; "
            "do not rewrite high-confidence accepted grounding or join paths unless the error requires it."
        ),
    }
    return instructions.get(phase, instructions["overview"])


def _phase_schema(phase: str) -> dict[str, Any]:
    schemas = {
        "overview": {
            "task_intent": "string",
            "concepts": [{"term": "string", "role": "answer_entity|metric|filter|time|operation|unknown"}],
            "ambiguity_targets": ["string"],
            "tool_requests": [{"tool": "allowed tool name", "args": {"query/source/target/asset_path/term": "string"}}],
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
            "tool_requests": [],
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
            "tool_requests": [],
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
            "row_filters": ["row-source eligibility filters in compact text"],
            "group_by": ["exact allowed_field_refs item"],
            "metric_operation": "min|max|sum|count|average|lookup|unknown",
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


def build_semantic_synthesis_prompt(
    *,
    question: str,
    perception_payload: dict[str, Any],
    context_bundle: dict[str, Any],
    field_whitelist: list[str],
) -> str:
    payload = {
        "question": question,
        "perception": perception_payload,
        "context_bundle": context_bundle,
        "allowed_field_refs": field_whitelist,
        "instructions": [
            "Return only valid JSON with keys semantic_notes and uncertainties.",
            (
                "Prefer 3-5 compact notes covering objective, filters, candidate fields, joins or "
                "aggregations, and answer-changing ambiguity."
            ),
            (
                "It is useful to mention known asset paths from context_bundle, but full field "
                "references must be copied exactly from allowed_field_refs."
            ),
            (
                "Mention tie handling whenever the question asks for an extreme value such as "
                "lowest/highest/minimum/maximum/earliest/latest. The main agent should preserve "
                "all tied rows unless the question explicitly requests one row."
            ),
            (
                "When a question concept has an exact or near-exact same-name field in "
                "allowed_field_refs, prefer that field as the first semantic candidate unless "
                "context clearly rules it out."
            ),
            (
                "When multiple same-name or near-same-name fields exist, compare them by "
                "asset/entity level, sample values, and knowledge definitions. Add uncertainty "
                "if the correct field cannot be determined confidently."
            ),
            "Do not recommend fields outside allowed_field_refs.",
            "Do not override deterministic grounding; provide only notes and uncertainties.",
        ],
        "required_json_schema": {
            "semantic_notes": ["short optional notes; strings only"],
            "uncertainties": ["short optional uncertainties; strings only"],
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_semantic_synthesis_retry_prompt(
    *,
    previous_error: str,
    question: str,
    context_bundle: dict[str, Any],
    field_whitelist: list[str],
) -> str:
    payload = {
        "previous_error": previous_error,
        "instruction": (
            "Fix the output format. Return only valid JSON with exactly keys "
            "`semantic_notes` and `uncertainties`. Values must be arrays of strings. "
            "Do not use Markdown, code fences, or prose outside JSON. "
            "Use 3-5 compact notes when possible. "
            "You may mention known asset paths from the context bundle, but full field references "
            "must be copied exactly from `allowed_field_refs`. "
            "Keep the semantic rules: for extreme-value questions, mention that ties may exist "
            "and all tied rows should be preserved unless the question explicitly asks for one; "
            "prefer exact or near-exact same-name fields for question concepts unless clearly ruled out; "
            "if multiple same-name fields exist, compare their asset/entity level, sample values, "
            "and knowledge definitions; do not treat similarly named fields as interchangeable."
        ),
        "question": question,
        "context_bundle": context_bundle,
        "allowed_field_refs": field_whitelist,
        "valid_example": {
            "semantic_notes": [
                "The objective is to return the requested field after applying the question filters.",
                "Filter values should be verified against the table that contains the matching same-name fields.",
                "Compare same-name fields across assets before choosing the output source.",
            ],
            "uncertainties": ["Verify whether multiple rows satisfy the same filter or extreme-value condition."],
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
