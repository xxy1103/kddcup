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
