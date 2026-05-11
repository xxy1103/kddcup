from __future__ import annotations

import json
from typing import Any

_MAX_DISTINCT_VALUE_CHARS = 200


def _truncate_distinct_values(values: list[Any], max_chars: int = _MAX_DISTINCT_VALUE_CHARS) -> list[str]:
    truncated: list[str] = []
    for v in values:
        s = str(v)
        if len(s) > max_chars:
            truncated.append(s[:max_chars] + "...[truncated]")
        else:
            truncated.append(s)
    return truncated


def build_global_profiling_prompt(
    *,
    catalog: dict[str, Any],
    knowledge_docs: list[dict[str, Any]],
) -> str:
    assets_summary = [
        {
            "path": asset.get("path"),
            "kind": asset.get("kind"),
            "size": asset.get("size"),
        }
        for asset in catalog.get("assets", [])
    ]
    schemas_summary = []
    for schema in catalog.get("schemas", []):
        if schema.get("kind") == "document":
            continue
        item = {
            "asset_path": schema.get("asset_path"),
            "kind": schema.get("kind"),
        }
        if schema.get("kind") == "sqlite":
            item["tables"] = schema.get("tables", [])
        else:
            item["fields"] = [
                {
                    "name": field.get("name"),
                    "type": field.get("type"),
                    "missing_count": field.get("missing_count"),
                    "cardinality": field.get("cardinality"),
                    "distinct_values": _truncate_distinct_values(field.get("distinct_values", [])),
                    **({"min_value": field["min_value"], "max_value": field["max_value"]} if "min_value" in field else {}),
                }
                for field in (schema.get("fields") or [])
            ]
        if schema.get("kind") != "sqlite":
            item["row_count"] = schema.get("row_count")
        if schema.get("kind") == "json" and schema.get("json_structure"):
            item["json_structure"] = schema["json_structure"]
        schemas_summary.append(item)

    uncertainties = [
        {"asset_path": u.get("asset_path"), "risk": u.get("risk"), "instruction": u.get("instruction")}
        for u in catalog.get("semantic_uncertainties", [])
    ]

    payload = {
        "phase": "global_data_profiling",
        "task_id": catalog.get("task_id", ""),
        "assets": assets_summary,
        "schemas": schemas_summary,
        "relationships": catalog.get("relationships", []),
        "relationship_warnings": catalog.get("relationship_warnings", []),
        "semantic_uncertainties": uncertainties,
        "knowledge_documents": [_knowledge_document_payload(doc) for doc in knowledge_docs],
        "required_json_schema": {
            "profile_markdown": "complete markdown document starting with `## Global Data Profile`",
        },
        "instructions": [
            "Produce a dense, factual, self-contained profile of the data landscape.",
            "Classify each asset as Entity Master, Event Log, or Junction table.",
            "For every categorical field: distinct_values shows the top 50 most frequent values; cardinality reports the total distinct count. Enumerate all displayed values verbatim. When cardinality > 50, note the count of additional rare values not shown.",
            "For every Junction/Bridge table: document BOTH join paths — from each side entity through the junction to the opposite side, using explicit Table.column notation at every hop.",
            "Write explicit multi-hop join paths for ALL entity pairs. Do not assume the reader knows how to traverse a junction table. Every hop must be spelled out.",
            "Cross-reference codes with knowledge.md labels. Mark unlabeled values as (unlabeled) rather than guessing.",
            "Anchor knowledge-base formulas and terminology to specific tables/columns.",
            "Every quality observation must cite a specific, verifiable catalog fact (count, range, presence). Do not write 'appears to be a sample', 'seems incomplete', or similar speculation without hard evidence.",
            "Note date formats, nullability, and semantic look-alikes.",
            "Do not guess the user question; maintain a domain-agnostic investigative tone.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _knowledge_document_payload(doc: dict[str, Any]) -> dict[str, Any]:
    asset_path = str(doc.get("asset_path", ""))
    content = str(doc.get("content", ""))
    content_len = len(content)
    headings = doc.get("headings", [])
    return {
        "asset_path": asset_path,
        "content": content,
        "char_count": content_len,
        "is_full_content": bool(content_len >= doc.get("char_count", 0)),
        "headings": [str(h) for h in headings] if headings else [],
    }
