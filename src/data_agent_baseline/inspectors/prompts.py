from __future__ import annotations

import json
from typing import Any

from data_agent_baseline.token_utils import count_tokens

def build_lightweight_catalog(
    *,
    catalog: dict[str, Any],
    knowledge_docs: list[dict[str, Any]],
) -> str:
    """Build a lightweight catalog JSON for injection into the agent prompt.

    Includes asset paths, field names/types, SQL-style relationships,
    and knowledge document content. Omits heavy sections like distinct_values,
    cardinality, min/max for individual fields — the agent uses lookup_schema for those.
    """
    assets_summary = [
        {
            "asset_path": asset.get("asset_path"),
            "kind": asset.get("kind"),
        }
        for asset in catalog.get("assets", [])
    ]
    schemas_summary: list[dict[str, Any]] = []
    for schema in catalog.get("schemas", []):
        if schema.get("kind") == "document":
            continue
        item: dict[str, Any] = {
            "asset_path": schema.get("asset_path"),
            "kind": schema.get("kind"),
        }
        if schema.get("kind") == "sqlite":
            item["tables"] = [
                {
                    "name": table.get("name"),
                    "fields": [
                        {"name": field.get("name"), "type": field.get("type")}
                        for field in (table.get("fields") or [])
                    ],
                }
                for table in schema.get("tables", [])
            ]
        else:
            item["fields"] = [
                {"name": field.get("name"), "type": field.get("type")}
                for field in (schema.get("fields") or [])
            ]
        schemas_summary.append(item)

    relationships_summary: list[dict[str, Any]] = []
    for rel in catalog.get("relationships", []):
        source = rel["source"]
        target = rel["target"]
        relationships_summary.append(
            {
                "from": f"{source['table']}.{source['fields'][0]}",
                "to": f"{target['table']}.{target['fields'][0]}",
                "type": rel.get("relationship_type"),
                "cardinality": rel.get("cardinality"),
                "confidence": rel.get("confidence"),
            }
        )

    return json.dumps(
        {
            "task_id": catalog.get("task_id", ""),
            "assets": assets_summary,
            "schemas": schemas_summary,
            "relationships": relationships_summary,
            "knowledge_documents": [_knowledge_document_payload(doc) for doc in knowledge_docs],
        },
        ensure_ascii=False,
        indent=2,
    )


def _knowledge_document_payload(doc: dict[str, Any]) -> dict[str, Any]:
    asset_path = str(doc.get("asset_path", ""))
    content = str(doc.get("content", ""))
    token_count = count_tokens(content)
    return {
        "asset_path": asset_path,
        "content": content,
        "token_count": token_count,
        "is_full_content": bool(token_count >= doc.get("token_count", 0)),
    }
