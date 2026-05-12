from __future__ import annotations

import json
from typing import Any

def build_lightweight_catalog(
    *,
    catalog: dict[str, Any],
    knowledge_docs: list[dict[str, Any]],
) -> str:
    """Build a lightweight catalog JSON for injection into the agent prompt.

    Only includes asset paths, field names/types, and knowledge document content.
    Omits distinct_values, cardinality, min/max, relationships, and instructions.
    For full field details and join relationships, the agent uses lookup_schema.
    """
    assets_summary = [
        {
            "asset_path": asset.get("asset_path"),
            "kind": asset.get("kind"),
            "size": asset.get("size"),
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

    return json.dumps(
        {
            "task_id": catalog.get("task_id", ""),
            "assets": assets_summary,
            "schemas": schemas_summary,
            "knowledge_documents": [_knowledge_document_payload(doc) for doc in knowledge_docs],
        },
        ensure_ascii=False,
        indent=2,
    )


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
