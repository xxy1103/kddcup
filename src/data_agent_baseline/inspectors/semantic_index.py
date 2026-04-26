from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


def tokenize(text: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return [token for token in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if token]


def _append_unique(index: dict[str, list[dict[str, Any]]], token: str, ref: dict[str, Any]) -> None:
    if ref not in index[token]:
        index[token].append(ref)


def build_semantic_index(
    *,
    question: str,
    catalog: dict[str, Any],
    perception_payload: dict[str, Any],
) -> dict[str, Any]:
    file_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    field_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    alias_index: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for asset in catalog.get("assets", []):
        for token in tokenize(str(asset.get("path", ""))):
            _append_unique(file_index, token, {"path": asset.get("path"), "kind": asset.get("kind")})

    for schema in catalog.get("schemas", []):
        asset_path = schema.get("asset_path")
        if schema.get("kind") == "sqlite":
            for table in schema.get("tables", []):
                for field in table.get("fields", []):
                    ref = {"asset_path": asset_path, "table": table.get("name"), "field": field.get("name")}
                    for token in tokenize(str(field.get("name", ""))):
                        _append_unique(field_index, token, ref)
                    _add_aliases(alias_index, ref, str(field.get("name", "")))
            continue

        for field in schema.get("fields", []):
            ref = {"asset_path": asset_path, "field": field.get("name")}
            for token in tokenize(str(field.get("name", ""))):
                _append_unique(field_index, token, ref)
            _add_aliases(alias_index, ref, str(field.get("name", "")))

    query_index = {
        token: {
            "files": file_index.get(token, []),
            "fields": field_index.get(token, []),
            "aliases": alias_index.get(token, []),
        }
        for token in tokenize(question)
    }
    risk_index = {
        risk: _risk_candidates(risk, field_index, alias_index)
        for risk in perception_payload.get("high_risk_terms", [])
    }
    tool_routing = {
        str(asset.get("path")): {
            "kind": asset.get("kind"),
            "recommended_tools": asset.get("recommended_tools", []),
        }
        for asset in catalog.get("assets", [])
    }
    return {
        "task_id": catalog.get("task_id"),
        "mode": "keyword",
        "file_token_index": dict(file_index),
        "field_token_index": dict(field_index),
        "field_alias_index": dict(alias_index),
        "query_index": query_index,
        "risk_index": risk_index,
        "tool_routing_index": tool_routing,
    }


def _add_aliases(index: dict[str, list[dict[str, Any]]], ref: dict[str, Any], field_name: str) -> None:
    lowered = field_name.lower()
    aliases: list[str] = []
    if lowered in {"id", "identifier"} or lowered.endswith("_id") or lowered.endswith("id"):
        aliases.extend(["id", "identifier", "key", "join"])
    if "time" in lowered:
        aliases.extend(["time", "finish", "duration"])
    if "rank" in lowered:
        aliases.extend(["rank", "ranked", "placing"])
    if "position" in lowered:
        aliases.extend(["position", "place", "placing"])
    if "number" in lowered or lowered == "no":
        aliases.extend(["number", "no"])
    for alias in aliases:
        _append_unique(index, alias, ref)


def _risk_candidates(
    risk: str,
    field_index: dict[str, list[dict[str, Any]]],
    alias_index: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    if risk == "rank_position_ambiguity":
        return {
            "rank": field_index.get("rank", []) + alias_index.get("rank", []),
            "position": field_index.get("position", []) + alias_index.get("position", []),
        }
    if risk == "per_unit_ratio":
        return {
            "amount": field_index.get("amount", []),
            "price": field_index.get("price", []),
            "unit": field_index.get("unit", []),
        }
    if risk == "number_source_ambiguity":
        return {"number": field_index.get("number", []) + alias_index.get("number", [])}
    return {}
