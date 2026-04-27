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
    if "county" in lowered:
        aliases.extend(["county", "geography", "scope"])
    if "district" in lowered:
        aliases.extend(["district", "group", "scope"])
    if "city" in lowered:
        aliases.extend(["city", "geography", "scope"])
    if "state" in lowered:
        aliases.extend(["state", "geography", "scope"])
    if "region" in lowered:
        aliases.extend(["region", "geography", "scope"])
    if "school" in lowered:
        aliases.extend(["school", "entity", "level"])
    if any(token in lowered for token in ("type", "category", "class")):
        aliases.extend(["type", "category", "label"])
    if any(token in lowered for token in ("average", "avg", "mean")):
        aliases.extend(["average", "avg", "mean", "metric"])
    if any(token in lowered for token in ("sum", "total", "count", "score", "amount", "cost")):
        aliases.extend(["metric", "operation"])
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
    if risk == "geographic_scope_ambiguity":
        return {
            "county": field_index.get("county", []) + alias_index.get("county", []),
            "district": field_index.get("district", []) + alias_index.get("district", []),
            "city": field_index.get("city", []) + alias_index.get("city", []),
            "state": field_index.get("state", []) + alias_index.get("state", []),
            "region": field_index.get("region", []) + alias_index.get("region", []),
        }
    if risk == "entity_level_ambiguity":
        return {
            "entity": alias_index.get("entity", []),
            "level": alias_index.get("level", []),
            "group": alias_index.get("group", []),
            "school": field_index.get("school", []) + alias_index.get("school", []),
            "district": field_index.get("district", []) + alias_index.get("district", []),
        }
    if risk in {"aggregation_grain", "metric_operation_ambiguity"}:
        return {
            "metric": alias_index.get("metric", []),
            "operation": alias_index.get("operation", []),
            "average": field_index.get("average", []) + alias_index.get("average", []),
            "count": field_index.get("count", []) + alias_index.get("count", []),
            "total": field_index.get("total", []) + alias_index.get("total", []),
        }
    if risk == "filter_scope_ambiguity":
        return {
            "scope": alias_index.get("scope", []),
            "entity": alias_index.get("entity", []),
            "group": alias_index.get("group", []),
        }
    if risk == "join_key_ambiguity":
        return {
            "id": field_index.get("id", []) + alias_index.get("id", []),
            "key": alias_index.get("key", []),
            "join": alias_index.get("join", []),
        }
    if risk == "type_category_level_ambiguity":
        return {
            "type": field_index.get("type", []) + alias_index.get("type", []),
            "category": field_index.get("category", []) + alias_index.get("category", []),
            "label": alias_index.get("label", []),
        }
    return {}
