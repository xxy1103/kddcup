from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_agent_baseline.inspectors.probe_engine import (
    execute_probe_query as _execute_probe_query,
    get_column_distinct_values as _get_column_distinct_values,
)
from data_agent_baseline.inspectors.semantic_index import tokenize

def _field_ref(asset_path: str, field: str, table: str | None = None) -> str:
    if table:
        return f"{asset_path}.{table}.{field}"
    return f"{asset_path}.{field}"

def _field_basename(field: str) -> str:
    return field.rsplit(".", 1)[-1]

def _normalize_field_key(field: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _field_basename(field).lower())

def _entity_from_asset(asset_path: str) -> str:
    name = asset_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return re.sub(r"[^a-z0-9]+", "", name.lower())

def _singular(value: str) -> str:
    return value[:-1] if value.endswith("s") and len(value) > 3 else value

@dataclass(frozen=True, slots=True)
class SemanticQueryTools:
    catalog: dict[str, Any]
    semantic_index: dict[str, Any]
    limit: int = 5
    context_dir: Path | None = None

    def search_semantic_index(
        self,
        query: str,
        *,
        scopes: tuple[str, ...] = ("files", "fields", "aliases"),
        limit: int | None = None,
    ) -> dict[str, Any]:
        effective_limit = limit or self.limit
        merged: dict[str, list[dict[str, Any]]] = {scope: [] for scope in scopes}
        for token in tokenize(query):
            token_result = self.semantic_index.get("query_index", {}).get(token)
            if token_result is None:
                token_result = {
                    "files": self.semantic_index.get("file_token_index", {}).get(token, []),
                    "fields": self.semantic_index.get("field_token_index", {}).get(token, []),
                    "aliases": self.semantic_index.get("field_alias_index", {}).get(token, []),
                }
            for scope in scopes:
                for item in token_result.get(scope, []):
                    if item not in merged[scope]:
                        merged[scope].append(item)
        return {scope: values[:effective_limit] for scope, values in merged.items()}

    def get_asset_schema(self, asset_path: str, *, include_samples: bool = False) -> dict[str, Any] | None:
        schema = next(
            (item for item in self.catalog.get("schemas", []) if item.get("asset_path") == asset_path),
            None,
        )
        if schema is None:
            schema = self._sqlite_table_schema(asset_path)
        if schema is None:
            return None
        if include_samples:
            return schema
        return _strip_large_schema_payload(schema)

    def lookup_knowledge(self, term: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        effective_limit = limit or self.limit
        term_tokens = set(tokenize(term))
        hits: list[dict[str, Any]] = []
        for schema in self.catalog.get("schemas", []):
            if schema.get("kind") != "document":
                continue
            text = str(schema.get("content") or schema.get("preview", ""))
            lines = text.splitlines()
            for index, line in enumerate(lines):
                line_tokens = set(tokenize(line))
                if not term_tokens or not (term_tokens & line_tokens):
                    continue
                start = max(index - 1, 0)
                end = min(index + 2, len(lines))
                hits.append(
                    {
                        "asset_path": schema.get("asset_path"),
                        "line": index + 1,
                        "snippet": "\n".join(lines[start:end]).strip(),
                    }
                )
                if len(hits) >= effective_limit:
                    return hits
        return hits

    def build_context_bundle(self, query: str) -> dict[str, Any]:
        search_result = self.search_semantic_index(query, limit=self.limit)
        field_candidates = search_result.get("fields", []) + search_result.get("aliases", [])
        unique_field_candidates: list[dict[str, Any]] = []
        for field in field_candidates:
            if field not in unique_field_candidates:
                unique_field_candidates.append(field)
        for token in tokenize(query):
            for field in self.semantic_index.get("field_token_index", {}).get(token, []):
                if field not in unique_field_candidates:
                    unique_field_candidates.append(field)
            for field in self.semantic_index.get("field_alias_index", {}).get(token, []):
                if field not in unique_field_candidates:
                    unique_field_candidates.append(field)

        asset_paths = [item.get("path") for item in search_result.get("files", []) if item.get("path")]
        for field in unique_field_candidates:
            asset_path = field.get("asset_path")
            if asset_path and asset_path not in asset_paths:
                asset_paths.append(asset_path)

        knowledge_terms = list(dict.fromkeys([*tokenize(query), "amount", "spent", "cost", "attribute", "value"]))
        knowledge_hits: list[dict[str, Any]] = []
        for term in knowledge_terms:
            for hit in self.lookup_knowledge(term, limit=2):
                if hit not in knowledge_hits:
                    knowledge_hits.append(hit)
                    snippet = hit.get("snippet", "").lower()
                    for schema in self.catalog.get("schemas", []):
                        asset_path = str(schema.get("asset_path", ""))
                        table_name = asset_path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
                        if table_name in snippet and asset_path not in asset_paths:
                            asset_paths.append(asset_path)
                if len(knowledge_hits) >= self.limit * 2:
                    break
            if len(knowledge_hits) >= self.limit * 2:
                break

        return {
            "query": query,
            "field_candidates": unique_field_candidates[: self.limit * 2],
            "asset_schemas": [
                schema
                for schema in (self.get_asset_schema(str(path)) for path in asset_paths[: self.limit])
                if schema is not None
            ],
            "knowledge_hits": knowledge_hits[: self.limit],
            "risk_candidates": self.semantic_index.get("risk_index", {}),
            "rejected_field_hints": self._rejected_field_hints(query, unique_field_candidates),
        }

    def known_field_refs(self) -> set[str]:
        return set(self._all_fields())

    def _rejected_field_hints(self, query: str, field_candidates: list[dict[str, Any]]) -> list[dict[str, str]]:
        query_tokens = set(tokenize(query))
        hints: list[dict[str, str]] = []
        if "cost" not in query_tokens:
            return hints
        candidate_items = list(field_candidates)
        for field_ref in self._all_fields():
            asset_path, field = _split_field_ref(field_ref, self.catalog.get("assets", []))
            candidate_items.append({"asset_path": asset_path, "field": field})

        for item in candidate_items:
            field = str(item.get("field", ""))
            asset_path = str(item.get("asset_path", ""))
            basename = _normalize_field_key(field)
            if basename == "amount":
                hints.append(
                    {
                        "asset": asset_path,
                        "field": field,
                        "reason": "amount usually means budgeted amount, not actual expense cost.",
                    }
                )
            if basename == "spent":
                hints.append(
                    {
                        "asset": asset_path,
                        "field": field,
                        "reason": "spent is usually an aggregate budget expenditure, not a single expense cost.",
                    }
                )
        deduped: list[dict[str, str]] = []
        for hint in hints:
            if hint not in deduped:
                deduped.append(hint)
        return deduped[: self.limit]

    def _sqlite_table_schema(self, table_ref: str) -> dict[str, Any] | None:
        for schema in self.catalog.get("schemas", []):
            if schema.get("kind") != "sqlite":
                continue
            asset_path = str(schema.get("asset_path", ""))
            prefix = f"{asset_path}."
            if not table_ref.startswith(prefix):
                continue
            table_name = table_ref[len(prefix) :]
            table = next((item for item in schema.get("tables", []) if item.get("name") == table_name), None)
            if table is None:
                continue
            narrowed = dict(schema)
            narrowed["tables"] = [table]
            return narrowed
        return None

    def _all_fields(self) -> list[str]:
        refs: list[str] = []
        for schema in self.catalog.get("schemas", []):
            asset_path = str(schema.get("asset_path"))
            if schema.get("kind") == "sqlite":
                for table in schema.get("tables", []):
                    for field in table.get("fields", []):
                        refs.append(_field_ref(asset_path, str(field.get("name")), table.get("name")))
                continue
            for field in schema.get("fields", []):
                refs.append(_field_ref(asset_path, str(field.get("name"))))
        return refs

    def execute_probe_query(
        self,
        sql: str,
        *,
        limit: int = 5,
    ) -> dict[str, Any]:
        """Execute a read-only SQL probe query against actual task data.

        Only SELECT/WITH statements are allowed.  Table names must match
        file-name stems (e.g. ``drivers`` for ``drivers.csv``).
        """
        if self.context_dir is None:
            return {"ok": False, "error": "Probe tools require context_dir; none configured."}
        return _execute_probe_query(self.context_dir, self.catalog, sql, limit=limit)

    def get_column_distinct_values(
        self,
        table: str,
        column: str,
        *,
        top_n: int = 20,
    ) -> dict[str, Any]:
        """Return the most frequent distinct values for a column.

        *table* is the file-name stem for CSV/JSON assets, or the SQLite
        table name for ``.db`` assets.
        """
        if self.context_dir is None:
            return {"ok": False, "error": "Probe tools require context_dir; none configured."}
        return _get_column_distinct_values(self.context_dir, self.catalog, table, column, top_n=top_n)

def _strip_large_schema_payload(schema: dict[str, Any]) -> dict[str, Any]:
    stripped = dict(schema)
    stripped.pop("content", None)
    stripped.pop("preview", None)
    if "sample_rows" in stripped:
        stripped["sample_rows"] = stripped["sample_rows"][:3]
    fields = stripped.get("fields")
    if isinstance(fields, list):
        stripped["fields"] = [
            {
                "name": field.get("name"),
                "type": field.get("type"),
                "sample_values": field.get("sample_values", [])[:3],
                "missing_count": field.get("missing_count"),
                "cardinality": field.get("cardinality"),
                "distinct_values": (field.get("distinct_values") or [])[:10],
                **({"min_value": field["min_value"], "max_value": field["max_value"]} if "min_value" in field else {}),
            }
            for field in fields
        ]
    # Propagate row_count and json_structure for non-sqlite schemas.
    if "row_count" in schema and stripped.get("kind") != "sqlite":
        stripped["row_count"] = schema["row_count"]
    if stripped.get("kind") == "json" and "json_structure" in schema:
        stripped["json_structure"] = schema["json_structure"]
    return stripped

def _split_field_ref(field_ref: str, assets: list[dict[str, Any]]) -> tuple[str, str]:
    for asset in assets:
        asset_path = str(asset.get("path"))
        prefix = f"{asset_path}."
        if field_ref.startswith(prefix):
            return asset_path, field_ref[len(prefix) :]
    return "", field_ref

def _relationship_reason(left: str, right: str) -> tuple[str | None, str]:
    left_base = _normalize_field_key(left)
    right_base = _normalize_field_key(right)
    left_asset = _entity_from_asset(left.split(".", 1)[0])
    right_asset = _entity_from_asset(right.split(".", 1)[0])
    if left_base == right_base and ("id" in left_base or left_base.endswith("id")):
        return "same normalized id field", "medium"
    if left_base.startswith("linkto") and right_base.endswith("id"):
        linked_entity = _singular(left_base[len("linkto") :])
        right_entity = _singular(right_asset)
        if linked_entity and linked_entity == right_entity:
            return "link_to field points to target asset id", "high"
    if right_base.startswith("linkto") and left_base.endswith("id"):
        linked_entity = _singular(right_base[len("linkto") :])
        left_entity = _singular(left_asset)
        if linked_entity and linked_entity == left_entity:
            return "link_to field points to target asset id", "high"
    return None, "low"

def _path_confidence(path: list[dict[str, Any]]) -> str:
    if path and all(edge.get("confidence") == "high" for edge in path):
        return "high"
    if path:
        return "medium"
    return "low"

def _looks_like_relationship_field(ref: str) -> bool:
    key = _normalize_field_key(ref)
    return key.startswith("linkto") or key.endswith("id") or key == "id"

def _looks_like_link_field(ref: str) -> bool:
    return _normalize_field_key(ref).startswith("linkto")
