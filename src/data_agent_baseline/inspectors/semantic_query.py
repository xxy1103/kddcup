from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Any

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
    max_join_hops: int = 3

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
            for item in schema.get("knowledge_items", []):
                item_text = str(item.get("text") or item.get("snippet", ""))
                item_tokens = set(tokenize(item_text))
                if not term_tokens or not (term_tokens & item_tokens):
                    continue
                hit = {
                    "asset_path": item.get("asset_path") or schema.get("asset_path"),
                    "line": item.get("line_start"),
                    "line_start": item.get("line_start"),
                    "line_end": item.get("line_end"),
                    "section_path": item.get("section_path", []),
                    "evidence_type": item.get("evidence_type", "free_text_note"),
                    "snippet": item.get("snippet", item_text),
                }
                if hit not in hits:
                    hits.append(hit)
                if len(hits) >= effective_limit:
                    return hits
            if hits:
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

    def find_join_paths(
        self,
        source: str,
        target: str,
        *,
        max_hops: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        effective_hops = max_hops or self.max_join_hops
        effective_limit = limit or self.limit
        graph = self._relationship_graph()
        source_nodes = self._matching_nodes(source, graph)
        target_nodes = self._matching_nodes(target, graph)
        if not source_nodes or not target_nodes:
            return []

        paths: list[dict[str, Any]] = []
        target_set = set(target_nodes)
        for source_node in source_nodes:
            queue: deque[tuple[str, list[dict[str, Any]]]] = deque([(source_node, [])])
            seen = {source_node}
            while queue and len(paths) < effective_limit:
                node, path = queue.popleft()
                if node in target_set and path:
                    paths.append(
                        {
                            "source": source_node,
                            "target": node,
                            "path": path,
                            "confidence": _path_confidence(path),
                        }
                    )
                    continue
                if len(path) >= effective_hops:
                    continue
                for edge in graph.get(node, []):
                    next_node = str(edge["to"])
                    if next_node in seen:
                        continue
                    seen.add(next_node)
                    queue.append((next_node, [*path, edge]))
        return paths

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

        knowledge_terms = list(dict.fromkeys([*tokenize(query), "amount", "spent", "cost"]))[:8]
        knowledge_hits: list[dict[str, Any]] = []
        for term in knowledge_terms:
            for hit in self.lookup_knowledge(term, limit=2):
                if hit not in knowledge_hits:
                    knowledge_hits.append(hit)
                if len(knowledge_hits) >= self.limit:
                    break
            if len(knowledge_hits) >= self.limit:
                break

        join_paths = []
        query_tokens = set(tokenize(query))
        if {"cost", "event"} <= query_tokens:
            join_paths.extend(self.find_join_paths("records.cost", "records.event_name", limit=self.limit))
        for path in self._join_paths_for_candidates(unique_field_candidates):
            if path not in join_paths:
                join_paths.append(path)
        return {
            "query": query,
            "field_candidates": unique_field_candidates[: self.limit * 2],
            "asset_schemas": [
                schema
                for schema in (self.get_asset_schema(str(path)) for path in asset_paths[: self.limit])
                if schema is not None
            ],
            "knowledge_hits": knowledge_hits[: self.limit],
            "join_paths": join_paths[: self.limit],
            "risk_candidates": self.semantic_index.get("risk_index", {}),
            "rejected_field_hints": self._rejected_field_hints(query, unique_field_candidates),
        }

    def known_field_refs(self) -> set[str]:
        return set(self._all_fields())

    def _join_paths_for_candidates(self, field_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        field_refs = [
            _field_ref(str(item.get("asset_path")), str(item.get("field")), item.get("table"))
            for item in field_candidates
            if item.get("asset_path") and item.get("field")
        ]
        paths: list[dict[str, Any]] = []
        for source in field_refs:
            for target in field_refs:
                if source == target:
                    continue
                for path in self.find_join_paths(source, target, limit=1):
                    if path not in paths:
                        paths.append(path)
                        if len(paths) >= self.limit:
                            return paths
        return paths

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

    def _relationship_graph(self) -> dict[str, list[dict[str, Any]]]:
        graph: dict[str, list[dict[str, Any]]] = {}
        fields = self._all_fields()
        for left in fields:
            for right in fields:
                if left == right:
                    continue
                reason, confidence = _relationship_reason(left, right)
                if reason is None:
                    continue
                graph.setdefault(left, []).append(
                    {
                        "from": left,
                        "to": right,
                        "confidence": confidence,
                        "reason": reason,
                    }
                )
        relationship_fields = [field for field in fields if _looks_like_relationship_field(field)]
        for left in relationship_fields:
            for right in relationship_fields:
                if left == right or self._asset_for_ref(left) != self._asset_for_ref(right):
                    continue
                graph.setdefault(left, []).append(
                    {
                        "from": left,
                        "to": right,
                        "confidence": "medium",
                        "reason": "relationship fields in the same asset can bridge records",
                    }
                )
        return graph

    def _matching_nodes(self, query: str, graph: dict[str, list[dict[str, Any]]]) -> list[str]:
        nodes = set(graph)
        for edges in graph.values():
            nodes.update(str(edge["to"]) for edge in edges)
        match_nodes = nodes | set(self._all_fields())
        query_tokens = set(tokenize(query))
        normalized_query = _normalize_field_key(query)
        matches = [
            node
            for node in match_nodes
            if normalized_query and normalized_query in _normalize_field_key(node)
        ]
        if matches:
            return self._expand_to_same_asset_relationship_nodes(matches, nodes)
        query_base = _normalize_field_key(_field_basename(query))
        basename_matches = [
            node
            for node in match_nodes
            if query_base and _normalize_field_key(node).endswith(query_base)
        ]
        if basename_matches:
            return self._expand_to_same_asset_relationship_nodes(basename_matches, nodes)
        token_matches = [
            node
            for node in match_nodes
            if query_tokens and query_tokens & set(tokenize(node))
        ]
        return self._expand_to_same_asset_relationship_nodes(token_matches, nodes)

    def _expand_to_same_asset_relationship_nodes(self, matches: list[str], nodes: set[str]) -> list[str]:
        expanded = list(matches)
        for match in matches:
            asset = self._asset_for_ref(match)
            if asset is None:
                continue
            for node in nodes:
                if node in expanded:
                    continue
                if self._asset_for_ref(node) == asset and _looks_like_relationship_field(node):
                    expanded.append(node)
        return expanded

    def _asset_for_ref(self, ref: str) -> str | None:
        for asset in self.catalog.get("assets", []):
            asset_path = str(asset.get("path"))
            if ref == asset_path or ref.startswith(f"{asset_path}."):
                return asset_path
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
            }
            for field in fields
        ]
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
