from __future__ import annotations

from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.config import DataInspectorConfig
from data_agent_baseline.inspectors.prompts import build_lightweight_catalog
from data_agent_baseline.inspectors.semantic_catalog import build_semantic_catalog
from data_agent_baseline.token_utils import count_tokens


class DataUnderstandingAgent:
    def __init__(self, config: DataInspectorConfig) -> None:
        self.config = config

    def explore_data_globally(self, *, context_dir: Path, task_id: str = "") -> tuple[str, dict[str, Any]]:
        catalog = build_semantic_catalog(
            PublicTask(
                record=type("TaskRecord", (), {"task_id": task_id, "difficulty": "", "question": ""})(),
                assets=type("TaskAssets", (), {"task_dir": context_dir, "context_dir": context_dir})(),
            ),
            budget=self.config.sample_budget,
        )
        knowledge_docs: list[dict[str, Any]] = []
        for schema in catalog.get("schemas", []):
            if schema.get("kind") == "document":
                doc_content = schema.get("content") or schema.get("preview") or ""
                if doc_content.strip():
                    knowledge_docs.append(
                        {
                            "asset_path": schema.get("asset_path", ""),
                            "content": doc_content,
                            "token_count": schema.get("token_count", count_tokens(doc_content)),
                            "headings": schema.get("headings", []),
                        }
                    )

        lightweight = build_lightweight_catalog(catalog=catalog, knowledge_docs=knowledge_docs)
        return lightweight, catalog
