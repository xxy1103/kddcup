"""One-shot test: run MemAgent ETL on task_396 superhero.md via easy.yaml config."""

import json
import sys
from pathlib import Path

from data_agent_baseline.config import load_app_config
from data_agent_baseline.agents.model import create_chat_model
from data_agent_baseline.tools.memagent_etl import (
    extract_tables_from_documents,
    list_store_tables,
    summarize_extraction,
)
from data_agent_baseline.tools.registry import _build_memagent_rule_provider

CONFIG_PATH = Path("configs/easy.yaml")


def main():
    config = load_app_config(CONFIG_PATH)
    print(f"Model: {config.agent.model}")
    print(f"Base URL: {config.agent.api_base}")

    model = create_chat_model(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        api_key_env=config.agent.api_key_env,
        temperature=config.agent.temperature,
    )

    # connectivity test
    print("\n[1] Testing model...")
    resp = model.invoke("OK")
    raw = str(resp.content).encode('ascii', errors='replace').decode('ascii')
    print(f"    OK (response: {raw[:80]})")

    doc_path = Path("data/public/input/task_396/context/doc/superhero.md")
    print(f"\n[2] Document: {doc_path.name} ({doc_path.stat().st_size} bytes)")

    rule_provider = _build_memagent_rule_provider(model)

    print("[3] Running extraction (2 repair rounds)...")
    ext_result = extract_tables_from_documents(
        sources=[("doc/superhero.md", doc_path)],
        workspace_root=Path("artifacts/memagent_test"),
        task_id="test_396",
        goal=(
            "Each paragraph describes one superhero. Extract: id (registration number "
            "like 'registration number 7', 'identifier 26', 'reference code 28', "
            "'registry number 31', 'ID 45', etc.), superhero_name (codename after "
            "'known as', 'designated', 'operating under'), full_name (civilian name), "
            "height_cm (height in centimeters), weight_kg (weight in kilograms), "
            "publisher_id (publisher affiliation number like 'publisher affiliation is 13', "
            "'registered with publisher 13')"
        ),
        rule_provider=rule_provider,
    )

    summary = summarize_extraction(ext_result)
    print("\n=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n=== Data Tables ===")
    tables = list_store_tables(ext_result.sqlite_path)
    for t in tables["tables"]:
        if t["name"].startswith("_"):
            continue
        print(f"\n--- {t['name']} ({t['row_count']} rows) ---")
        print(f"  Columns: {t['columns']}")
        for row in t["sample_rows"][:10]:
            print(f"  {json.dumps(row, ensure_ascii=False)}")

    # Show unresolved stats
    unresolved = [t for t in tables["tables"] if t["name"] == "_unresolved_blocks"]
    if unresolved:
        u = unresolved[0]
        print(f"\nUnresolved blocks: {u['row_count']}")
        # count by status
        from collections import Counter
        import sqlite3
        with sqlite3.connect(ext_result.sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM _unresolved_blocks GROUP BY status ORDER BY cnt DESC"
            ).fetchall()
            for row in rows:
                print(f"  {row['status']}: {row['cnt']}")

    print(f"\nSQLite: {ext_result.sqlite_path}")


if __name__ == "__main__":
    main()
