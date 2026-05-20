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
CONFIG_PATH = Path("configs/easy.yaml")


def _build_memagent_rule_provider(model):
    import json
    import re
    from typing import Any

    def rule_provider(params: dict[str, Any]) -> dict[str, Any] | None:
        kind = params.get("kind")
        goal = params.get("goal", "")
        schema = params.get("schema", {})

        if kind == "initial":
            sample_blocks = params.get("sample_blocks", [])
            sample_text = "\n\n".join([f"--- Block {b['block_id']} ---\n{b['text']}" for b in sample_blocks])

            prompt = f"""You are a data engineering assistant. Your task is to analyze the provided text samples and generate a "Rule Pack" in JSON format to extract structured entities.
The goal is: {goal}
The base schema is: {json.dumps(schema)}

Here are some sample blocks from the document:
{sample_text}

Analyze the pattern in the blocks and output a JSON dictionary. The JSON must match the following format:
{{
  "table": "superhero",
  "columns": ["id", "superhero_name", "full_name", "height_cm", "weight_kg", "publisher_id"],
  "primary_key": ["id"],
  "key_patterns": [
    // Regular expressions to capture the ID. Each regex MUST contain EXACTLY ONE capturing group, e.g. "registration number (\\\\d+)"
  ],
  "field_patterns": {{
    "superhero_name": [
      // Regular expressions to capture the superhero name with EXACTLY ONE capturing group, e.g. "known as ([A-Za-z0-9 ]+)"
    ],
    "full_name": [
      // Regular expressions to capture the full civilian name with EXACTLY ONE capturing group, e.g. "civilian name is ([A-Za-z0-9 ]+)"
    ],
    "height_cm": [
      // Regular expressions to capture height with EXACTLY ONE capturing group, e.g. "height is (\\\\d+) cm" or "(\\\\d+) centimeters tall"
    ],
    "weight_kg": [
      // Regular expressions to capture weight with EXACTLY ONE capturing group, e.g. "weight is (\\\\d+) kg" or "(\\\\d+) kilograms"
    ],
    "publisher_id": [
      // Regular expressions to capture publisher ID with EXACTLY ONE capturing group, e.g. "publisher affiliation is (\\\\d+)" or "registered with publisher (\\\\d+)"
    ]
  }},
  "field_types": {{
    "height_cm": ["int"],
    "weight_kg": ["int"],
    "publisher_id": ["int"]
  }},
  "null_values": ["None", "NaN", "-", ""]
}}

CRITICAL RULES:
1. Every regular expression in `key_patterns` and `field_patterns` MUST have EXACTLY ONE capturing group `(...)` representing the extracted value.
2. In JSON strings, remember to escape backslashes in regex, e.g., use `\\\\d` instead of `\\d` or `\d`.
3. Do not include any text, explanations, or commentary in your response. Output ONLY the valid JSON block inside markdown fence ```json ... ```.
"""
        elif kind == "repair":
            current_rule_pack = params.get("current_rule_pack", {})
            diagnostics = params.get("diagnostics", {})
            residual_blocks = params.get("residual_blocks", [])

            residual_text = "\n\n".join([f"--- Unresolved Block {b['block_id']} ({b['status']}: {b['reason']}) ---\n{b['text']}" for b in residual_blocks])

            prompt = f"""You are a data engineering assistant. Your task is to diagnose and generate a repair "Patch" in JSON format for the extraction rule pack.
The goal is: {goal}
Current rule pack: {json.dumps(current_rule_pack)}
Diagnostics of failed extraction: {json.dumps(diagnostics)}

Here are some residual (unresolved/failed) blocks that current rules could not fully extract:
{residual_text}

Provide additional regular expression patterns to repair the extraction. Output a JSON patch dictionary. The JSON must match the following format:
{{
  "key_patterns": [
    // ANY NEW regular expressions to capture the ID from the failed blocks
  ],
  "field_patterns": {{
    "superhero_name": [
      // ANY NEW regular expressions to capture superhero name
    ],
    "full_name": [
      // ANY NEW regular expressions to capture civilian name
    ],
    "height_cm": [
      // ANY NEW regular expressions to capture height
    ],
    "weight_kg": [
      // ANY NEW regular expressions to capture weight
    ],
    "publisher_id": [
      // ANY NEW regular expressions to capture publisher ID
    ]
  }},
  "field_types": {{
    "height_cm": ["int"],
    "weight_kg": ["int"],
    "publisher_id": ["int"]
  }}
}}

CRITICAL RULES:
1. Every regular expression in `key_patterns` and `field_patterns` MUST have EXACTLY ONE capturing group `(...)` representing the extracted value.
2. In JSON strings, remember to escape backslashes in regex, e.g., use `\\\\d` instead of `\\d` or `\d`.
3. Only add NEW patterns that are missing in the current rule pack. Do not duplicate existing ones.
4. Do not include any text, explanations, or commentary in your response. Output ONLY the valid JSON block inside markdown fence ```json ... ```.
"""
        else:
            return None

        response = model.invoke(prompt)
        text = response.content
        
        # Extremely robust JSON extraction using first and last curly braces
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = text[first_brace:last_brace + 1]
            try:
                return json.loads(json_str)
            except Exception as e:
                print(f"JSON parsing error from bracket substring: {e}")
                # Fallback to try cleaning comments if any exist in model's JSON
                try:
                    # Remove single-line comments in JSON if any
                    cleaned = re.sub(r"^\s*//.*$", "", json_str, flags=re.MULTILINE)
                    return json.loads(cleaned)
                except Exception as ex:
                    print(f"JSON parsing error after cleaning comments: {ex}")
        
        # Fallback to regex if braces method failed
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except Exception as e:
                print(f"JSON parsing error from fence fallback: {e}")
        try:
            return json.loads(text)
        except Exception as e:
            print(f"JSON parsing error from raw fallback: {e}")
            return None

    return rule_provider



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
