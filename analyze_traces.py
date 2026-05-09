import json
import glob

for path in glob.glob('artifacts/runs/20260509T074146Z/*/task_86/trace.json'):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
        ans = data.get("answer", {})
        print(f"--- {path} ---")
        if ans:
            print(f"Columns: {ans.get('columns')}")
            print(f"Rows count: {len(ans.get('rows', []))}")
        else:
            print("No answer found or failed.")
            
        for s in data.get('steps', []):
            for c in s.get('tool_calls', []):
                if c['name'] in ('execute_python', 'execute_context_sql', 'read_csv', 'read_json'):
                    print(f"Tool: {c['name']}")
                    if c['name'] == 'execute_python':
                        code = c['args'].get('code', '')
                        print("Code snippet (first 10 lines):")
                        print("\n".join(code.split("\n")[:10]))
