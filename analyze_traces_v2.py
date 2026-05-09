import json
import glob

for path in glob.glob('artifacts/runs/20260509T074647Z/*/task_86/trace.json'):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
        ans = data.get("answer", {})
        print(f"\n--- {path} ---")
        if ans:
            print(f"Columns: {ans.get('columns')}")
            print(f"Rows count: {len(ans.get('rows', []))}")
        else:
            print("No answer found or failed.")
            
        tool_counts = {}
        for s in data.get('steps', []):
            for c in s.get('tool_calls', []):
                tool_counts[c['name']] = tool_counts.get(c['name'], 0) + 1
        print(f"Tools used: {tool_counts}")
        
        print("Python executions:")
        for s in data.get('steps', []):
            for c in s.get('tool_calls', []):
                if c['name'] == 'execute_python':
                    code = c['args'].get('code', '')
                    lines = code.split("\n")
                    important_lines = [l for l in lines if 'round' in l.lower() or 'position' in l.lower() or 'track number' in l.lower()]
                    if important_lines:
                         print("  > " + " ".join(important_lines[:2]))
