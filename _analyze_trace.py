import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('artifacts/runs/20260623T125002Z/task_39/trace.json', encoding='utf-8') as f:
    trace = json.load(f)

print(f"Total steps: {len(trace['steps'])}")
print(f"Final answer: {json.dumps(trace.get('answer', {}), ensure_ascii=False, indent=2)[:1000]}")
print()

for step in trace['steps']:
    node = step['node']
    si = step['step_index']
    msg = step.get('assistant_message') or ''
    tc_names = [tc['name'] for tc in step.get('tool_calls', [])]

    print(f"\n{'='*60}")
    print(f"STEP {si}: {node} | tools={tc_names}")
    print(f"{'='*60}")

    if node == 'model' and msg:
        # Print first 500 and last 500 chars
        print(f"MSG ({len(msg)} chars):")
        print(msg[:500])
        if len(msg) > 1000:
            print(f"... [{len(msg) - 1000} chars omitted] ...")
            print(msg[-500:])

    if node == 'validate_process':
        print(f"PROCESS VALIDATOR ({len(msg)} chars):")
        print(msg[:2000])

    if node == 'validate_answer':
        print(f"ANSWER VALIDATOR ({len(msg)} chars):")
        print(msg[:2000])

    if node == 'tool':
        for tr in step.get('tool_results', []):
            content = tr.get('content', '')
            ok = tr.get('ok', None)
            if isinstance(content, dict):
                if 'row_count' in content:
                    print(f"  TOOL RESULT: ok={ok}, row_count={content['row_count']}")
                if 'columns' in content:
                    print(f"  columns: {content['columns']}")
                if 'error' in content:
                    print(f"  error: {str(content['error'])[:300]}")
                if 'rows' in content:
                    rows = content['rows']
                    print(f"  rows count: {len(rows)}")
                    if rows:
                        print(f"  first 3: {rows[:3]}")
            elif isinstance(content, str) and len(content) > 5:
                print(f"  TOOL RESULT ({len(content)} chars): {content[:300]}")
