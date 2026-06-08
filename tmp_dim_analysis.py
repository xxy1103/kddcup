"""Analyze which dimension tables can be safely removed from query_surfaces."""
import json
from pathlib import Path

PROJECT = Path(r"C:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit")
TEST_DIR = PROJECT / "artifacts" / "test"

total_tables = 0
total_enriched = 0
total_dimension_only = 0
total_standalone = 0
total_removable = 0
per_task_results = []

for td in sorted(TEST_DIR.iterdir(), key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else 0):
    if not td.is_dir():
        continue
    gp = json.loads((td / "global_data_profile.json").read_text(encoding="utf-8"))
    sc = json.loads((td / "semantic_catalog.json").read_text(encoding="utf-8"))

    surfaces = gp.get("query_surfaces", [])
    views = sc.get("derived_views", [])

    # Collect all table names from query_surfaces
    all_surface_tables = {s["table"] for s in surfaces}

    # Classify each table
    base_tables = set()       # tables that have enriched views
    dim_tables = set()        # tables that appear as attached dimensions
    dim_to_bases = {}         # dimension -> set of base tables it's joined into

    for s in surfaces:
        if s.get("kind") == "derived_view":
            base_tables.add(s["table"])
            for dim in s.get("attached_dimensions", []):
                dim_name = dim.get("table", "")
                if dim_name:
                    dim_tables.add(dim_name)
                    dim_to_bases.setdefault(dim_name, set()).add(s["base_table"])

    # Also get the base_table names (original names before v_ prefix)
    base_original_names = set()
    for s in surfaces:
        if s.get("kind") == "derived_view":
            base_original_names.add(s.get("base_table", ""))

    # Now classify every query surface
    removable = []
    kept = []
    for s in surfaces:
        tname = s["table"]
        base_name = s.get("base_table", tname)

        if s.get("kind") == "derived_view":
            # This is an enriched view - keep it
            kept.append(tname)
        elif tname in dim_tables and tname not in base_original_names:
            # This table is ONLY a dimension (never a base table itself)
            # Check: are ALL the base tables it joins into also enriched?
            joined_bases = dim_to_bases.get(tname, set())
            # Can be removed if it's absorbed into at least one enriched view
            if joined_bases:
                removable.append({
                    "table": tname,
                    "absorbed_into": sorted(joined_bases),
                })
            else:
                kept.append(tname)
        else:
            # Standalone original table - keep it
            kept.append(tname)

    n_tables = len(surfaces)
    n_removable = len(removable)
    n_enriched = len([s for s in surfaces if s.get("kind") == "derived_view"])
    n_kept = len(kept)

    total_tables += n_tables
    total_enriched += n_enriched
    total_removable += n_removable
    total_standalone += n_kept - n_enriched
    total_dimension_only += n_removable

    if n_removable > 0 or n_enriched > 0:
        per_task_results.append({
            "task_id": td.name,
            "surfaces": n_tables,
            "enriched": n_enriched,
            "removable_dims": n_removable,
            "reduced_surfaces": n_tables - n_removable,
            "reduction_pct": round(n_removable / n_tables * 100, 1) if n_tables > 0 else 0,
            "removable_tables": removable,
        })

# Print summary
print("=" * 70)
print("维度表移除可行性分析")
print("=" * 70)
print()
print(f"全部 60 题统计:")
print(f"  query_surfaces 总数:       {total_tables}")
print(f"  enriched views:            {total_enriched}")
print(f"  可移除的纯维度表:          {total_removable}")
print(f"  不可移除的表:              {total_tables - total_removable}")
print(f"  移除后 surfaces 减少:      {total_removable}/{total_tables} = {total_removable/total_tables*100:.1f}%")
print()

print(f"有可移除维度表的题目 ({len(per_task_results)} 题):")
print(f"{'task_id':>10s}  {'surf':>5s}  {'enr':>4s}  {'rem':>4s}  {'new':>5s}  {'reduc%':>6s}  removable_tables")
print("-" * 90)
for r in per_task_results:
    names = ", ".join(d["table"] for d in r["removable_tables"][:3])
    if len(r["removable_tables"]) > 3:
        names += f" +{len(r['removable_tables'])-3}"
    print(f"{r['task_id']:>10s}  {r['surfaces']:>5d}  {r['enriched']:>4d}  "
          f"{r['removable_dims']:>4d}  {r['reduced_surfaces']:>5d}  "
          f"{r['reduction_pct']:>5.1f}%  {names}")

# Save detailed report
report_path = TEST_DIR / "_dimension_removal_analysis.json"
report_path.write_text(json.dumps({
    "summary": {
        "total_surfaces": total_tables,
        "enriched_views": total_enriched,
        "removable_dimension_tables": total_removable,
        "surfaces_after_removal": total_tables - total_removable,
        "reduction_percentage": f"{total_removable/total_tables*100:.1f}%",
    },
    "per_task": per_task_results,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n详细报告: {report_path}")
