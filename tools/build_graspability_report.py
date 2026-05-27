#!/usr/bin/env python3
"""Build graspability CSV + HTML report from the GraspGen survey results.

Reads the per-object ``_success`` / ``_fail`` directories produced by
``render_grasps`` and the run log, updates the graspability CSV,
and generates a standalone HTML dashboard.

Usage::

    python tools/build_graspability_report.py \
        --survey-dir outputs/grasp_datasets/graspgen_full \
        --csv maniguard/task_generation/utils/franka_graspability_full.csv \
        --output-dir outputs/grasp_datasets/survey
"""
from __future__ import annotations

import argparse
import csv
import html
import re
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--survey-dir", type=Path,
                   default=Path("outputs/grasp_datasets/graspgen_full"))
    p.add_argument("--csv", type=Path,
                   default=Path("maniguard/task_generation/utils/franka_graspability_full.csv"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("docs"))
    return p.parse_args()


def _parse_log(log_path: Path) -> dict[str, dict]:
    """Parse run.log to extract per-object timing and grasp counts."""
    results = {}
    current_obj = None
    n_cand = 0
    n_held = 0

    for line in log_path.read_text().splitlines():
        # Match: [HH:MM:SS] (N/M) category/model
        m = re.match(r'\[[\d:]+\]\s+\(\d+/\d+\)\s+(\S+)/(\S+)', line)
        if m:
            current_obj = (m.group(1), m.group(2))
            n_cand = 0
            n_held = 0
            continue

        if current_obj is None:
            continue

        # GraspGen: 200 grasps, score range=...
        m = re.search(r'GraspGen:\s+(\d+)\s+grasps', line)
        if m:
            n_cand = int(m.group(1))

        # -> N valid grasps  (Xs)
        m = re.search(r'->\s+(\d+)\s+valid grasps\s+\(([\d.]+)s\)', line)
        if m:
            n_held = int(m.group(1))
            elapsed = float(m.group(2))
            results[current_obj] = {
                "n_cand": n_cand, "n_held": n_held,
                "elapsed_s": elapsed, "status": "graspable",
            }
            current_obj = None

        # -> FAILED  (Xs)
        m = re.search(r'->\s+FAILED\s+\(([\d.]+)s\)', line)
        if m:
            elapsed = float(m.group(1))
            results[current_obj] = {
                "n_cand": n_cand, "n_held": 0,
                "elapsed_s": elapsed, "status": "no_grasp",
            }
            current_obj = None

    return results


def _scan_directories(
    survey_dir: Path, known_pairs: set[tuple[str, str]],
) -> dict[tuple[str, str], str]:
    """Scan _success/_fail directories to get definitive results.

    Uses ``known_pairs`` (from the CSV) to correctly split directory
    names like ``bag_of_chips_qstxpj_success`` where the category
    itself contains underscores.
    """
    # Build a lookup: stem -> (category, model)
    stem_to_pair = {f"{c}_{m}": (c, m) for c, m in known_pairs}

    results = {}
    for d in survey_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if name.endswith("_success"):
            stem = name[:-len("_success")]
            if stem in stem_to_pair:
                results[stem_to_pair[stem]] = "graspable"
        elif name.endswith("_fail"):
            stem = name[:-len("_fail")]
            if stem in stem_to_pair:
                results[stem_to_pair[stem]] = "no_grasp"
    return results


def _update_csv(csv_path: Path, dir_results: dict, log_results: dict):
    """Update franka_graspability_full.csv with survey results."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            key = (row["category"], row["model"])
            if key in dir_results:
                row["status"] = dir_results[key]
                if key in log_results:
                    lr = log_results[key]
                    row["n_cand"] = str(lr["n_cand"])
                    row["elapsed_s"] = f"{lr['elapsed_s']:.1f}"
                    row["n_tried"] = str(lr.get("n_held", 0))
            rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


TASK_COLS = [
    "clutter_target", "stack_flat_target", "liquid_target",
    "food_transfer_target", "table_obstacle", "wide_opening_container",
]

# Structural / environmental categories that are never manipulable objects.
BLACKLIST_CATEGORIES = {"ceilings", "floors", "walls"}


def _load_existing_classifications(csv_path: Path) -> dict[tuple[str, str], dict]:
    """Load task-classification columns from an existing classified CSV."""
    classifications: dict[tuple[str, str], dict] = {}
    if not csv_path.exists():
        return classifications
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return classifications
        present = [c for c in TASK_COLS if c in reader.fieldnames]
        if not present:
            return classifications
        for row in reader:
            key = (row["category"], row["model"])
            classifications[key] = {c: row.get(c, "not_classified") for c in TASK_COLS}
    return classifications


def _build_classified_csv(rows: list[dict], output_path: Path):
    """Write graspability_classified.csv, preserving task-classification columns.

    Rows whose category is in BLACKLIST_CATEGORIES are excluded.
    """
    existing = _load_existing_classifications(output_path)

    fieldnames = [
        "category", "model", "status", "n_candidates", "n_held",
        "elapsed_s", "note",
    ] + TASK_COLS
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if row["category"] in BLACKLIST_CATEGORIES:
                continue
            key = (row["category"], row["model"])
            cls = existing.get(key, {c: "not_classified" for c in TASK_COLS})
            writer.writerow({
                "category": row["category"],
                "model": row["model"],
                "status": row["status"],
                "n_candidates": row.get("n_cand", "0"),
                "n_held": row.get("n_tried", "0"),
                "elapsed_s": row.get("elapsed_s", "0.0"),
                "note": row.get("note", ""),
                **cls,
            })


def _build_html(classified_csv: Path, output_path: Path):
    """Generate standalone HTML dashboard from the classified CSV."""
    with open(classified_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    col_labels = {
        "clutter_target": "Clutter Target",
        "stack_flat_target": "Stack Flat Target",
        "liquid_target": "Liquid Target",
        "food_transfer_target": "Food Transfer",
        "table_obstacle": "Table Obstacle",
        "wide_opening_container": "Wide Opening Container",
    }
    rating_colors = {
        "perfect": "#22c55e", "possible": "#f59e0b",
        "not_suitable": "#94a3b8", "not_classified": "#a78bfa",
    }
    rating_values = ["perfect", "possible", "not_suitable", "not_classified"]

    # Category-level summary stats
    seen: set[str] = set()
    cat_stats = {col: {v: 0 for v in rating_values} for col in TASK_COLS}
    for r in rows:
        cat = r["category"]
        if cat in seen:
            continue
        seen.add(cat)
        for col in TASK_COLS:
            val = r.get(col, "not_classified")
            cat_stats[col][val] = cat_stats[col].get(val, 0) + 1

    n_cats = len(seen)
    parts: list[str] = []
    parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Graspability Classification</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f172a; color: #e2e8f0; padding: 24px; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 8px; color: #f1f5f9; }}
  .subtitle {{ color: #94a3b8; margin-bottom: 20px; font-size: 0.9rem; }}
  .summary {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 24px; }}
  .card {{ background: #1e293b; border-radius: 8px; padding: 14px 18px; min-width: 170px; flex: 1; }}
  .card h3 {{ font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em;
              color: #94a3b8; margin-bottom: 8px; }}
  .card .nums {{ display: flex; gap: 10px; font-size: 0.85rem; flex-wrap: wrap; }}
  .card .nums span {{ display: inline-flex; align-items: center; gap: 4px; }}
  .dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
  .filters {{ background: #1e293b; border-radius: 8px; padding: 16px 20px;
             margin-bottom: 20px; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }}
  .filters label {{ font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;
                   letter-spacing: 0.04em; }}
  .filters select, .filters input {{
    background: #0f172a; color: #e2e8f0; border: 1px solid #334155;
    border-radius: 6px; padding: 6px 10px; font-size: 0.85rem; }}
  .filters input {{ width: 220px; }}
  .filter-group {{ display: flex; flex-direction: column; gap: 4px; }}
  .table-wrap {{ overflow-x: auto; border-radius: 8px; border: 1px solid #1e293b; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  thead {{ position: sticky; top: 0; z-index: 2; }}
  th {{ background: #1e293b; color: #94a3b8; text-align: left; padding: 10px 12px;
       font-weight: 600; text-transform: uppercase; font-size: 0.72rem;
       letter-spacing: 0.05em; cursor: pointer; user-select: none;
       border-bottom: 2px solid #334155; white-space: nowrap; }}
  th:hover {{ color: #e2e8f0; }}
  th .arrow {{ font-size: 0.6rem; margin-left: 4px; opacity: 0.5; }}
  th.sorted .arrow {{ opacity: 1; color: #38bdf8; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #1e293b; white-space: nowrap; }}
  tr {{ background: #0f172a; }}
  tr:hover {{ background: #1e293b; }}
  .pill {{ display: inline-block; padding: 2px 10px; border-radius: 9999px;
          font-size: 0.75rem; font-weight: 600; }}
  .pill-perfect {{ background: #16532820; color: #4ade80; border: 1px solid #16532840; }}
  .pill-possible {{ background: #78350f20; color: #fbbf24; border: 1px solid #78350f40; }}
  .pill-not_suitable {{ background: #33415520; color: #94a3b8; border: 1px solid #33415540; }}
  .pill-not_classified {{ background: #4c1d9520; color: #a78bfa; border: 1px solid #4c1d9540; }}
  .status-graspable {{ color: #4ade80; }}
  .status-no_grasp {{ color: #f87171; }}
  .status-no_candidates {{ color: #64748b; }}
  .status-too_large {{ color: #64748b; }}
  .row-count {{ color: #64748b; font-size: 0.8rem; margin-top: 12px; }}
</style>
</head>
<body>
<h1>Graspability Classification Survey</h1>
<p class="subtitle">{len(rows)} models &middot; {n_cats} unique categories &middot; 6 task dimensions</p>
""")

    # Summary cards for each task column
    parts.append('<div class="summary">')
    for col in TASK_COLS:
        s = cat_stats[col]
        parts.append(f'<div class="card"><h3>{col_labels[col]}</h3><div class="nums">')
        for val in rating_values:
            if s.get(val, 0) > 0:
                parts.append(
                    f'<span><span class="dot" style="background:{rating_colors[val]}">'
                    f'</span> {s[val]}</span>'
                )
        parts.append('</div></div>')
    parts.append('</div>')

    # Filters
    parts.append("""<div class="filters">
  <div class="filter-group"><label>Search</label>
    <input type="text" id="searchBox" placeholder="Filter by category..." oninput="applyFilters()">
  </div>
  <div class="filter-group"><label>Status</label>
    <select id="filterStatus" onchange="applyFilters()">
      <option value="">All</option>
      <option value="graspable">Graspable</option>
      <option value="no_grasp">No Grasp</option>
      <option value="too_large">Too Large</option>
    </select>
  </div>""")
    for col in TASK_COLS:
        cid = col.replace("_", "")
        parts.append(f"""  <div class="filter-group"><label>{col_labels[col]}</label>
    <select id="filter_{cid}" onchange="applyFilters()">
      <option value="">All</option>
      <option value="perfect">Perfect</option>
      <option value="possible">Possible</option>
      <option value="not_suitable">Not Suitable</option>
      <option value="not_classified">Not Classified</option>
    </select></div>""")
    parts.append('</div>')

    # Table
    headers = ["category", "model", "status", "n_candidates", "n_held", "elapsed_s"] + TASK_COLS
    header_labels = {
        "category": "Category", "model": "Model", "status": "Status",
        "n_candidates": "Candidates", "n_held": "Held", "elapsed_s": "Time (s)",
        **col_labels,
    }
    parts.append('<div class="table-wrap"><table id="mainTable"><thead><tr>')
    for i, h in enumerate(headers):
        parts.append(
            f'<th onclick="sortTable({i})" data-col="{i}">'
            f'{header_labels.get(h, h)} <span class="arrow">&#9650;</span></th>'
        )
    parts.append('</tr></thead><tbody id="tableBody">')

    for r in rows:
        parts.append('<tr>')
        parts.append(f'<td>{html.escape(r["category"])}</td>')
        parts.append(f'<td style="font-family:monospace;color:#64748b">{html.escape(r["model"])}</td>')
        status = r["status"]
        parts.append(f'<td class="status-{status}">{status}</td>')
        parts.append(f'<td style="text-align:right">{r.get("n_candidates", "0")}</td>')
        parts.append(f'<td style="text-align:right">{r.get("n_held", "0")}</td>')
        parts.append(f'<td style="text-align:right">{r.get("elapsed_s", "0.0")}</td>')
        for col in TASK_COLS:
            val = r.get(col, "not_classified")
            parts.append(f'<td><span class="pill pill-{val}">{val.replace("_", " ")}</span></td>')
        parts.append('</tr>')

    parts.append('</tbody></table></div>')
    parts.append('<p class="row-count" id="rowCount"></p>')

    n_rating = len(TASK_COLS)
    js_ids = ",".join(f'"{c.replace("_","")}"' for c in TASK_COLS)
    parts.append(f"""
<script>
const RATING_COLS_IDS = [{js_ids}];
const tbody = document.getElementById('tableBody');
const allRows = Array.from(tbody.rows);
const rowCountEl = document.getElementById('rowCount');

function applyFilters() {{
  const search = document.getElementById('searchBox').value.toLowerCase();
  const status = document.getElementById('filterStatus').value;
  const ratingFilters = RATING_COLS_IDS.map(id => document.getElementById('filter_' + id).value);
  let visible = 0;
  allRows.forEach(row => {{
    const cells = row.cells;
    const cat = cells[0].textContent.toLowerCase();
    const st = cells[2].textContent;
    let show = true;
    if (search && !cat.includes(search)) show = false;
    if (status && st !== status) show = false;
    if (show) {{
      for (let i = 0; i < ratingFilters.length; i++) {{
        if (ratingFilters[i]) {{
          const val = cells[6 + i].textContent.trim().replace(/ /g, '_');
          if (val !== ratingFilters[i]) {{ show = false; break; }}
        }}
      }}
    }}
    row.style.display = show ? '' : 'none';
    if (show) visible++;
  }});
  rowCountEl.textContent = `Showing ${{visible}} of ${{allRows.length}} rows`;
}}

let sortCol = -1, sortAsc = true;
function sortTable(col) {{
  if (sortCol === col) {{ sortAsc = !sortAsc; }}
  else {{ sortCol = col; sortAsc = true; }}
  document.querySelectorAll('th').forEach(th => th.classList.remove('sorted'));
  const th = document.querySelector(`th[data-col="${{col}}"]`);
  th.classList.add('sorted');
  th.querySelector('.arrow').innerHTML = sortAsc ? '&#9650;' : '&#9660;';
  const numCols = new Set([3, 4, 5]);
  allRows.sort((a, b) => {{
    let va = a.cells[col].textContent.trim();
    let vb = b.cells[col].textContent.trim();
    if (numCols.has(col)) {{
      va = parseFloat(va) || 0; vb = parseFloat(vb) || 0;
    }} else {{
      va = va.toLowerCase(); vb = vb.toLowerCase();
    }}
    if (va < vb) return sortAsc ? -1 : 1;
    if (va > vb) return sortAsc ? 1 : -1;
    return 0;
  }});
  allRows.forEach(row => tbody.appendChild(row));
}}
applyFilters();
</script>
</body></html>""")

    output_path.write_text("".join(parts))


def main():
    args = parse_args()
    survey_dir = args.survey_dir.resolve()
    csv_path = args.csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load known (category, model) pairs from CSV for correct name parsing.
    known_pairs = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            known_pairs.add((row["category"], row["model"]))
    print(f"Loaded {len(known_pairs)} pairs from CSV", flush=True)

    print(f"Scanning {survey_dir} ...", flush=True)
    dir_results = _scan_directories(survey_dir, known_pairs)
    print(f"  {len(dir_results)} objects from directories "
          f"({sum(1 for v in dir_results.values() if v == 'graspable')} graspable, "
          f"{sum(1 for v in dir_results.values() if v == 'no_grasp')} no_grasp)")

    # Parse all run logs
    log_results = {}
    for log_file in sorted(survey_dir.glob("run*.log")):
        lr = _parse_log(log_file)
        log_results.update(lr)
        print(f"  {log_file.name}: {len(lr)} entries")

    # Update the full CSV
    print(f"Updating {csv_path} ...", flush=True)
    rows = _update_csv(csv_path, dir_results, log_results)
    n_graspable = sum(1 for r in rows if r["status"] == "graspable")
    n_no_grasp = sum(1 for r in rows if r["status"] == "no_grasp")
    print(f"  {n_graspable} graspable, {n_no_grasp} no_grasp, "
          f"{len(rows)} total rows")

    # Build classified CSV
    classified_csv = output_dir / "graspability_classified.csv"
    _build_classified_csv(rows, classified_csv)
    print(f"Wrote {classified_csv}")

    # Build HTML (reads from classified CSV to include task columns)
    html_path = output_dir / "graspability_classified.html"
    _build_html(classified_csv, html_path)
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
