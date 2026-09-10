"""Human-readable reports built from the same rows exported for custom analysis."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

from halt.evaluation.analysis import (
    BOOL_FIELDS,
    JSON_FIELDS,
    RESULT_FIELDS,
    compare,
    measurement_row,
    paired_bootstrap,
)

__all__ = ["paired_bootstrap", "summarize", "write_reports", "render_comparison"]


def summarize(records: list[dict[str, Any]], *, baseline: str, bootstrap_samples: int = 2000,
              seed: int = 0) -> dict[str, Any]:
    """Backward-compatible entry point for the shared public comparison API."""
    return compare(records, baseline=baseline, bootstrap_samples=bootstrap_samples, seed=seed)


def _number(value: float | None, *, percent: bool = False, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    if percent:
        return f"{value * 100:+.1f}" if signed else f"{value * 100:.1f}%"
    return f"{value:.3f}"


def render_comparison(summary: dict[str, Any]) -> str:
    """Return a concise terminal table without printing or importing a display package."""
    table = [["Method", "N", "Accuracy", "Delta (pp)", "Tokens/q", "Seconds/q", "Not completed"]]
    for row in summary["methods"]:
        table.append([row["method_id"], str(row["sample_count"]), _number(row["accuracy"], percent=True),
            _number(row["paired_accuracy_difference"], percent=True, signed=True),
            f"{row['mean_total_generated_tokens']:.1f}", _number(row["mean_latency_seconds"]),
            str(row["failure_count"])])
    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    lines = [summary["interpretation"], "", f"Baseline: {summary['baseline']}"]
    for index, row in enumerate(table):
        lines.append("  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)))
        if index == 0:
            lines.append("  ".join("-" * width for width in widths))
    lines.append("Tokens include reasoning + answers + generated probes. Scoring/input work is separate in the reports.")
    lines.append("Delta pairs identical question IDs and seeds; not completed includes errors, abstentions, and incomplete answers.")
    return "\n".join(lines)


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _table(headers: list[str], rows: list[list[Any]], *, element_id: str = "") -> str:
    heading = "".join(f'<th scope="col"><button type="button" data-column="{i}" title="Sort by {_escape(name)}">{_escape(name)}</button></th>' for i, name in enumerate(headers))
    body = "".join("<tr>" + "".join(f"<td>{_escape(cell)}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table id="{_escape(element_id)}"><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table></div>'


def _scatter(summary: dict[str, Any]) -> str:
    rows = [r for r in summary["methods"] if r["mean_latency_seconds"] is not None]
    if not rows:
        return "<p>No measured durations available.</p>"
    maximum = max(r["mean_latency_seconds"] for r in rows) or 1.0
    pieces = ['<svg viewBox="0 0 780 300" role="img" aria-labelledby="scatter-title scatter-desc">',
        '<title id="scatter-title">Accuracy versus average runtime</title>',
        '<desc id="scatter-desc">Higher accuracy and lower runtime are preferable. This plot shows point estimates; consult paired intervals and timing scope.</desc>',
        '<path d="M65 25V245H580" fill="none" stroke="currentColor"/>',
        '<text x="320" y="286" text-anchor="middle">Mean seconds per question</text>',
        '<text transform="translate(16 140) rotate(-90)" text-anchor="middle">Accuracy</text>']
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = 245 - fraction * 210
        x = 65 + fraction * 500
        pieces.append(f'<text x="55" y="{y + 4}" text-anchor="end">{fraction:.0%}</text><path d="M65 {y}H580" stroke="#d8e0e8"/>')
        pieces.append(f'<text x="{x}" y="263" text-anchor="middle">{maximum * fraction:.2f}</text>')
    colors = ("#0f766e", "#1d4ed8", "#b45309", "#be185d", "#6d28d9", "#334155")
    for index, row in enumerate(rows):
        x = 65 + row["mean_latency_seconds"] / maximum * 500
        y = 245 - row["accuracy"] * 210
        color = colors[index % len(colors)]
        title = _escape(f"{row['method_id']}: {row['accuracy']:.1%}, {row['mean_latency_seconds']:.3f} seconds/question")
        pieces.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{color}"><title>{title}</title></circle>')
        pieces.append(f'<text x="{x + 8:.2f}" y="{y - 7:.2f}" fill="{color}">{index + 1}</text>')
    pieces.append("</svg><ol>" + "".join(f"<li>{_escape(r['method_id'])}</li>" for r in rows) + "</ol>")
    return "".join(pieces)


def _html_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    overview, work = [], []
    for method in summary["methods"]:
        ci = method["paired_accuracy_ci95"]
        interval = f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]" if ci is not None else "unavailable"
        overview.append([method["method_id"], method["sample_count"], f"{method['accuracy']:.1%}",
            _number(method["paired_accuracy_difference"], percent=True, signed=True), interval,
            method["paired_sample_count"], f"{method['mean_total_generated_tokens']:.1f}",
            _number(method["mean_latency_seconds"]), method["failure_count"]])
        work.append([method["method_id"], *[f"{method['mean_' + name]:.1f}" for name in (
            "reasoning_tokens", "answer_tokens", "probe_output_tokens", "input_tokens",
            "recomputed_prefix_tokens", "forced_context_tokens", "scored_tokens", "probe_calls",
            "forward_calls", "embedding_calls")]])
    detail = [[r["question_id"], r["seed"], r["method_id"], r["question"],
        json.dumps(r["choices"], ensure_ascii=False) if r["choices"] else "", r["reference"], r["answer"],
        "yes" if r["correct"] else "no", r["status"], r["stop_reason"], r["stop_position"],
        r["total_generated_tokens"], r["input_tokens"], r["scored_tokens"], r["probe_calls"],
        _number(r["elapsed_seconds"]), r["error"], json.dumps(r["diagnostics"], ensure_ascii=False)] for r in rows]
    experiment = summary["experiment"]
    identity = " · ".join(f"{key}: {value}" for key, value in experiment.items())
    notes = "".join(f"<li><strong>{_escape(m['method_id'])}:</strong> {_escape(m['uncertainty_note'])} Timing: {_escape(m['timing_scope'])}; {m['timing_sample_count']} measured durations. Unmatched method/baseline observations: {m['unpaired_sample_count']}/{m['baseline_unmatched_sample_count']}.</li>" for m in summary["methods"])
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>HALT method comparison</title><style>
:root{color-scheme:light;font:15px/1.5 system-ui,sans-serif;color:#172b3a;background:#f5f7fa}body{max-width:1400px;margin:auto;padding:2rem}h1,h2{line-height:1.2}section{background:white;padding:1.3rem;margin:1.2rem 0;border-radius:10px;border:1px solid #dae2ea}.notice{background:#e9f1fb;padding:1rem;border-left:4px solid #1d4ed8}.metadata{overflow-wrap:anywhere;color:#42566c}a{color:#175ac0}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:.65rem;text-align:left;border-bottom:1px solid #e0e5eb;vertical-align:top}th{background:#f0f4f8;white-space:nowrap}td{max-width:32rem;overflow-wrap:anywhere}.scroll{overflow:auto;max-height:75vh}th button{font:inherit;font-weight:bold;color:inherit;background:none;border:0;cursor:pointer;padding:0}input{font:inherit;padding:.5rem;width:min(90%,34rem)}svg{max-width:850px;width:100%;font:12px system-ui,sans-serif}button:focus-visible,a:focus-visible,input:focus-visible{outline:3px solid #1d4ed8}footer{color:#42566c}
</style></head><body><h1>HALT method comparison</h1>
""" + f'<p class="notice">{_escape(summary["interpretation"])}</p><p class="metadata">{_escape(identity)}<br>Seeds: {_escape(summary["seeds"])} · Baseline: {_escape(summary["baseline"])}</p>' + """
<p>Save or open the underlying files: <a href="results.csv">individual results CSV</a> · <a href="summary.csv">summary CSV</a> · <a href="results.jsonl">detailed JSONL</a> · <a href="manifest.json">experiment settings</a>.</p>
<section><h2>Overview</h2><p>All observed runs are retained, including failures. Tokens include reasoning, final answers, and generated probes. Delta is the paired accuracy change in percentage points relative to the baseline. Click a column heading to sort.</p>
""" + _table(["Method", "N", "Accuracy", "Delta (pp)", "95% CI (pp)", "Paired N", "Tokens/question", "Seconds/question", "Not completed"], overview, element_id="overview") + "<p>Not completed includes errors, abstentions, and incomplete answers.</p><ul>" + notes + "</ul></section><section><h2>Accuracy and runtime</h2>" + _scatter(summary) + """<p>Point estimates only. Simulated and replay timings do not measure model inference. Runtime includes probing and finalization within each run; model loading and report generation are excluded.</p></section><section><h2>Work per question</h2><p>Input tokens count repeated processing across model calls. Recomputed prefix tokens overlap input tokens; scored tokens and forced context describe different work. These columns must not be added together into a single token total.</p>""" + _table(["Method", "Reasoning", "Answer", "Probe output", "Processed input", "Recomputed prefix", "Forced context", "Scored", "Probe calls", "Forward calls", "Embedding calls"], work, element_id="work") + """</section><section><h2>Individual questions</h2><p><label for="search">Filter by question, method, answer, status, or stop reason</label><br><input id="search" type="search" placeholder="Search results"></p><p id="visible-count" aria-live="polite"></p>""" + _table(["Question ID", "Seed", "Method", "Question", "Choices", "Reference", "Answer", "Correct", "Status", "Stop reason", "Stop position", "Generated", "Processed input", "Scored", "Probes", "Seconds", "Error", "Diagnostics"], detail, element_id="questions") + r"""</section><footer>HALT results schema 1.0. This standalone report makes no network requests. The JSONL export includes original manifests, operation ledgers, and method diagnostics. Question text and evaluator reference answers are included in exports; review those contents before sharing.</footer>
<script>
const search = document.getElementById('search');
const table = document.getElementById('questions');
function filterRows(){let shown=0;const query=search.value.toLowerCase();for(const row of table.tBodies[0].rows){row.hidden=!row.textContent.toLowerCase().includes(query);if(!row.hidden)shown++;}document.getElementById('visible-count').textContent=shown+' of '+table.tBodies[0].rows.length+' rows shown';}
search.addEventListener('input',filterRows);filterRows();
for(const button of document.querySelectorAll('th button')){button.addEventListener('click',()=>{const current=button.closest('table');const column=Number(button.dataset.column);const direction=button.dataset.direction==='asc'?-1:1;for(const other of current.querySelectorAll('th button')){delete other.dataset.direction;other.closest('th').removeAttribute('aria-sort');}button.dataset.direction=direction===1?'asc':'desc';button.closest('th').setAttribute('aria-sort',direction===1?'ascending':'descending');const rows=Array.from(current.tBodies[0].rows);rows.sort((a,b)=>{const left=a.cells[column].textContent;const right=b.cells[column].textContent;const numeric=/^[+-]?\d+(?:\.\d+)?%?$/;const value=numeric.test(left)&&numeric.test(right)?parseFloat(left)-parseFloat(right):left.localeCompare(right,undefined,{numeric:true});return direction*value;});for(const row of rows)current.tBodies[0].appendChild(row);});}
</script></body></html>
"""


def write_reports(records: list[dict[str, Any]], output_dir: str | Path, *, baseline: str,
                  bootstrap_samples: int = 2000, seed: int = 0) -> dict[str, Any]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rows = [measurement_row(record) for record in records]
    summary = compare(rows, baseline=baseline, bootstrap_samples=bootstrap_samples, seed=seed)
    (directory / "per_example.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records), encoding="utf-8")
    (directory / "results.jsonl").write_text("".join(json.dumps({**row, "details": record}, allow_nan=False) + "\n" for row, record in zip(rows, records, strict=True)), encoding="utf-8")
    with (directory / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = {**row, **{key: json.dumps(row[key], ensure_ascii=False, allow_nan=False) for key in JSON_FIELDS},
                          **{key: str(row[key]).lower() for key in BOOL_FIELDS}}
            writer.writerow(serialized)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    nested = {"status_counts", "stop_reasons", "stop_positions", "evidence_kind", "recipe", "resources",
              "mean_component_seconds", "paired_accuracy_ci95"}
    percentiles = ["p50_latency_seconds", "p95_latency_seconds"]
    fields = list(dict.fromkeys(key for row in summary["methods"] for key in row
                              if key not in nested and key not in percentiles)) + percentiles
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["methods"])
    lines = ["# HALT evaluation", "", summary["interpretation"], "",
        "| Method | Evidence | N | Accuracy | Paired change (95% CI) | Mean generated tokens | Mean seconds | Not completed |",
        "|---|---|---:|---:|---|---:|---:|---:|"]
    for row in summary["methods"]:
        interval = row["paired_accuracy_ci95"]
        value = row["paired_accuracy_difference"]
        change = f"{value:+.3f}" if value is not None else "n/a"
        change += f" [{interval[0]:+.3f}, {interval[1]:+.3f}]" if interval is not None else " (CI unavailable; see uncertainty_note)"
        method = row["method_id"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {method} | {', '.join(row['evidence_kind'])} | {row['sample_count']} | {row['accuracy']:.3f} | {change} | {row['mean_total_generated_tokens']:.1f} | {_number(row['mean_latency_seconds'])} | {row['failure_count']} |")
    lines.extend(["", "Generated tokens include reasoning, answer, and probe output. Input/scoring work is reported separately. Latency scope and complete work/resource records are in summary.json. Percentiles are omitted below 20 measured durations. Every output is evaluated against an evaluator-only reference label. Resolved configurations and run identities are in manifest.json and results.jsonl. Open report.html to sort and filter per-question results.", ""])
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (directory / "report.html").write_text(_html_report(summary, rows), encoding="utf-8")
    return summary
