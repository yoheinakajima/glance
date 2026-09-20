"""Offline cascade analysis: how much of the frontier's cost can a free local model save?

Question this answers: if the local model answers the items it is confident about and only
escalates the rest to an expensive frontier model, how accurate is the combination, and how
many frontier calls does it save? Everything here is computed from an existing run's
`predictions.jsonl` (already-logged logits/probabilities and, for frontier rows, `correct`).
Nothing here re-runs a model, loads model weights, or talks to a network: numpy + matplotlib
only.

A run directory can still be growing (a separate process appending frontier rows), so reading
tolerates a truncated last line and suites/backends with no (or few) frontier rows yet.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..config import load_config
from ..scorer import confidence, noul_confidence

LOCAL_BACKENDS = ("vlm", "siglip")
TARGET_PRECISIONS = (0.90, 0.95, 0.97, 0.99)
COVERAGE_GRID = [round(i * 0.05, 2) for i in range(21)]  # 0.00, 0.05, ..., 1.00


# --- loading (tolerant of a partial-write tail) --------------------------------------------------


def read_predictions_tolerant(path: Path) -> list[dict[str, Any]]:
    """Read a predictions.jsonl that another process may still be appending to.

    Every line is parsed as JSON except possibly the very last non-blank one: if that one fails
    to parse (a writer flushed mid-line), it is silently skipped. A parse failure anywhere else
    is a real problem and is raised.
    """
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").split("\n")
    while lines and lines[-1] == "":
        lines.pop()
    rows: list[dict[str, Any]] = []
    last_idx = len(lines) - 1
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == last_idx:
                continue
            raise
    return rows


# --- local prediction / confidence, per the project's own scorer -----------------------------------


def local_prediction(row: dict[str, Any]) -> tuple[int, float, bool]:
    """(predicted index, confidence, correct) for one local (vlm/siglip) row."""
    p = row.get("calibrated")
    if p is None:
        p = row["raw"]
    if row["type"] == "noul":
        p = float(p)
        pred = int(p >= 0.5)
        conf = noul_confidence(p)
    else:
        arr = np.asarray(p, dtype=np.float64)
        pred = int(np.argmax(arr))
        conf = confidence(arr)
    correct = pred == row["label_index"]
    return pred, conf, correct


# --- indexing rows --------------------------------------------------------------------------------


def _index_rows(rows: list[dict[str, Any]]):
    """Group rows into local test/calibration rows keyed by (suite, backend) and frontier test rows
    keyed by suite. `method == "letter"` rows are dropped, per spec (only one local row per item)."""
    local_test: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    local_cal: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    frontier: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        if r.get("method") == "letter":
            continue
        backend = r.get("backend")
        suite = r.get("suite")
        if backend == "frontier":
            if r.get("split") == "test":
                frontier[suite][r["item_id"]] = r
            continue
        if backend not in LOCAL_BACKENDS:
            continue
        split = r.get("split")
        if split == "test":
            local_test[(suite, backend)][r["item_id"]] = r
        elif split == "calibration":
            local_cal[(suite, backend)][r["item_id"]] = r
    return local_test, local_cal, frontier


# --- threshold selection (calibration split only, never sees test labels or frontier results) -----


def find_tau(confidences: list[float], corrects: list[bool], target: float) -> float | None:
    """Lowest confidence threshold tau such that accuracy among calibration items with
    confidence >= tau is >= target. None if no threshold reaches it."""
    if not confidences:
        return None
    best: float | None = None
    for tau in sorted(set(confidences)):
        subset = [c for cf, c in zip(confidences, corrects) if cf >= tau]
        if not subset:
            continue
        acc = sum(subset) / len(subset)
        if acc >= target and (best is None or tau < best):
            best = tau
    return best


# --- cascade for one (suite, local backend) --------------------------------------------------------


def cascade_unit(
    test_rows: dict[str, dict[str, Any]],
    cal_rows: dict[str, dict[str, Any]],
    frontier_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """All cascade numbers for one (suite, local backend), joined against frontier rows on item_id."""
    joined_ids = sorted(set(test_rows) & set(frontier_rows))
    n = len(joined_ids)
    if n == 0:
        return {"n_joined": 0, "n_calibration": len(cal_rows)}

    conf: dict[str, float] = {}
    local_correct: dict[str, bool] = {}
    for iid in joined_ids:
        _, c, correct = local_prediction(test_rows[iid])
        conf[iid] = c
        local_correct[iid] = correct
    frontier_correct = {iid: bool(frontier_rows[iid]["correct"]) for iid in joined_ids}

    # Rank by confidence descending; ties broken by the (already alphabetical) order of joined_ids,
    # which `sorted` preserves because Python's sort is stable.
    order = sorted(range(n), key=lambda i: -conf[joined_ids[i]])
    local_ranked = np.array([int(local_correct[joined_ids[i]]) for i in order])
    frontier_ranked = np.array([int(frontier_correct[joined_ids[i]]) for i in order])
    cum_local = np.concatenate([[0], np.cumsum(local_ranked)])
    cum_frontier = np.concatenate([[0], np.cumsum(frontier_ranked)])
    total_frontier = int(cum_frontier[-1])
    total_local = int(cum_local[-1])

    def combined_correct(k: int) -> int:
        # top k (by confidence) answered locally; the rest escalated to the frontier.
        return int(cum_local[k] + (total_frontier - cum_frontier[k]))

    local_acc = total_local / n
    frontier_acc = total_frontier / n
    oracle_acc = sum(1 for iid in joined_ids if local_correct[iid] or frontier_correct[iid]) / n

    curve = [{"coverage": c, "combined_accuracy": combined_correct(round(c * n)) / n} for c in COVERAGE_GRID]

    # Hindsight: largest coverage (by rank, full resolution) whose combined accuracy is still >=
    # frontier-alone. combined_correct(k) - total_frontier == cum_local[k] - cum_frontier[k], both
    # integers, so this is an exact comparison (no floating-point slop).
    best_k = 0
    for k in range(n, -1, -1):
        if cum_local[k] >= cum_frontier[k]:
            best_k = k
            break
    hindsight = {"coverage": best_k / n, "combined_accuracy": combined_correct(best_k) / n}

    cal_conf, cal_correct = [], []
    for row in cal_rows.values():
        _, c, correct = local_prediction(row)
        cal_conf.append(c)
        cal_correct.append(correct)

    operating_points: dict[str, dict[str, Any]] = {}
    for t in TARGET_PRECISIONS:
        key = f"{t:.2f}"
        tau = find_tau(cal_conf, cal_correct, t)
        if tau is None:
            operating_points[key] = {
                "tau": None, "coverage": None, "local_accuracy": None, "combined_accuracy": None,
                "frontier_calls_saved": None, "points_vs_frontier_alone": None,
            }
            continue
        answered = [iid for iid in joined_ids if conf[iid] >= tau]
        escalated = [iid for iid in joined_ids if conf[iid] < tau]
        coverage = len(answered) / n
        local_acc_at_tau = (sum(local_correct[iid] for iid in answered) / len(answered)) if answered else None
        combined = (sum(local_correct[iid] for iid in answered) + sum(frontier_correct[iid] for iid in escalated)) / n
        operating_points[key] = {
            "tau": tau, "coverage": coverage, "local_accuracy": local_acc_at_tau,
            "combined_accuracy": combined, "frontier_calls_saved": coverage,
            "points_vs_frontier_alone": 100.0 * (combined - frontier_acc),
        }

    return {
        "n_joined": n, "n_calibration": len(cal_rows),
        "local_accuracy": local_acc, "frontier_accuracy": frontier_acc, "oracle_accuracy": oracle_acc,
        "curve": curve, "operating_points": operating_points, "hindsight": hindsight,
    }


# --- macro average over suites (vlm only) -----------------------------------------------------------


def macro_average(units: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    valid = {suite: u for suite, u in units.items() if u.get("n_joined", 0) > 0}
    if not valid:
        return None

    def avg(key: str) -> float:
        return float(np.mean([u[key] for u in valid.values()]))

    curve = []
    for i, c in enumerate(COVERAGE_GRID):
        curve.append({"coverage": c, "combined_accuracy": float(np.mean([u["curve"][i]["combined_accuracy"] for u in valid.values()]))})

    operating_points: dict[str, dict[str, Any]] = {}
    any_unit = next(iter(valid.values()))
    for key in any_unit["operating_points"]:
        entries = [u["operating_points"][key] for u in valid.values() if u["operating_points"][key]["tau"] is not None]
        if not entries:
            operating_points[key] = {
                "coverage": None, "local_accuracy": None, "combined_accuracy": None,
                "frontier_calls_saved": None, "points_vs_frontier_alone": None, "n_suites": 0,
            }
            continue
        local_vals = [e["local_accuracy"] for e in entries if e["local_accuracy"] is not None]
        operating_points[key] = {
            "coverage": float(np.mean([e["coverage"] for e in entries])),
            "local_accuracy": float(np.mean(local_vals)) if local_vals else None,
            "combined_accuracy": float(np.mean([e["combined_accuracy"] for e in entries])),
            "frontier_calls_saved": float(np.mean([e["frontier_calls_saved"] for e in entries])),
            "points_vs_frontier_alone": float(np.mean([e["points_vs_frontier_alone"] for e in entries])),
            "n_suites": len(entries),
        }

    return {
        "n_suites": len(valid), "suites": sorted(valid.keys()),
        "n_joined_total": int(sum(u["n_joined"] for u in valid.values())),
        "local_accuracy": avg("local_accuracy"), "frontier_accuracy": avg("frontier_accuracy"),
        "oracle_accuracy": avg("oracle_accuracy"), "curve": curve, "operating_points": operating_points,
        "hindsight": {
            "coverage": float(np.mean([u["hindsight"]["coverage"] for u in valid.values()])),
            "combined_accuracy": float(np.mean([u["hindsight"]["combined_accuracy"] for u in valid.values()])),
        },
    }


# --- top-level analysis ------------------------------------------------------------------------------


def analyze_run(run_dir: Path) -> dict[str, Any]:
    """Read a run directory's predictions.jsonl and compute every cascade number for it."""
    run_dir = Path(run_dir)
    rows = read_predictions_tolerant(run_dir / "predictions.jsonl")
    local_test, local_cal, frontier = _index_rows(rows)

    suites = sorted({s for (s, _b) in local_test} | set(frontier))
    frontier_rows_available = {s: len(frontier.get(s, {})) for s in suites}

    result_suites: dict[str, dict[str, Any]] = {}
    vlm_units: dict[str, dict[str, Any]] = {}
    for suite in suites:
        by_backend: dict[str, Any] = {}
        for backend in LOCAL_BACKENDS:
            test_rows = local_test.get((suite, backend))
            if not test_rows:
                continue
            cal_rows = local_cal.get((suite, backend), {})
            unit = cascade_unit(test_rows, cal_rows, frontier.get(suite, {}))
            by_backend[backend] = unit
            if backend == "vlm" and unit.get("n_joined", 0) > 0:
                vlm_units[suite] = unit
        if by_backend:
            result_suites[suite] = by_backend

    return {
        "run_id": run_dir.name,
        "frontier_rows_available": frontier_rows_available,
        "suites": result_suites,
        "macro_vlm": macro_average(vlm_units),
    }


# --- rendering: markdown -----------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _op(unit: dict[str, Any], t: str, field: str) -> Any:
    return unit.get("operating_points", {}).get(t, {}).get(field)


def render_markdown(result: dict[str, Any]) -> str:
    out = [f"# Cascade analysis · run `{result['run_id']}`", ""]
    out += [
        "This asks: if a free local model (vlm, or siglip where it has rows) answers only the items it is "
        "confident about and escalates the rest to the frontier model, how accurate is the combination and how "
        "many frontier calls does that save? Coverage below is the fraction of items the local model answers on "
        "its own; the rest go to the frontier.",
        "",
    ]
    fr = result["frontier_rows_available"]
    out += ["**Frontier rows available so far (test split), so partial data is obvious:** "
            + (", ".join(f"{s} {n}" for s, n in sorted(fr.items())) if fr else "none"), ""]

    for backend in LOCAL_BACKENDS:
        rows_for_backend = {s: u[backend] for s, u in result["suites"].items() if backend in u}
        if not rows_for_backend:
            continue
        out += [f"## Local backend `{backend}`", ""]
        headers = ["Suite", "n", "Local acc", "Frontier acc", "Oracle",
                   "Cov @0.95", "Combined @0.95", "Δ pts @0.95",
                   "Cov @0.97", "Combined @0.97", "Δ pts @0.97"]
        body = []
        for suite in sorted(rows_for_backend):
            u = rows_for_backend[suite]
            if u.get("n_joined", 0) == 0:
                body.append([suite, 0, "-", "-", "-", "-", "-", "-", "-", "-", "-"])
                continue
            body.append([
                suite, u["n_joined"], _fmt(u["local_accuracy"]), _fmt(u["frontier_accuracy"]), _fmt(u["oracle_accuracy"]),
                _fmt(_op(u, "0.95", "coverage")), _fmt(_op(u, "0.95", "combined_accuracy")), _fmt(_op(u, "0.95", "points_vs_frontier_alone"), 1),
                _fmt(_op(u, "0.97", "coverage")), _fmt(_op(u, "0.97", "combined_accuracy")), _fmt(_op(u, "0.97", "points_vs_frontier_alone"), 1),
            ])
        if backend == "vlm" and result.get("macro_vlm"):
            m = result["macro_vlm"]
            body.append([
                f"**macro ({m['n_suites']} suites)**", m["n_joined_total"], _fmt(m["local_accuracy"]), _fmt(m["frontier_accuracy"]), _fmt(m["oracle_accuracy"]),
                _fmt(_op(m, "0.95", "coverage")), _fmt(_op(m, "0.95", "combined_accuracy")), _fmt(_op(m, "0.95", "points_vs_frontier_alone"), 1),
                _fmt(_op(m, "0.97", "coverage")), _fmt(_op(m, "0.97", "combined_accuracy")), _fmt(_op(m, "0.97", "points_vs_frontier_alone"), 1),
            ])
        out += [_md_table(headers, body), ""]
        out += [
            "\"Cov\" and \"Combined\" are the honest operating points: a confidence threshold picked only from the "
            "local calibration-split rows of that suite (never test labels or frontier results) to reach that "
            "target local precision; \"null\"/`-` means no threshold on the calibration split reached that target. "
            "Δ pts is combined accuracy minus frontier-alone accuracy, in percentage points.",
            "",
        ]

        out += ["Hindsight best coverage (uses test labels; not an operating point you could choose in advance):", ""]
        for suite in sorted(rows_for_backend):
            u = rows_for_backend[suite]
            if u.get("n_joined", 0) == 0:
                out += [f"- `{suite}`: no joined items yet (0 frontier rows)."]
                continue
            h = u["hindsight"]
            out += [f"- `{suite}`: coverage {h['coverage']:.3f} reaches combined accuracy {h['combined_accuracy']:.3f} "
                    f"(frontier alone {u['frontier_accuracy']:.3f})."]
        out += [""]

    return "\n".join(out)


# --- rendering: plots --------------------------------------------------------------------------------


def make_plots(result: dict[str, Any], plots_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..evals.report import GRID, INK, INK_MUTED, SERIES, SURFACE, _titles

    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
        "axes.labelcolor": INK_MUTED, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "grid.linestyle": "-", "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    })

    written: dict[str, str] = {}
    for suite, by_backend in sorted(result["suites"].items()):
        unit = by_backend.get("vlm")
        if not unit or unit.get("n_joined", 0) == 0:
            continue
        stem = f"cascade__{suite.replace('/', '_')}"
        n = unit["n_joined"]
        subtitle = f"{suite} · vlm · test split, n={n} joined with frontier"

        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        xs = [p["coverage"] for p in unit["curve"]]
        ys = [p["combined_accuracy"] for p in unit["curve"]]
        ax.plot(xs, ys, color=SERIES["raw"], linewidth=2, marker="o", markersize=4,
                markeredgecolor=SURFACE, markeredgewidth=0.5, label="combined accuracy", solid_capstyle="round")
        ax.axhline(unit["frontier_accuracy"], color=INK_MUTED, linewidth=2, linestyle="--", label="frontier alone")

        for t, marker in (("0.95", "*"), ("0.97", "D")):
            cov, acc = _op(unit, t, "coverage"), _op(unit, t, "combined_accuracy")
            if cov is not None and acc is not None:
                ax.scatter([cov], [acc], color=SERIES["calibrated"], marker=marker, s=90, zorder=5,
                           edgecolor=SURFACE, linewidth=0.5, label=f"t={t} operating point")

        ax.set_xlim(-0.02, 1.02)
        ylo = min([*ys, unit["frontier_accuracy"]]) - 0.03
        yhi = max([*ys, unit["frontier_accuracy"]]) + 0.03
        ax.set_ylim(max(0.0, ylo), min(1.02, yhi))
        ax.set_xlabel("coverage (fraction answered locally)")
        ax.set_ylabel("combined accuracy")
        _titles(ax, "Escalation curve", subtitle)
        ax.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        path = plots_dir / f"{stem}.png"
        fig.savefig(path, dpi=144)
        plt.close(fig)
        written[suite] = f"plots/{stem}.png"
    return written


# --- CLI ------------------------------------------------------------------------------------------


def _default_run_id(runs_dir: Path) -> str:
    """The run under `runs/` whose predictions.jsonl is currently the largest."""
    candidates = []
    for d in runs_dir.iterdir():
        p = d / "predictions.jsonl"
        if d.is_dir() and p.exists():
            candidates.append((p.stat().st_size, d.name))
    if not candidates:
        raise SystemExit(f"no run with a predictions.jsonl found under {runs_dir}")
    candidates.sort()
    return candidates[-1][1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None, help="run id under runs/ (default: the one with the largest predictions.jsonl)")
    args = parser.parse_args(argv)

    cfg = load_config()
    runs_dir = cfg.path("runs")
    run_id = args.run or _default_run_id(runs_dir)
    run_dir = runs_dir / run_id

    result = analyze_run(run_dir)
    (run_dir / "cascade.json").write_text(json.dumps(result, indent=2) + "\n")
    make_plots(result, run_dir / "plots")
    markdown = render_markdown(result)
    (run_dir / "cascade.md").write_text(markdown + "\n")
    print(markdown)


if __name__ == "__main__":
    main()
