"""Turns a finished run's predictions into calibration params, metrics.json, plots and report.md.

Nothing here re-runs a model: everything is computed from predictions.jsonl (logits included) and extras.json.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .. import calibration
from ..config import Config
from ..logging_utils import dumps, read_jsonl
from ..scorer import softmax
from . import metrics as M

GO_THRESHOLDS = {"accuracy_gap_points": 5.0, "ece": 0.05, "permutation": 1e-3, "latency_cuda_ms": 500, "latency_apple_ms": 2000}

# Chart colors: a validated two-series pair (blue, orange) on a light surface; text stays in ink, never series color.
SERIES = {"raw": "#2a78d6", "calibrated": "#eb6834"}
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


# --- calibration --------------------------------------------------------------------------------------


def _probability_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("z") is not None]


def fit_run_calibration(cfg: Config, rows: list[dict[str, Any]], run_id: str, isotonic: bool = False) -> list[calibration.CalibrationParams]:
    """One parameter set per configuration key, fit on the run's calibration split."""
    cal_rows = [r for r in _probability_rows(rows) if r["split"] == "calibration"]
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in cal_rows:
        model = r["model"].split(":", 1)[1]
        groups[(r["backend"], model, r["prompt_version"], r["image_token_budget"])].append(r)
    out = []
    for (backend, model, prompt_version, budget), group in sorted(groups.items(), key=lambda kv: str(kv[0])):
        methods = sorted({r["choice_method"] for r in group if r["choice_method"]}) or ["independent"]
        for method in methods:
            subset = [r for r in group if r["choice_method"] in (None, method)]
            n_noul = sum(r["type"] == "noul" for r in subset)
            key = calibration.CalibrationKey(backend=backend, model=model, prompt_version=prompt_version,
                                             choice_method=method, image_token_budget=budget)
            out.append(calibration.build_params(
                key, subset, source_run=run_id, n_bins=cfg.eval.ece_bins,
                isotonic=isotonic and n_noul >= cfg.calibration.isotonic_min_n,
            ))
    return out


def apply_run_calibration(rows: list[dict[str, Any]], params_list: list[calibration.CalibrationParams]) -> None:
    """Fill `calibrated` (pooled fit) and `calibrated_suite` (per-suite fit) on every probability row, in place."""
    by_key = {(p.key.backend, p.key.model, p.key.prompt_version, p.key.image_token_budget, p.key.choice_method): p
              for p in params_list}
    for r in _probability_rows(rows):
        model = r["model"].split(":", 1)[1]
        params = by_key.get((r["backend"], model, r["prompt_version"], r["image_token_budget"], r["choice_method"] or "independent"))
        if params is None:  # noul/score rows of a run whose only choice method was `letter`
            params = next((p for k, p in by_key.items() if k[:4] == (r["backend"], model, r["prompt_version"], r["image_token_budget"])), None)
        if params is None or r["type"] not in params.types:
            continue
        z = np.asarray(r["z"], dtype=np.float64)
        r["calibrated"] = _listify(calibration.apply_fit(params.types[r["type"]], z))
        r["calibration_version"] = params.version
        suite_fit = params.per_suite.get(r["suite"])
        if suite_fit is not None:
            r["calibrated_suite"] = _listify(calibration.apply_fit(suite_fit, z))


def _listify(p: float | np.ndarray) -> float | list[float]:
    return float(p) if np.ndim(p) == 0 else [float(v) for v in p]


# --- metrics ------------------------------------------------------------------------------------------


def compute_metrics(cfg: Config, rows: list[dict[str, Any]], extras: dict[str, Any], run_config: dict[str, Any],
                    params_list: list[calibration.CalibrationParams]) -> dict[str, Any]:
    bins = cfg.eval.ece_bins
    units: dict[str, Any] = {}
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["suite"], r["backend"], r["method"])].append(r)

    for (suite, backend, method), group in sorted(grouped.items()):
        test = [r for r in group if r["split"] == "test"]
        unit: dict[str, Any] = {
            "suite": suite, "backend": backend, "method": method, "type": group[0]["type"],
            "n_test": len(test), "n_calibration": len(group) - len(test),
        }
        fail = extras.get("failures", {}).get(f"{suite.split('/')[0]}|{backend}|{method}", {"failed": 0, "attempted": len(group)})
        rate = fail["failed"] / max(1, fail["attempted"])
        unit["failures"] = {**fail, "rate": rate}
        unit["valid"] = rate <= cfg.eval.max_failure_rate
        if backend == "frontier":
            unit["hard_pick"] = M.hard_pick_metrics(test)
        elif test:
            unit["raw"] = M.probability_metrics(test, "raw", bins)
            if all(r.get("calibrated") is not None for r in test):
                unit["calibrated"] = M.probability_metrics(test, "calibrated", bins)
            if all(r.get("calibrated_suite") is not None for r in test):
                unit["calibrated_suite_fit"] = M.probability_metrics(test, "calibrated_suite", bins)
            unit["throughput"] = M.throughput(group)
            field = "calibrated" if "calibrated" in unit else "raw"
            unit["confusions"] = M.confusion_patterns(test, field)
            unit["top_errors"] = M.top_confident_errors(test, field, cfg.eval.top_errors)
            if any(r.get("human_disagreement") is not None for r in test):
                unit["human_disagreement_mean"] = float(np.mean([r["human_disagreement"] for r in test if r.get("human_disagreement") is not None]))
        units[f"{suite}|{backend}|{method}"] = unit

    out: dict[str, Any] = {
        "run_id": run_config.get("run_id"), "units": units,
        "letter_vs_independent": _letter_vs_independent(grouped, bins),
        "baseline": _baseline(grouped, bins),
        "permutation": extras.get("permutation", {}), "latency": extras.get("latency", {}),
        "calibration": {p.version: json.loads(p.model_dump_json()) for p in params_list},
    }
    out["go_no_go"], out["verdict"] = go_no_go(out, extras)
    return out


def _letter_vs_independent(grouped: dict[tuple, list[dict[str, Any]]], bins: int) -> dict[str, Any]:
    """`letter` next to `independent` on the same items and the same options (independent logits restricted to
    letter's option subset when the suite has more than 26 options)."""
    out = {}
    for (suite, backend, method), letter_rows in grouped.items():
        if method != "letter":
            continue
        indep = {r["item_id"]: r for r in grouped.get((suite, backend, "independent"), [])}
        letter_test = [r for r in letter_rows if r["split"] == "test" and r["item_id"] in indep]
        if not letter_test:
            continue
        same = []
        for lr in letter_test:
            ir = indep[lr["item_id"]]
            idx = [ir["keys"].index(k) for k in lr["keys"]]
            z = np.asarray(ir["z"], dtype=np.float64)[idx]
            temperature = 1.0
            same.append({**ir, "keys": lr["keys"], "label_index": lr["label_index"], "z": z.tolist(),
                         "raw": softmax(z, temperature).tolist()})
        full = [indep[r["item_id"]] for r in letter_test]
        entry = {
            "n": len(letter_test), "options_letter": len(letter_test[0]["keys"]), "options_full": len(full[0]["keys"]),
            "letter_raw": M.probability_metrics(letter_test, "raw", bins),
            "independent_same_options_raw": M.probability_metrics(same, "raw", bins),
            "independent_full_raw": M.probability_metrics(full, "raw", bins),
            "letter_latency": M.throughput(letter_test)["latency"], "independent_latency": M.throughput(full)["latency"],
        }
        if all(r.get("calibrated") is not None for r in letter_test):
            entry["letter_calibrated"] = M.probability_metrics(letter_test, "calibrated", bins)
        if all(r.get("calibrated") is not None for r in full):
            entry["independent_full_calibrated"] = M.probability_metrics(full, "calibrated", bins)
        out[f"{suite}|{backend}"] = entry
    return out


def _baseline(grouped: dict[tuple, list[dict[str, Any]]], bins: int) -> dict[str, Any]:
    """Local backends against the frontier baseline, on exactly the items the baseline answered."""
    out = {}
    for (suite, backend, method), frontier_rows in grouped.items():
        if backend != "frontier":
            continue
        answered = {r["item_id"] for r in frontier_rows}
        entry: dict[str, Any] = {"frontier": M.hard_pick_metrics(frontier_rows), "local": {}}
        for (s2, b2, m2), local_rows in grouped.items():
            if s2 != suite or b2 == "frontier" or m2 == "letter":
                continue
            same_items = [r for r in local_rows if r["item_id"] in answered]
            full_test = [r for r in local_rows if r["split"] == "test"]
            field = "calibrated" if all(r.get("calibrated") is not None for r in full_test) else "raw"
            entry["local"][b2] = {
                "accuracy_same_items": M.probability_metrics(same_items, field, bins).get("accuracy"),
                "selective_accuracy_80": M.probability_metrics(full_test, field, bins).get("selective_accuracy", {}).get("80"),
                "field": field,
            }
        out[suite] = entry
    return out


def error_breakdown(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Per suite, for the primary backend: error rate, calibrated ECE, and the gap to the baseline when there is one.
    Sorted so the suites that most need v1 data come first."""
    primary = primary_backend(metrics)
    rows = []
    for unit in metrics["units"].values():
        if unit["backend"] != primary or unit["method"] == "letter" or "raw" not in unit:
            continue
        best = unit.get("calibrated") or unit["raw"]
        base = metrics["baseline"].get(unit["suite"], {})
        local = (base.get("local") or {}).get(primary) or {}
        gap = None
        if base.get("frontier", {}).get("accuracy") is not None and local.get("accuracy_same_items") is not None:
            gap = 100 * (base["frontier"]["accuracy"] - local["accuracy_same_items"])
        rows.append({"suite": unit["suite"], "type": unit["type"], "error_rate": 1 - best["accuracy"], "ece": best["ece"],
                     "ece_over": best["ece"] > GO_THRESHOLDS["ece"], "gap_points": gap, "n": unit["n_test"],
                     "confusions": unit.get("confusions", [])})
    return sorted(rows, key=lambda r: (-(r["gap_points"] if r["gap_points"] is not None else -1e9), -r["error_rate"]))


def recommended_v1_data(metrics: dict[str, Any]) -> str:
    rows = error_breakdown(metrics)
    if not rows:
        return "not enough results"
    by_error = sorted(rows, key=lambda r: -r["error_rate"])
    parts = ["by error rate: " + ", ".join(f"{r['suite']} ({r['type']}) {100 * r['error_rate']:.1f}%" for r in by_error[:3])]
    over = [r for r in rows if r["ece_over"]]
    if over:
        parts.append("calibrated ECE still over 0.05: " + ", ".join(f"{r['suite']} {r['ece']:.3f}" for r in over))
    gaps = [r for r in rows if r["gap_points"] is not None]
    if gaps:
        parts.append("largest gap to baseline: " + ", ".join(f"{r['suite']} {r['gap_points']:+.1f} pts" for r in gaps[:3]))
    else:
        parts.append("gap to a frontier baseline not measured")
    return "; ".join(parts)


def primary_backend(metrics: dict[str, Any]) -> str | None:
    backends = {u["backend"] for u in metrics["units"].values()}
    return "vlm" if "vlm" in backends else ("siglip" if "siglip" in backends else None)


def go_no_go(metrics: dict[str, Any], extras: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    primary = primary_backend(metrics)
    units = [u for u in metrics["units"].values() if u["backend"] == primary and u["method"] != "letter"]
    table: list[dict[str, Any]] = []

    gaps, sel_ok = [], []
    for suite, entry in metrics["baseline"].items():
        local = entry["local"].get(primary)
        if local and local["accuracy_same_items"] is not None and entry["frontier"].get("accuracy") is not None:
            gaps.append(100 * (entry["frontier"]["accuracy"] - local["accuracy_same_items"]))
            if local["selective_accuracy_80"] is not None:
                sel_ok.append((suite, local["selective_accuracy_80"], entry["frontier"]["accuracy"]))
    if gaps:
        gap = float(np.mean(gaps))
        table.append({"metric": "Accuracy gap vs frontier baseline", "threshold": "<= 5 points, macro-averaged over suites",
                      "measured": f"{gap:+.1f} points over {len(gaps)} suites", "pass": gap <= GO_THRESHOLDS["accuracy_gap_points"]})
    else:
        table.append({"metric": "Accuracy gap vs frontier baseline", "threshold": "<= 5 points, macro-averaged over suites",
                      "measured": "not measured (no frontier baseline run)", "pass": None})

    eces = [(u["suite"], u["calibrated"]["ece"]) for u in units if "calibrated" in u]
    if eces:
        worst = max(eces, key=lambda t: t[1])
        failing = [s for s, e in eces if e > GO_THRESHOLDS["ece"]]
        floors = {u["suite"]: u["calibrated"].get("ece_floor") for u in units if "calibrated" in u}
        table.append({"metric": "ECE after calibration", "threshold": "<= 0.05 per suite (15 equal-mass bins)",
                      "measured": f"worst {worst[1]:.3f} ({worst[0]}, sampling floor {floors[worst[0]]:.3f}); "
                                  + (f"over threshold: {', '.join(failing)}" if failing else "all suites under"),
                      "pass": not failing})
    else:
        table.append({"metric": "ECE after calibration", "threshold": "<= 0.05 per suite (15 equal-mass bins)",
                      "measured": "not measured (uncalibrated run)", "pass": None})

    if sel_ok:
        local_mean = float(np.mean([s for _, s, _ in sel_ok]))
        base_mean = float(np.mean([b for _, _, b in sel_ok]))
        table.append({"metric": "Selective accuracy at 80% coverage", "threshold": ">= baseline full-coverage accuracy",
                      "measured": f"{local_mean:.3f} vs baseline {base_mean:.3f} (macro over {len(sel_ok)} suites)",
                      "pass": local_mean >= base_mean})
    else:
        sel = [u.get("calibrated", u.get("raw", {})).get("selective_accuracy", {}).get("80") for u in units]
        sel = [s for s in sel if s is not None]
        measured = f"{np.mean(sel):.3f} local (macro); baseline not measured" if sel else "not measured"
        table.append({"metric": "Selective accuracy at 80% coverage", "threshold": ">= baseline full-coverage accuracy",
                      "measured": measured, "pass": None})

    perms = {k: v for k, v in metrics["permutation"].items() if k.endswith("|independent") and f"|{primary}|" in k}
    if perms:
        worst_key = max(perms, key=lambda k: perms[k]["max_abs_dp"])
        worst = perms[worst_key]["max_abs_dp"]
        table.append({"metric": "Permutation invariance (choice)", "threshold": "max probability shift <= 1e-3 (independent)",
                      "measured": f"{worst:.1e} ({worst_key.split('|')[0]})", "pass": worst <= GO_THRESHOLDS["permutation"]})
    else:
        table.append({"metric": "Permutation invariance (choice)", "threshold": "max probability shift <= 1e-3 (independent)",
                      "measured": "not measured", "pass": None})

    lat = metrics["latency"].get(primary)
    device = (extras.get("devices", {}).get(primary) or {}).get("device")
    target = GO_THRESHOLDS["latency_cuda_ms"] if device == "cuda" else GO_THRESHOLDS["latency_apple_ms"]
    if lat:
        table.append({"metric": "Latency, 1 image + 5 questions", "threshold": f"recorded; target <= {target} ms on {device} (not a gate)",
                      "measured": f"p50 {lat['p50_ms']:.0f} ms, p95 {lat['p95_ms']:.0f} ms", "pass": "recorded",
                      "within_target": lat["p50_ms"] <= target})
    else:
        table.append({"metric": "Latency, 1 image + 5 questions", "threshold": f"recorded; target <= {target} ms (not a gate)",
                      "measured": "not measured", "pass": "recorded"})

    gates = [row["pass"] for row in table if row["pass"] != "recorded"]
    if any(g is False for g in gates):
        verdict = "NO-GO"
    elif all(g is True for g in gates):
        verdict = "GO"
    else:
        verdict = "PARTIAL"
    return table, verdict


# --- plots --------------------------------------------------------------------------------------------


def make_plots(cfg: Config, rows: list[dict[str, Any]], plots_dir: Path) -> dict[str, dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
        "axes.labelcolor": INK_MUTED, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "grid.linestyle": "-", "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    })
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in _probability_rows(rows):
        if r["split"] == "test":
            grouped[(r["suite"], r["backend"], r["method"])].append(r)

    written: dict[str, dict[str, str]] = {}
    for (suite, backend, method), group in sorted(grouped.items()):
        fields = ["raw"] + (["calibrated"] if all(r.get("calibrated") is not None for r in group) else [])
        curves = {f: M.curves(group, f, cfg.eval.ece_bins) for f in fields}
        stem = f"{suite.replace('/', '_')}__{backend}__{method}"
        subtitle = f"{suite} · {backend} · {method} · test split, n={len(group)}"

        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1, label="perfect calibration")
        for f in fields:
            pts = curves[f]["reliability"]
            ax.plot([p["confidence"] for p in pts], [p["accuracy"] for p in pts], color=SERIES[f], linewidth=2,
                    marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1, label=f, solid_capstyle="round")
        ax.set_xlim(0, 1.02), ax.set_ylim(0, 1.02)
        ax.set_xlabel("confidence in the top answer (equal-mass bins)"), ax.set_ylabel("accuracy")
        _titles(ax, "Reliability", subtitle)
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(plots_dir / f"{stem}__reliability.png", dpi=144)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        for f in fields:
            ax.plot(curves[f]["coverage"], curves[f]["risk"], color=SERIES[f], linewidth=2, label=f, solid_capstyle="round")
        ax.set_xlim(0, 1), ax.set_ylim(bottom=0)
        ax.set_xlabel("coverage (most confident first)"), ax.set_ylabel("risk (error rate)")
        _titles(ax, "Risk-coverage", subtitle)
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(plots_dir / f"{stem}__risk_coverage.png", dpi=144)
        plt.close(fig)
        written[f"{suite}|{backend}|{method}"] = {"reliability": f"plots/{stem}__reliability.png",
                                                  "risk_coverage": f"plots/{stem}__risk_coverage.png"}
    return written


def _titles(ax, title: str, subtitle: str) -> None:
    ax.set_title(title, pad=22)
    ax.text(0.0, 1.025, subtitle, transform=ax.transAxes, color=INK_MUTED, fontsize=9, va="bottom", ha="left")


# --- report.md ----------------------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pass(value: Any) -> str:
    return {True: "pass", False: "FAIL", None: "not measured", "recorded": "recorded"}[value]


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def render_report(cfg: Config, metrics: dict[str, Any], extras: dict[str, Any], run_config: dict[str, Any],
                  plots: dict[str, dict[str, str]]) -> str:
    primary = primary_backend(metrics)
    trim = run_config.get("trim", {})
    out = [f"# glance eval report · run `{metrics['run_id']}`", ""]
    out += [f"**Go/no-go: {metrics['verdict']}** (judged on held-out test splits; primary local backend: `{primary}`)", ""]
    out += [_table(["Metric", "Threshold", "Measured", "Pass"],
                   [[r["metric"], r["threshold"], r["measured"], _pass(r["pass"])] for r in metrics["go_no_go"]]), ""]

    out += ["## Run", ""]
    doctor = (extras.get("doctor") or {})
    out += [f"- Machine: {doctor.get('chip')}, {doctor.get('ram_gb')} GB RAM, device `{doctor.get('device')}`, tier `{doctor.get('selected_tier')}`"]
    for name, model in extras.get("models", {}).items():
        dev = extras.get("devices", {}).get(name, {})
        out += [f"- `{name}`: `{model}` ({dev.get('dtype')}, image token budget {dev.get('image_token_budget')})"]
    out += [f"- Harness {run_config.get('harness_version')} at git `{(run_config.get('git_sha') or '')[:10]}`, prompts `p1`, "
            f"prefix cache {'on' if extras.get('prefix_cache') else 'off (reference path)'}"]
    if extras.get("cache_speedup"):
        out += [f"- Prefix cache check: {extras['cache_speedup']}"]
    final_n = trim.get("final_n") or {}
    used = ", ".join(f"{suite} {count}" for suite, count in final_n.items()) if isinstance(final_n, dict) else str(final_n)
    out += [f"- n per suite: requested {trim.get('requested_n')}; used {used}"
            + (f" (**trimmed** to fit --max-hours {trim.get('max_hours')}: ETA at the requested n was "
               f"{trim.get('eta_hours_at_requested_n')} h; each suite got the same time cap, so only expensive suites lost items)"
               if trim.get("trimmed") else "")
            + "; seed 7, calibration/test alternate down the seeded order"]
    for note in run_config.get("notes", []):
        out += [f"- Note: {note}"]
    for name, meta in (run_config.get("suites") or {}).items():
        if str(meta.get("status", "ok")) != "ok":
            out += [f"- Suite `{name}`: {meta['status']}"]
    out += [""]

    out += ["## Per-suite results (test split)", ""]
    headers = ["Suite", "Backend", "Method", "n", "Acc", "AUROC / F1 / MAE", "NLL raw→cal", "Brier raw→cal", "ECE raw→cal",
               "ECE floor at this n", "Sel acc 50/80/90/100 (cal)", "Failures", "Valid"]
    body = []
    for unit in metrics["units"].values():
        if unit["backend"] == "frontier":
            body.append([unit["suite"], "frontier", "pick", unit["n_test"], _fmt(unit.get("hard_pick", {}).get("accuracy")),
                         "-", "-", "-", "-", "-", "-", f"{unit['failures']['failed']}/{unit['failures']['attempted']}", _fmt(unit["valid"])])
            continue
        raw, cal = unit.get("raw", {}), unit.get("calibrated", {})
        extra = raw.get("auroc") if unit["type"] == "noul" else raw.get("macro_f1") if unit["type"] == "choice" else (cal or raw).get("mae_levels")
        sel = (cal or raw).get("selective_accuracy", {})
        body.append([
            unit["suite"], unit["backend"], unit["method"], unit["n_test"],
            f"{_fmt(raw.get('accuracy'))}" + (f" → {_fmt(cal.get('accuracy'))}" if cal and cal.get("accuracy") != raw.get("accuracy") else ""),
            _fmt(extra), f"{_fmt(raw.get('nll'))} → {_fmt(cal.get('nll'))}", f"{_fmt(raw.get('brier'))} → {_fmt(cal.get('brier'))}",
            f"{_fmt(raw.get('ece'))} → {_fmt(cal.get('ece'))}", _fmt((cal or raw).get("ece_floor")),
            " / ".join(_fmt(sel.get(k)) for k in ("50", "80", "90", "100")),
            f"{unit['failures']['failed']}/{unit['failures']['attempted']}", _fmt(unit["valid"]),
        ])
    out += [_table(headers, body), ""]
    out += ["AUROC for noul, macro-F1 for choice, mean absolute error in levels (expected score vs label) for score. "
            "Selective accuracy ranks by `confidence` (noul: `2·|p − 0.5|`). A suite with more than 2% failed items is invalid. "
            "`ECE floor at this n` is the ECE a perfectly calibrated predictor with the same confidences would measure on this "
            "many items (200 simulated draws): equal-mass ECE is biased upward on small samples, so read each ECE against its floor.", ""]

    flagged = [u for u in metrics["units"].values() if "calibrated" in u and u["calibrated"]["ece"] >= u["raw"]["ece"]]
    if flagged:
        out += ["**Flagged: calibrated ECE is not below raw ECE on the test split for:** "
                + ", ".join(f"`{u['suite']}` ({u['backend']}, {u['method']}: {u['raw']['ece']:.3f} → {u['calibrated']['ece']:.3f})" for u in flagged), ""]

    out += ["## Error breakdown (what v1 data this points to)", ""]
    out += [_table(["Suite", "Type", "Test n", "Error rate", "ECE (best available)", "Over ECE threshold", "Gap to baseline (points)", "Top confusions"],
                   [[r["suite"], r["type"], r["n"], f"{100 * r['error_rate']:.1f}%", _fmt(r["ece"]), _fmt(r["ece_over"]),
                     "-" if r["gap_points"] is None else f"{r['gap_points']:+.1f}",
                     "; ".join(f"{c['true']} → {c['predicted']} ({c['count']})" for c in r["confusions"]) or "-"]
                    for r in error_breakdown(metrics)]), ""]
    out += [f"Primary backend `{primary}`, `independent` for choice. Recommended v1 data: {recommended_v1_data(metrics)}.", ""]

    out += ["## Calibration", ""]
    rows = []
    for version, params in metrics["calibration"].items():
        key = params["key"]
        for qtype, fit in params["types"].items():
            shown = {k: round(v, 4) for k, v in fit["params"].items() if not isinstance(v, list)}
            rows.append([version, key["backend"], key["choice_method"], qtype, fit["method"], shown, fit["n"],
                         ", ".join(fit["suite_ids"]), f"{fit['nll_before']:.3f} → {fit['nll_after']:.3f}",
                         f"{fit['ece_before']:.3f} → {fit['ece_after']:.3f}"])
    if rows:
        out += [_table(["Version", "Backend", "Choice method", "Type", "Fit", "Params", "n (cal split)", "Suites pooled",
                        "NLL before→after", "ECE before→after"], rows), ""]
        gain = []
        for unit in metrics["units"].values():
            if "calibrated" in unit and "calibrated_suite_fit" in unit:
                gain.append([unit["suite"], unit["backend"], unit["method"], _fmt(unit["raw"]["ece"]), _fmt(unit["calibrated"]["ece"]),
                             _fmt(unit["calibrated_suite_fit"]["ece"]), _fmt(unit["calibrated"]["nll"]), _fmt(unit["calibrated_suite_fit"]["nll"])])
        out += ["Pooled fit (what the API applies) against a per-suite fit, on the test split:", "",
                _table(["Suite", "Backend", "Method", "ECE raw", "ECE pooled fit", "ECE per-suite fit", "NLL pooled", "NLL per-suite"], gain), ""]
    else:
        out += ["This run is uncalibrated (no calibration params were fit).", ""]

    out += ["## `independent` against `letter`", ""]
    if metrics["letter_vs_independent"]:
        rows = []
        for key, e in metrics["letter_vs_independent"].items():
            suite, backend = key.split("|")
            lat_l, lat_i = e.get("letter_latency") or {}, e.get("independent_latency") or {}
            rows.append([suite, backend, e["n"], f"{e['options_letter']} of {e['options_full']}",
                         _fmt(e["letter_raw"]["accuracy"]), _fmt(e["independent_same_options_raw"]["accuracy"]),
                         _fmt(e["independent_full_raw"]["accuracy"]),
                         f"{_fmt(e['letter_raw']['ece'])} → {_fmt(e.get('letter_calibrated', {}).get('ece'))}",
                         f"{_fmt(e['independent_full_raw']['ece'])} → {_fmt(e.get('independent_full_calibrated', {}).get('ece'))}",
                         f"{_fmt(lat_l.get('p50_ms'), 0)} / {_fmt(lat_i.get('p50_ms'), 0)}"])
        out += [_table(["Suite", "Backend", "n", "Options seen by letter", "Acc letter", "Acc independent (same options)",
                        "Acc independent (all options)", "ECE letter raw→cal", "ECE independent raw→cal", "p50 ms letter / independent"], rows), ""]
        out += ["`letter` is capped at 26 options. On larger suites it sees the true label plus 25 seeded random distractors; "
                "`independent (same options)` restricts the independent logits to that same subset, so the two columns are comparable.", ""]
        perm = [[k.replace("|", " · "), v["requests"], f"{v['max_abs_dp']:.1e}", f"{v['mean_abs_dp']:.1e}", v["choice_flips"]]
                for k, v in metrics["permutation"].items()]
        if perm:
            out += ["Permutation sensitivity (test items × random option orders, shift in raw probabilities):", "",
                    _table(["Unit", "Requests", "max abs Δp", "mean abs Δp", "Choice flips"], perm), ""]
    else:
        out += ["No choice suite ran with both methods.", ""]

    out += ["## Local against the frontier baseline", ""]
    if metrics["baseline"]:
        rows = []
        for suite, e in metrics["baseline"].items():
            for backend, local in e["local"].items():
                rows.append([suite, e["frontier"]["n"], _fmt(e["frontier"]["accuracy"]), backend, _fmt(local["accuracy_same_items"]),
                             f"{100 * (e['frontier']['accuracy'] - local['accuracy_same_items']):+.1f}", _fmt(local["selective_accuracy_80"])])
        out += [_table(["Suite", "Baseline n", "Baseline acc", "Local backend", "Local acc (same items)", "Gap (points)",
                        "Local sel acc @80% (full test split)"], rows), ""]
    else:
        out += ["The frontier baseline did not run, so the accuracy-gap and selective-accuracy rows read \"not measured\". "
                "Set `FRONTIER_MODEL` and its API key in `.env`, then run `glance eval --model frontier --confirm-spend` "
                "(or the full eval again with `--confirm-spend`).", ""]

    out += ["## Latency and throughput", ""]
    rows = [[name, v["images"], v["questions"], v["statements"], f"{v['p50_ms']:.0f}", f"{v['p95_ms']:.0f}", v["n"]] for name, v in metrics["latency"].items()]
    if rows:
        out += [_table(["Backend", "Images", "Questions", "Statements", "p50 ms", "p95 ms", "Requests"], rows), ""]
    rows = []
    for unit in metrics["units"].values():
        t = unit.get("throughput")
        if t and t.get("latency"):
            rows.append([unit["suite"], unit["backend"], unit["method"], f"{t['latency']['p50_ms']:.0f}", f"{t['latency']['p95_ms']:.0f}",
                         _fmt(t.get("statements_per_second"), 1), _fmt(t.get("mean_image_tokens"), 0),
                         _fmt(t.get("off_mass_mean"), 5), _fmt(t.get("off_mass_p95"), 5)])
    out += [_table(["Suite", "Backend", "Method", "p50 ms / request", "p95 ms / request", "Statements / s", "Mean image tokens",
                    "off_mass mean", "off_mass p95"], rows), ""]

    out += ["## Plots", ""]
    for key, files in plots.items():
        out += [f"**{key.replace('|', ' · ')}**", "", f"![reliability]({files['reliability']}) ![risk-coverage]({files['risk_coverage']})", ""]

    out += [f"## Highest-confidence errors (up to {cfg.eval.top_errors} per suite)", ""]
    for unit in metrics["units"].values():
        if unit["backend"] != primary or unit["method"] == "letter" or not unit.get("top_errors"):
            continue
        out += [f"### {unit['suite']} · {unit['backend']} · {unit['method']}", ""]
        if unit.get("confusions"):
            out += ["Most common confusions: " + "; ".join(f"{c['true']} → {c['predicted']} ({c['count']})" for c in unit["confusions"]), ""]
        out += [_table(["Item", "True", "Predicted", "Confidence", "Top probability", "Image"],
                       [[e["item_id"], e["true"], e["predicted"], _fmt(e["confidence"]), _fmt(e["top_probability"]), f"`{e['image_path']}`"]
                        for e in unit["top_errors"]]), ""]
    return "\n".join(out)


def summary_text(metrics: dict[str, Any], extras: dict[str, Any], run_config: dict[str, Any]) -> str:
    """The section 12 final summary block."""
    doctor = extras.get("doctor") or {}
    primary = primary_backend(metrics)
    models = extras.get("models", {})
    units = [u for u in metrics["units"].values() if u["backend"] == primary and u["method"] != "letter" and "raw" in u]
    lines = ["GLANCE v0 SUMMARY", f"run_id:             {metrics['run_id']}",
             f"machine:            {doctor.get('chip')}, {doctor.get('ram_gb')} GB RAM, {doctor.get('device')}, {doctor.get('dtype')}",
             f"models:             {models.get('siglip', 'none')}, {models.get('vlm', 'none')}, {models.get('frontier') or 'none'}",
             f"go/no-go:           {metrics['verdict']}", "", "| metric | threshold | measured | pass |", "| --- | --- | --- | --- |"]
    lines += [f"| {r['metric']} | {r['threshold']} | {r['measured']} | {_pass(r['pass'])} |" for r in metrics["go_no_go"]]
    lines += [""]
    lv = metrics["letter_vs_independent"]
    if lv:
        parts = []
        for key, e in lv.items():
            lat_l, lat_i = e.get("letter_latency") or {}, e.get("independent_latency") or {}
            parts.append(f"{key.split('|')[0]}: acc {e['independent_same_options_raw']['accuracy']:.3f} vs {e['letter_raw']['accuracy']:.3f} (same options), "
                         f"ECE cal {_fmt(e.get('independent_full_calibrated', {}).get('ece'))} vs {_fmt(e.get('letter_calibrated', {}).get('ece'))}, "
                         f"p50 {_fmt(lat_i.get('p50_ms'), 0)} vs {_fmt(lat_l.get('p50_ms'), 0)} ms")
        lines += ["choice method:      independent vs letter: " + "; ".join(parts)]
    else:
        lines += ["choice method:      independent vs letter: not measured"]
    if units:
        weakest = min(units, key=lambda u: (u.get("calibrated") or u["raw"])["accuracy"])
        conf = "; ".join(f"{c['true']} -> {c['predicted']} ({c['count']})" for c in weakest.get("confusions", [])) or "none"
        lines += [f"weakest suite:      {weakest['suite']} (acc {(weakest.get('calibrated') or weakest['raw'])['accuracy']:.3f}), top confusions: {conf}"]
        lines += ["calibration gain:   " + "; ".join(
            f"{u['suite']} {u['raw']['ece']:.3f} -> {_fmt(u.get('calibrated', {}).get('ece'))}" for u in units)]
    lines += [f"cache speedup:      {extras.get('cache_speedup') or 'not measured'}"]
    lines += ["failures:           " + "; ".join(f"{k.replace('|', '/')} {v['failed']}/{v['attempted']}" for k, v in extras.get("failures", {}).items())]
    lines += ["deviations:         " + ("; ".join(run_config.get("notes", [])) or "none")]
    lines += [f"recommended v1 data: {recommended_v1_data(metrics)}"]
    return "\n".join(lines)


# --- entry point --------------------------------------------------------------------------------------


def finalize_run(cfg: Config, run_dir: Path, calibrate: bool = True) -> dict[str, Any]:
    """predictions.jsonl -> calibration (in the run dir), metrics.json, plots/, report.md, summary.txt."""
    run_dir = Path(run_dir)
    rows = read_jsonl(run_dir / "predictions.jsonl")
    extras = json.loads((run_dir / "extras.json").read_text()) if (run_dir / "extras.json").exists() else {}
    env = json.loads((run_dir / "env.json").read_text()) if (run_dir / "env.json").exists() else {}
    extras["doctor"] = env.get("doctor")
    run_config = yaml.safe_load((run_dir / "config.yaml").read_text()) if (run_dir / "config.yaml").exists() else {}
    run_config["run_id"] = run_dir.name
    cache_check = cfg.path("logs") / "cache_check.json"
    if cache_check.exists():  # written by tests/test_m4_prefix_cache.py, the section 6 acceptance check
        check = json.loads(cache_check.read_text())
        met = check["argmax_agree"] == check["items"] and check["max_abs_dz"] <= check["threshold"]
        extras["cache_speedup"] = (
            f"{check['speedup']:.2f}x on {check['items']} items ({check['statements']} statements); argmax agreement "
            f"{check['argmax_agree']}/{check['items']}, max |dz| {check['max_abs_dz']:.3f} vs limit {check['threshold']} -> "
            f"{'acceptance met' if met else 'acceptance NOT met, shipped uncached'}"
        )

    params_list: list[calibration.CalibrationParams] = []
    if calibrate and any(r["split"] == "calibration" for r in _probability_rows(rows)):
        params_list = fit_run_calibration(cfg, rows, run_dir.name)
        for params in params_list:
            calibration.save_params(run_dir / "calibration", params)
        apply_run_calibration(rows, params_list)
        (run_dir / "predictions.jsonl").write_text("".join(dumps(r) + "\n" for r in rows))

    metrics = compute_metrics(cfg, rows, extras, run_config, params_list)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    plots = make_plots(cfg, rows, run_dir / "plots")
    (run_dir / "report.md").write_text(render_report(cfg, metrics, extras, run_config, plots) + "\n")
    (run_dir / "summary.txt").write_text(summary_text(metrics, extras, run_config) + "\n")
    return metrics
