"""Turns score-lab result JSONs into publication-quality matplotlib figures.

Reads two JSON files that `glance.lab` already writes to disk:

  --report   an analysis JSON keyed "<ladder>|<method_key>" (e.g. lab/runs/pilot2_dev.json), each value
             holding `variants` (calibration name -> {cv_nll, fit_split_nll, eval}), `chosen`, `forward_passes`,
             `latency_ms_p50`, `image_tokens`.
  --extras   a follow-up JSON (e.g. lab/runs/pilot_extras_dev.json) with up to three sections:
             "learning_curve", "transfer", "ensemble".

Nothing here re-runs a model, fits anything, or touches glance.lab.collect / glance eval: every plotted value
comes straight out of the two input files. Dependencies are numpy, matplotlib and stdlib only (report.py is
imported only for its colour constants).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402

from glance.evals.report import GRID, INK, INK_MUTED, SERIES, SURFACE  # noqa: E402

SCALE_ORDER = ["blur", "noise", "jpeg", "exposure", "resolution"]
SCALE_COLORS = {
    "blur": "#2a78d6",
    "noise": "#eb6834",
    "jpeg": "#1baf7a",
    "exposure": "#eda100",
    "resolution": "#e87ba4",
}
TARGET_ACCURACY = 0.85


# --- shared style --------------------------------------------------------------------------------------


def set_rcparams() -> None:
    """Copied from glance.evals.report.make_plots so figures match the project's existing plots."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
        "axes.labelcolor": INK_MUTED, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "grid.linestyle": "-", "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    })


def _titles(ax, title: str, subtitle: str) -> None:
    ax.set_title(title, pad=20)
    ax.text(0.0, 1.025, subtitle, transform=ax.transAxes, color=INK_MUTED, fontsize=8.5, va="bottom", ha="left")


def _save(fig, out_dir: Path, stem: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png, pdf = out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    return {"png": png.name, "pdf": pdf.name}


def warn(msg: str) -> None:
    print(f"make_lab_figures: {msg}", file=sys.stderr)


def declutter_labels(fig, ax, anns: list, max_iter: int = 120, px_step: float = 2.0, max_shift_px: float = 90.0) -> None:
    """Nudge overlapping annotate() labels apart vertically (in pixel space), each already drawn with a leader
    line back to its data point (textcoords='data'), so direct labels on a crowded scatter stay readable.
    Both labels in an overlapping pair move apart symmetrically, and each label's total displacement from its
    starting position is capped so a persistent overlap (e.g. 3+ labels stacked in a small area) settles for
    "good enough" instead of drifting off the axes."""
    if len(anns) < 2:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    start_px = [ax.transData.transform(a.xyann) for a in anns]
    for _ in range(max_iter):
        boxes = [a.get_window_extent(renderer) for a in anns]
        moved = False
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                if not boxes[i].overlaps(boxes[j]):
                    continue
                ci, cj = boxes[i].y0 + boxes[i].y1, boxes[j].y0 + boxes[j].y1
                direction = 1 if cj >= ci else -1
                if direction == 0:
                    direction = 1 if j > i else -1
                for idx, sign in ((i, -direction), (j, direction)):
                    px, py = ax.transData.transform(anns[idx].xyann)
                    new_py = py + sign * px_step
                    if abs(new_py - start_px[idx][1]) > max_shift_px:
                        continue  # already at its displacement cap; leave it be
                    anns[idx].xyann = inv.transform((px, new_py))
                moved = True
        if not moved:
            break
        fig.canvas.draw()
    fig.canvas.draw()


# --- report.json helpers --------------------------------------------------------------------------------


def present_scales(report: dict[str, Any]) -> list[str]:
    have = {e["ladder"] for e in report.values()}
    return [s for s in SCALE_ORDER if s in have] + sorted(have - set(SCALE_ORDER))


def method_keys(report: dict[str, Any]) -> list[str]:
    return sorted({e["method_key"] for e in report.values()})


def variant_eval(entry: dict[str, Any], calib_name: str | None, field: str) -> Any:
    if calib_name is None:
        return None
    v = entry.get("variants", {}).get(calib_name)
    return None if v is None else v.get("eval", {}).get(field)


def chosen_accuracy(entry: dict[str, Any]) -> float | None:
    return variant_eval(entry, entry.get("chosen"), "accuracy")


def mean_chosen_accuracy(report: dict[str, Any], method: str) -> float | None:
    vals = [a for s in present_scales(report) if (e := report.get(f"{s}|{method}")) is not None
            and (a := chosen_accuracy(e)) is not None]
    return float(np.mean(vals)) if vals else None


def scale_n(report: dict[str, Any], scale: str) -> int | None:
    ns = [n for m in method_keys(report) if (e := report.get(f"{scale}|{m}")) is not None
          for v in e.get("variants", {}).values() if (n := v.get("eval", {}).get("n")) is not None]
    if not ns:
        return None
    vals, counts = np.unique(ns, return_counts=True)
    return int(vals[np.argmax(counts)])


def report_test_n(report: dict[str, Any]) -> int | None:
    ns = [scale_n(report, s) for s in present_scales(report)]
    ns = [n for n in ns if n is not None]
    if not ns:
        return None
    vals, counts = np.unique(ns, return_counts=True)
    return int(vals[np.argmax(counts)])


# --- Figure A: methods_by_scale --------------------------------------------------------------------------


def fig_methods_by_scale(report: dict[str, Any], out_dir: Path, split_label: str) -> dict[str, Any] | None:
    scales = present_scales(report)
    if not scales:
        warn("skip fig_methods_by_scale: --report has no entries with a recognized 'ladder'")
        return None
    methods = method_keys(report)
    ranked = sorted(methods, key=lambda m: (mean_chosen_accuracy(report, m) is None, -(mean_chosen_accuracy(report, m) or 0.0)))
    n_methods = len(ranked)
    y_pos = np.arange(n_methods)

    fig, axes = plt.subplots(1, len(scales), figsize=(4.3 * len(scales), 0.46 * n_methods + 2.6), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, scale in zip(axes, scales):
        n = scale_n(report, scale)
        for i, method in enumerate(ranked):
            e = report.get(f"{scale}|{method}")
            if e is None:
                continue
            y = y_pos[i]
            chosen = e.get("chosen")
            raw_acc = variant_eval(e, "raw", "accuracy")
            chosen_acc = variant_eval(e, chosen, "accuracy")
            if chosen_acc is None:
                continue
            if raw_acc is not None and chosen != "raw":
                ax.plot([raw_acc, chosen_acc], [y, y], color=INK_MUTED, linewidth=1, zorder=1)
                ax.plot(raw_acc, y, marker="o", markersize=6, markerfacecolor=SURFACE,
                        markeredgecolor=SERIES["raw"], markeredgewidth=1.6, linestyle="none", zorder=2)
                ax.plot(chosen_acc, y, marker="o", markersize=6.5, markerfacecolor=SERIES["calibrated"],
                        markeredgecolor=SERIES["calibrated"], linestyle="none", zorder=3)
            elif chosen == "raw":
                # Nothing to calibrate against: a single dot in the "raw" colour, no dumbbell.
                ax.plot(chosen_acc, y, marker="o", markersize=6.5, markerfacecolor=SERIES["raw"],
                        markeredgecolor=SERIES["raw"], linestyle="none", zorder=3)
            else:
                # No "raw" variant at all (ensembles only report the combined fit).
                ax.plot(chosen_acc, y, marker="o", markersize=6.5, markerfacecolor=SERIES["calibrated"],
                        markeredgecolor=SERIES["calibrated"], linestyle="none", zorder=3)
            ax.annotate(f"{chosen_acc:.3f}", (chosen_acc, y), xytext=(6, 0), textcoords="offset points",
                        fontsize=7.3, color=INK, va="center", ha="left", zorder=4)
        ax.axvline(TARGET_ACCURACY, color=INK_MUTED, linewidth=1, linestyle="--", zorder=0)
        ax.set_xlim(0, 1.2)
        ax.set_ylim(n_methods - 0.4, -0.6)  # inverted: rank 0 (best) at the top
        _titles(ax, scale, f"{split_label}, n={n}" if n is not None else split_label)
        ax.set_xlabel("accuracy")
        ax.tick_params(axis="y", length=0)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(ranked, fontsize=8.7)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    axes[-1].text(TARGET_ACCURACY, -1.05, "0.85 target", color=INK_MUTED, fontsize=8, ha="center", va="bottom")

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=6.5, markerfacecolor=SURFACE,
               markeredgecolor=SERIES["raw"], markeredgewidth=1.6, label="raw"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=7, markerfacecolor=SERIES["calibrated"],
               markeredgecolor=SERIES["calibrated"], label="chosen calibration"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.995), ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Calibrated vs raw accuracy, by method and degradation scale", x=0.005, ha="left",
                 fontsize=13, fontweight="bold", y=0.985)
    fig.text(0.005, 0.945, "methods ranked by mean chosen-calibration accuracy across scales (best at top)",
             color=INK_MUTED, fontsize=9, ha="left", va="top")
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.88))

    files = _save(fig, out_dir, "fig_methods_by_scale")
    return {
        "stem": "fig_methods_by_scale", **files,
        "description": "Dumbbell plot of raw vs chosen-calibration accuracy for every method, one panel per "
                        "degradation scale; methods are ordered top-to-bottom by mean chosen-calibration accuracy "
                        "across the five scales. The dashed line marks the project's 0.85 accuracy target.",
        "inputs": ["report"],
    }


# --- Figure B: learning_curve -----------------------------------------------------------------------------


def fig_learning_curve(extras: dict[str, Any], report: dict[str, Any], out_dir: Path, split_label: str) -> dict[str, Any] | None:
    lc = extras.get("learning_curve")
    if not lc:
        warn("skip fig_learning_curve: --extras has no 'learning_curve' section")
        return None
    scales = [s for s in SCALE_ORDER if s in lc] + sorted(set(lc) - set(SCALE_ORDER))
    all_ns = sorted({int(n) for curve in lc.values() for n in curve})
    positive_ns = [n for n in all_ns if n > 0]
    linthresh = min(positive_ns) if positive_ns else 1

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    end_labels: list[tuple[float, float, str]] = []
    for scale in scales:
        curve = lc[scale]
        ns = sorted(int(n) for n in curve)
        means = np.array([curve[str(n)]["accuracy"][0] for n in ns])
        sds = np.array([curve[str(n)]["accuracy"][1] for n in ns])
        color = SCALE_COLORS.get(scale, INK_MUTED)
        ax.plot(ns, means, color=color, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=1, solid_capstyle="round", zorder=3, label=scale)
        ax.fill_between(ns, means - sds, means + sds, color=color, alpha=0.15, linewidth=0, zorder=2)
        end_labels.append((ns[-1], means[-1], scale))

    # Stack direct end-of-line labels that would otherwise collide (same n, close accuracy).
    end_labels.sort(key=lambda t: (t[0], t[1]))
    used: dict[float, list[float]] = {}
    for x, y, scale in end_labels:
        placed = used.setdefault(x, [])
        dy = 0.0
        for prior in placed:
            if abs(y + dy - prior) < 0.035:
                dy += 0.035 if y >= prior else -0.035
        placed.append(y + dy)
        ax.annotate(scale, (x, y), xytext=(8, dy * 260), textcoords="offset points", fontsize=9,
                    color=INK, va="center", ha="left", fontweight="bold")

    ax.set_xscale("symlog", linthresh=linthresh, linscale=1.0)
    ax.xaxis.set_major_locator(FixedLocator(all_ns))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: str(int(x))))
    ax.set_xlim(-linthresh * 0.6, max(all_ns) * 2.2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("labeled examples used to fit calibration (n); n=0 is the raw, uncalibrated readout")
    ax.set_ylabel("accuracy")

    test_n = report_test_n(report)
    subtitle = f"{split_label}, mean ± 1 sd over random draws of the labeled subset" + (
        f"; evaluated on n={test_n} held-out items per scale" if test_n else "")
    _titles(ax, "Learning curve: calibration-set size vs accuracy", subtitle)
    ax.legend(loc="lower right", fontsize=8.7, ncol=1)
    fig.tight_layout()

    files = _save(fig, out_dir, "fig_learning_curve")
    return {
        "stem": "fig_learning_curve", **files,
        "description": "Accuracy vs number of labeled examples used to fit the per-scale calibration, one line "
                        "per degradation scale with a ±1 sd band over random draws; n=0 is the uncalibrated (raw) "
                        "readout.",
        "inputs": ["extras", "report (for the held-out n in the subtitle)"],
    }


# --- Figure C: accuracy_vs_cost / accuracy_vs_latency ------------------------------------------------------


def fig_accuracy_vs_x(report: dict[str, Any], out_dir: Path, split_label: str, *, x_field: str, x_label: str,
                       stem: str, title: str, xscale: str = "linear") -> dict[str, Any] | None:
    points = []
    for m in method_keys(report):
        accs, xs = [], []
        for scale in present_scales(report):
            e = report.get(f"{scale}|{m}")
            if e is None:
                continue
            a = chosen_accuracy(e)
            x = e.get(x_field)
            if a is not None:
                accs.append(a)
            if x is not None:
                xs.append(x)
        if not accs or not xs:
            continue
        points.append({"method": m, "accuracy": float(np.mean(accs)), "x": float(np.mean(xs)),
                        "is_ensemble": m.startswith("ens(")})
    if not points:
        warn(f"skip {stem}: no method in --report has both a chosen-calibration accuracy and '{x_field}'")
        return None

    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    ax.axhline(TARGET_ACCURACY, color=INK_MUTED, linewidth=1, linestyle="--", zorder=1)
    xs_all = [p["x"] for p in points]
    pad = (max(xs_all) - min(xs_all)) * 0.35 or 1.0
    ax.set_xlim(min(xs_all) - pad * 0.15, max(xs_all) + pad)
    if xscale == "log":
        ax.set_xscale("log")
    ax.set_ylim(0, 1.02)

    # Group points that land on (almost) the same spot so their labels don't repeat the same text.
    groups: dict[tuple[float, float], list[str]] = {}
    for p in points:
        key = (round(p["x"], 1), round(p["accuracy"], 3))
        groups.setdefault(key, []).append(p["method"])
    plotted_labels: set[tuple[float, float]] = set()
    anns, marker_xy = [], []
    for p in points:
        marker, size = ("D", 66) if p["is_ensemble"] else ("o", 70)
        # zorder above the label boxes below, so a marker that ends up under a neighboring label's background
        # patch (crowded low-cost cluster) still shows fully instead of being partly painted over.
        ax.scatter(p["x"], p["accuracy"], marker=marker, s=size, color=SERIES["calibrated"],
                   edgecolor=SURFACE, linewidth=0.8, zorder=6)
        key = (round(p["x"], 1), round(p["accuracy"], 3))
        if key in plotted_labels:
            continue
        plotted_labels.add(key)
        label = ", ".join(groups[key])
        # Initial label position a little above-right of the marker, in data coordinates (so decluttering
        # can move it further without the label text itself carrying an arrow through the marker).
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        tx, ty = p["x"] + 0.022 * (x1 - x0), p["accuracy"] + 0.02 * (y1 - y0)
        # A subtle surface-colored backing so a label that ends up sitting on a gridline or the 0.85 target
        # line (e.g. ens(digits+zoom_digits) is right next to it) stays legible without needing to dodge it.
        ann = ax.annotate(label, xy=(p["x"], p["accuracy"]), xytext=(tx, ty), textcoords="data", fontsize=8,
                           color=INK, va="bottom", ha="left", zorder=5,
                           bbox=dict(boxstyle="round,pad=0.12", facecolor=SURFACE, edgecolor="none", alpha=0.85))
        anns.append(ann)
        marker_xy.append((p["x"], p["accuracy"]))
    declutter_labels(fig, ax, anns)
    # Only draw a leader line where decluttering actually moved the label away from its marker; a label that
    # stayed put next to its point doesn't need one (and a near-zero-length line just draws a stub on the dot).
    fig.canvas.draw()
    for ann, (mx, my) in zip(anns, marker_xy):
        tx, ty = ann.xyann
        mpx, mpy = ax.transData.transform((mx, my))
        tpx, tpy = ax.transData.transform((tx, ty))
        if ((mpx - tpx) ** 2 + (mpy - tpy) ** 2) ** 0.5 > 24:
            ax.annotate("", xy=(mx, my), xytext=(tx, ty), textcoords="data", xycoords="data",
                        arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.7, shrinkA=2, shrinkB=8),
                        zorder=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel("mean chosen-calibration accuracy (across 5 scales)")
    n = report_test_n(report)
    subtitle = f"{split_label}, one point per method, mean across scales" + (f"; n={n} per scale" if n else "")
    _titles(ax, title, subtitle)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=SERIES["calibrated"], markersize=7.5, label="single method"),
        Line2D([0], [0], marker="D", linestyle="none", color=SERIES["calibrated"], markersize=7, label="ensemble"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8.7)
    fig.tight_layout()

    files = _save(fig, out_dir, stem)
    return {"stem": stem, **files, "description": title + ". Diamonds are ensembles; every other method is a circle.",
            "inputs": ["report"]}


# --- Figure D: transfer -------------------------------------------------------------------------------------


def fig_transfer(extras: dict[str, Any], out_dir: Path, split_label: str) -> dict[str, Any] | None:
    transfer = extras.get("transfer")
    if not transfer:
        warn("skip fig_transfer: --extras has no 'transfer' section")
        return None
    row_order = [r for r in SCALE_ORDER if r in transfer] + [r for r in ("POOLED", "NONE (raw)") if r in transfer]
    row_order += sorted(set(transfer) - set(row_order))
    col_candidates = {c for row in transfer.values() for c in row}
    col_order = [c for c in SCALE_ORDER if c in col_candidates] + sorted(col_candidates - set(SCALE_ORDER))
    matrix = np.array([[transfer.get(r, {}).get(c, np.nan) for c in col_order] for r in row_order])

    fig, ax = plt.subplots(figsize=(1.05 * len(col_order) + 2.6, 0.62 * len(row_order) + 2.2))
    cmap = plt.get_cmap("Blues")
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, rotation=30, ha="right")
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order)
    ax.grid(False)
    ax.set_xlabel("evaluated on")
    ax.set_ylabel("calibration fit on")
    for i in range(len(row_order)):
        for j in range(len(col_order)):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            rgba = cmap(norm(v))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = SURFACE if lum < 0.6 else INK
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9.5, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("accuracy", color=INK_MUTED)
    cbar.ax.yaxis.set_tick_params(color=INK_MUTED, labelcolor=INK_MUTED)
    plt.setp(cbar.ax.get_yticklabels(), color=INK_MUTED)

    _titles(ax, "Calibration transfer across degradation scales", f"{split_label}; POOLED fits one calibration "
            "across all scales, NONE (raw) applies no calibration")
    fig.tight_layout()

    files = _save(fig, out_dir, "fig_transfer")
    return {
        "stem": "fig_transfer", **files,
        "description": "Heatmap of accuracy when a calibration fit on one scale (row) is evaluated on another "
                        "scale (column); POOLED and NONE (raw) rows are included for reference.",
        "inputs": ["extras"],
    }


# --- README ---------------------------------------------------------------------------------------------


def build_readme(records: list[dict[str, Any]], skipped: list[str], cmd: str, report_path: Path, extras_path: Path) -> str:
    lines = [
        "# Lab figures",
        "",
        f"Generated by `tools/make_lab_figures.py` from `{report_path}` and `{extras_path}`.",
        "Nothing is re-run: every value plotted here is read straight out of those two JSON files.",
        "",
        "Regenerate everything with:",
        "",
        "```",
        cmd,
        "```",
        "",
        "## Figures",
        "",
    ]
    if not records:
        lines.append("(none produced)")
    for r in records:
        lines += [
            f"### {r['stem']}",
            "",
            r["description"],
            "",
            f"- Input file(s): {', '.join(r['inputs'])}",
            f"- Files: `{r['png']}`, `{r['pdf']}`",
            "",
        ]
    if skipped:
        lines += ["## Skipped", ""]
        lines += [f"- {s}" for s in skipped]
        lines += [""]
    return "\n".join(lines)


# --- entry point -----------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path, help="Analysis JSON, e.g. lab/runs/pilot2_dev.json")
    parser.add_argument("--extras", required=True, type=Path, help="Follow-up JSON, e.g. lab/runs/pilot_extras_dev.json")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for figures and README.md")
    parser.add_argument("--split-label", default="test split", help="Text used in subtitles, e.g. 'dev split'")
    args = parser.parse_args()

    def load(path: Path, flag: str) -> dict[str, Any]:
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            print(f"make_lab_figures: error: {flag} file not found: {path}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as exc:
            print(f"make_lab_figures: error: {flag} file is not valid JSON: {path} ({exc})", file=sys.stderr)
            sys.exit(1)

    report = load(args.report, "--report")
    extras = load(args.extras, "--extras")

    set_rcparams()
    args.out.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for result in (
        fig_methods_by_scale(report, args.out, args.split_label),
        fig_learning_curve(extras, report, args.out, args.split_label),
        fig_accuracy_vs_x(report, args.out, args.split_label, x_field="forward_passes", x_label="forward passes per question",
                           stem="fig_accuracy_vs_cost", title="Accuracy vs cost (forward passes)"),
        fig_accuracy_vs_x(report, args.out, args.split_label, x_field="latency_ms_p50", x_label="p50 latency (ms)",
                           stem="fig_accuracy_vs_latency", title="Accuracy vs latency", xscale="log"),
        fig_transfer(extras, args.out, args.split_label),
    ):
        if result is not None:
            records.append(result)

    input_names = {
        "report": args.report.name,
        "extras": args.extras.name,
        "report (for the held-out n in the subtitle)": f"{args.report.name} (for the held-out n in the subtitle)",
    }
    for r in records:
        r["inputs"] = [input_names.get(i, i) for i in r["inputs"]]

    skipped = []
    if not extras.get("learning_curve"):
        skipped.append("fig_learning_curve: --extras has no 'learning_curve' section")
    if not extras.get("transfer"):
        skipped.append("fig_transfer: --extras has no 'transfer' section")
    if not present_scales(report):
        skipped.append("fig_methods_by_scale, fig_accuracy_vs_cost, fig_accuracy_vs_latency: --report has no usable entries")

    cmd = (f"uv run python tools/make_lab_figures.py --report {args.report} --extras {args.extras} "
           f"--out {args.out} --split-label \"{args.split_label}\"")
    (args.out / "README.md").write_text(build_readme(records, skipped, cmd, args.report, args.extras))

    print(f"make_lab_figures: wrote {len(records)} figure(s) to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
