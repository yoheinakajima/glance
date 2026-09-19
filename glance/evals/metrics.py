"""Metrics over prediction rows (HANDOFF section 8). Pure numpy; nothing here touches a model.

A prediction row carries: type, label_index (noul: 0/1), z (logits), raw and calibrated probabilities
(noul: a float; choice/score: a list aligned with `keys`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..calibration import EPS, ece_equal_mass
from ..scorer import confidence as entropy_confidence
from ..scorer import expected_score, noul_confidence

COVERAGES = (0.5, 0.8, 0.9, 1.0)


def selective_accuracy(conf: np.ndarray, correct: np.ndarray, coverages=COVERAGES) -> dict[str, float]:
    """Accuracy on the most confident fraction of items, for each coverage level."""
    order = np.argsort(-np.asarray(conf, dtype=np.float64), kind="stable")
    ranked = np.asarray(correct, dtype=np.float64)[order]
    out = {}
    for cov in coverages:
        k = max(1, int(round(cov * len(ranked))))
        out[f"{int(cov * 100)}"] = float(ranked[:k].mean())
    return out


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coverage grid and the error rate (risk) among the most confident items at each coverage."""
    order = np.argsort(-np.asarray(conf, dtype=np.float64), kind="stable")
    ranked = np.asarray(correct, dtype=np.float64)[order]
    counts = np.arange(1, len(ranked) + 1)
    return counts / len(ranked), 1.0 - np.cumsum(ranked) / counts


def reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> list[dict[str, float]]:
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    order = np.argsort(conf, kind="stable")
    bins = []
    for idx in np.array_split(order, min(n_bins, len(conf))):
        if len(idx):
            bins.append({"confidence": float(conf[idx].mean()), "accuracy": float(correct[idx].mean()), "n": int(len(idx))})
    return bins


def auroc(p: np.ndarray, y: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y).astype(int)
    if len(set(y.tolist())) < 2:
        return None
    return float(roc_auc_score(y, p))


def macro_f1(pred: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y, pred, average="macro", zero_division=0))


def _views(rows: list[dict[str, Any]], field: str) -> dict[str, np.ndarray]:
    """Top-label confidence, ranking confidence, correctness and predictions for one probability field."""
    qtype = rows[0]["type"]
    labels = np.array([int(r["label_index"]) for r in rows])
    if qtype == "noul":
        p = np.array([float(r[field]) for r in rows])
        pred = (p >= 0.5).astype(int)
        return {
            "labels": labels, "pred": pred, "correct": pred == labels, "p": p,
            "top_conf": np.maximum(p, 1 - p), "rank_conf": np.array([noul_confidence(v) for v in p]),
        }
    probs = [np.asarray(r[field], dtype=np.float64) for r in rows]
    pred = np.array([int(np.argmax(q)) for q in probs])
    return {
        "labels": labels, "pred": pred, "correct": pred == labels, "probs": probs,
        "top_conf": np.array([q.max() for q in probs]), "rank_conf": np.array([entropy_confidence(q) for q in probs]),
    }


def probability_metrics(rows: list[dict[str, Any]], field: str, n_bins: int = 15) -> dict[str, Any]:
    """Accuracy, NLL, Brier, ECE, selective accuracy (+ AUROC, macro-F1 or MAE by type) for `raw` or `calibrated`."""
    if not rows:
        return {}
    qtype = rows[0]["type"]
    v = _views(rows, field)
    out: dict[str, Any] = {"n": len(rows), "accuracy": float(v["correct"].mean())}
    if qtype == "noul":
        p = np.clip(v["p"], EPS, 1 - EPS)
        y = v["labels"].astype(np.float64)
        out["auroc"] = auroc(v["p"], v["labels"])
        out["nll"] = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        out["brier"] = float(((v["p"] - y) ** 2).mean())
    else:
        out["nll"] = float(-np.mean([np.log(max(q[y], EPS)) for q, y in zip(v["probs"], v["labels"])]))
        out["brier"] = float(np.mean([((q - np.eye(len(q))[y]) ** 2).sum() for q, y in zip(v["probs"], v["labels"])]))
        if qtype == "choice":
            out["macro_f1"] = macro_f1(v["pred"], v["labels"])
        else:
            scores = np.array([expected_score(q) for q in v["probs"]])
            out["mae_levels"] = float(np.abs(scores - v["labels"]).mean())
    out["ece"] = ece_equal_mass(v["top_conf"], v["correct"], n_bins)
    out["selective_accuracy"] = selective_accuracy(v["rank_conf"], v["correct"])
    return out


def hard_pick_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Frontier baseline rows carry only `correct`: accuracy is all there is."""
    if not rows:
        return {}
    return {"n": len(rows), "accuracy": float(np.mean([bool(r["correct"]) for r in rows]))}


def curves(rows: list[dict[str, Any]], field: str, n_bins: int = 15) -> dict[str, Any]:
    v = _views(rows, field)
    coverage, risk = risk_coverage(v["rank_conf"], v["correct"])
    return {"reliability": reliability_bins(v["top_conf"], v["correct"], n_bins), "coverage": coverage, "risk": risk}


def latency_stats(latencies_ms: list[float]) -> dict[str, float] | None:
    if not latencies_ms:
        return None
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {"p50_ms": float(np.percentile(arr, 50)), "p95_ms": float(np.percentile(arr, 95)), "n": int(len(arr))}


def throughput(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Latency p50/p95 per request, statements per second, mean image tokens, off_mass mean and p95."""
    lat = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    passes = sum(r.get("forward_passes") or 0 for r in rows)
    out: dict[str, Any] = {"latency": latency_stats(lat)}
    out["statements_per_second"] = float(passes / (sum(lat) / 1000)) if lat and sum(lat) > 0 else None
    tokens = [r["image_tokens"] for r in rows if r.get("image_tokens")]
    out["mean_image_tokens"] = float(np.mean(tokens)) if tokens else None
    off = [v for r in rows for v in (r.get("off_mass") or [])]
    out["off_mass_mean"] = float(np.mean(off)) if off else None
    out["off_mass_p95"] = float(np.percentile(off, 95)) if off else None
    return out


def confusion_patterns(rows: list[dict[str, Any]], field: str, top: int = 3) -> list[dict[str, Any]]:
    """Most common (true -> predicted) mistakes, by key."""
    from collections import Counter

    counter: Counter = Counter()
    for r in rows:
        if r["type"] == "noul":
            pred = int(float(r[field]) >= 0.5)
            names = ["no", "yes"]
        else:
            pred = int(np.argmax(r[field]))
            names = r["keys"]
        if pred != int(r["label_index"]):
            counter[(names[int(r["label_index"])], names[pred])] += 1
    return [{"true": t, "predicted": p, "count": c} for (t, p), c in counter.most_common(top)]


def top_confident_errors(rows: list[dict[str, Any]], field: str, top: int = 20) -> list[dict[str, Any]]:
    if not rows:
        return []
    v = _views(rows, field)
    wrong = [i for i in np.argsort(-v["rank_conf"], kind="stable") if not v["correct"][i]][:top]
    out = []
    for i in wrong:
        r = rows[i]
        names = ["no", "yes"] if r["type"] == "noul" else r["keys"]
        out.append({
            "item_id": r["item_id"], "image_path": r["image_path"], "true": names[int(r["label_index"])],
            "predicted": names[int(v["pred"][i])], "confidence": float(v["rank_conf"][i]),
            "top_probability": float(v["top_conf"][i]),
        })
    return out
