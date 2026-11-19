import json
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    brier_score_loss,
    confusion_matrix,
)
from src.config import settings


@dataclass
class ModelMetrics:
    pr_auc: float
    roc_auc: float
    f1: float
    f2: float
    precision: float
    recall: float
    brier_score: float
    threshold: float
    tn: int
    fp: int
    fn: int
    tp: int


def find_optimal_f2_threshold(y_true: np.ndarray, y_proba: np.ndarray, min_precision: float = 0.30) -> float:
    """
    Seleciona threshold que maximiza F2-Score (prioriza Recall para nao deixar jovens
    em risco sem acompanhamento), garantindo precisao minima operacional.
    """
    best_thr = 0.50
    best_f2 = -1.0
    for thr in np.linspace(0.15, 0.80, 66):
        preds = (y_proba >= thr).astype(int)
        prec = precision_score(y_true, preds, zero_division=0)
        if prec < min_precision:
            continue
        score = fbeta_score(y_true, preds, beta=2.0, zero_division=0)
        if score > best_f2:
            best_f2 = score
            best_thr = float(round(thr, 4))
    return best_thr


def compute_classification_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> ModelMetrics:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return ModelMetrics(
        pr_auc=float(round(average_precision_score(y_true, y_proba), 4)),
        roc_auc=float(round(roc_auc_score(y_true, y_proba), 4)),
        f1=float(round(f1_score(y_true, y_pred, zero_division=0), 4)),
        f2=float(round(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0), 4)),
        precision=float(round(precision_score(y_true, y_pred, zero_division=0), 4)),
        recall=float(round(recall_score(y_true, y_pred, zero_division=0), 4)),
        brier_score=float(round(brier_score_loss(y_true, y_proba), 4)),
        threshold=float(round(threshold, 4)),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def evaluate_quality_gate(
    candidate_metrics: ModelMetrics,
    baseline_metrics: ModelMetrics,
) -> Tuple[bool, Dict[str, dict]]:
    """
    Requisito 6: Quality Gate objetivo e auditavel antes da promocao do modelo.
    """
    gain_pr_auc = round(candidate_metrics.pr_auc - baseline_metrics.pr_auc, 4)
    checks = {
        "min_pr_auc": {
            "passed": candidate_metrics.pr_auc >= settings.MIN_PR_AUC,
            "actual": candidate_metrics.pr_auc,
            "required": settings.MIN_PR_AUC,
        },
        "min_roc_auc": {
            "passed": candidate_metrics.roc_auc >= settings.MIN_ROC_AUC,
            "actual": candidate_metrics.roc_auc,
            "required": settings.MIN_ROC_AUC,
        },
        "min_recall": {
            "passed": candidate_metrics.recall >= settings.MIN_RECALL,
            "actual": candidate_metrics.recall,
            "required": settings.MIN_RECALL,
        },
        "beats_heuristic_baseline_pr_auc": {
            "passed": gain_pr_auc >= settings.MIN_MARGIN_OVER_BASELINE,
            "actual_gain": gain_pr_auc,
            "required_margin": settings.MIN_MARGIN_OVER_BASELINE,
        },
    }
    approved = all(item["passed"] for item in checks.values())
    return approved, checks


def save_quality_gate_report(report: dict, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
