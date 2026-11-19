import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from src.config import settings
from src.pipeline.features import FEATURE_COLUMNS


def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Calcula o Population Stability Index (PSI) conforme formulacao canonica (Aula 03):
      PSI = sum((Actual_i - Expected_i) * ln(Actual_i / Expected_i))
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 10 or len(actual) < 10:
        return 0.0

    quantiles = np.linspace(0, 100, buckets + 1)
    breakpoints = np.unique(np.percentile(expected, quantiles))
    if len(breakpoints) <= 2:
        return 0.0

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    exp_counts = np.histogram(expected, bins=breakpoints)[0]
    act_counts = np.histogram(actual, bins=breakpoints)[0]

    eps = 1e-4
    exp_pct = np.clip(exp_counts / len(expected), eps, 1.0)
    act_pct = np.clip(act_counts / len(actual), eps, 1.0)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(round(psi, 4))


def classify_psi_status(psi_value: float) -> str:
    if psi_value < settings.PSI_MODERATE_THRESHOLD:
        return "ESTAVEL"
    if psi_value < settings.PSI_SEVERE_THRESHOLD:
        return "MODERADO_INVESTIGAR"
    return "SEVERO_RETREINAR"


def run_drift_audit(reference_df: pd.DataFrame, current_df: pd.DataFrame, window_label: str) -> Dict:
    feature_reports = {}
    severe_features = []
    moderate_features = []

    for col in FEATURE_COLUMNS:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        exp_arr = reference_df[col].to_numpy(dtype=float)
        act_arr = current_df[col].to_numpy(dtype=float)
        psi_val = calculate_psi(exp_arr, act_arr)
        status = classify_psi_status(psi_val)

        exp_clean = exp_arr[~np.isnan(exp_arr)]
        act_clean = act_arr[~np.isnan(act_arr)]
        ks_stat, ks_pval = (0.0, 1.0)
        if len(exp_clean) > 5 and len(act_clean) > 5:
            ks_res = ks_2samp(exp_clean, act_clean)
            ks_stat, ks_pval = float(round(ks_res.statistic, 4)), float(round(ks_res.pvalue, 4))

        feature_reports[col] = {
            "psi": psi_val,
            "status": status,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pval,
            "ref_mean": float(round(np.nanmean(exp_arr), 3)),
            "cur_mean": float(round(np.nanmean(act_arr), 3)),
        }
        if status == "SEVERO_RETREINAR":
            severe_features.append(col)
        elif status == "MODERADO_INVESTIGAR":
            moderate_features.append(col)

    retrain_triggered = len(severe_features) >= 2
    return {
        "window": window_label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "retrain_triggered": retrain_triggered,
        "severe_drift_count": len(severe_features),
        "moderate_drift_count": len(moderate_features),
        "severe_features": severe_features,
        "moderate_features": moderate_features,
        "features": feature_reports,
    }


def log_prediction_event(payload: dict, response: dict, log_path: Path = None) -> None:
    target_path = log_path or (settings.ARTIFACTS_DIR / "inference_audit_log.jsonl")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input": payload,
        "output": response,
    }
    with open(target_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ref = pd.read_parquet(settings.REFERENCE_DIR / "reference_train.parquet")
    val = pd.read_parquet(settings.PROCESSED_DIR / "val_2023_2024.parquet")
    test = pd.read_parquet(settings.PROCESSED_DIR / "test_2024_2025.parquet")

    rep_val = run_drift_audit(ref, val, "2023->2024 (Validacao)")
    rep_test = run_drift_audit(ref, test, "2024->2025 (Pos-Reestruturacao Fases)")

    full_drift = {"val_2023_2024": rep_val, "oot_2024_2025": rep_test}
    settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(settings.ARTIFACTS_DIR / "drift_audit_report.json", "w", encoding="utf-8") as f:
        json.dump(full_drift, f, indent=2, ensure_ascii=False)

    print(f"[Drift 2023->2024] Retrain Trigger={rep_val['retrain_triggered']} | Severas ({rep_val['severe_drift_count']}): {rep_val['severe_features']}")
    print(f"[Drift 2024->2025] Retrain Trigger={rep_test['retrain_triggered']} | Severas ({rep_test['severe_drift_count']}): {rep_test['severe_features']}")
