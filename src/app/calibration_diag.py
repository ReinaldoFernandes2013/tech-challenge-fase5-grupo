from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def _to_log_odds(probs: np.ndarray) -> np.ndarray:
    """Converte probabilidades em [0, 1] para log-odds com clip seguro (Platt scaling)."""
    clipped = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def compute_calibration_diagnostics(
    y_true: np.ndarray,
    y_proba_raw: np.ndarray,
    n_bins: int = 8,
    y_fit_true: Optional[np.ndarray] = None,
    y_fit_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Calcula Brier Score, curva de confiabilidade e comparacao Isotonic/Platt."""
    y_t = np.asarray(y_true, dtype=int).ravel()
    p_raw = np.clip(np.asarray(y_proba_raw, dtype=float).ravel(), 0.0, 1.0)

    if y_t.size == 0 or y_t.shape[0] != p_raw.shape[0]:
        return {
            "available": False,
            "reason": "Vetores de rotulos e probabilidades vazios ou com tamanhos incompativeis.",
            "brier_raw": None,
            "brier_isotonic": None,
            "brier_platt": None,
            "curve_df": pd.DataFrame(),
            "summary_df": pd.DataFrame(),
        }

    brier_raw = float(round(brier_score_loss(y_t, p_raw), 4))

    fit_y = y_t if y_fit_true is None else np.asarray(y_fit_true, dtype=int).ravel()
    fit_p = p_raw if y_fit_proba is None else np.clip(np.asarray(y_fit_proba, dtype=float).ravel(), 0.0, 1.0)

    brier_iso: Optional[float] = None
    brier_platt: Optional[float] = None

    if len(np.unique(fit_y)) >= 2 and fit_y.shape[0] >= 10:
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(fit_p, fit_y)
        p_iso = np.clip(iso.predict(p_raw), 0.0, 1.0)
        brier_iso = float(round(brier_score_loss(y_t, p_iso), 4))

        platt = LogisticRegression(solver="lbfgs")
        platt.fit(_to_log_odds(fit_p), fit_y)
        p_platt = np.clip(platt.predict_proba(_to_log_odds(p_raw))[:, 1], 0.0, 1.0)
        brier_platt = float(round(brier_score_loss(y_t, p_platt), 4))

    bins_eff = min(n_bins, max(2, len(np.unique(p_raw))))
    prob_true, prob_pred = calibration_curve(y_t, p_raw, n_bins=bins_eff, strategy="quantile")
    curve_df = pd.DataFrame({
        "Probabilidade Prevista Média": np.round(prob_pred, 4),
        "Frequência Observada (Positivos)": np.round(prob_true, 4),
        "Diferença (Observado - Previsto)": np.round(prob_true - prob_pred, 4),
    })

    summary_rows = [
        {
            "Versão de Probabilidade": "Probabilidade Bruta (Modelo Campeão)",
            "Brier Score (menor = melhor)": brier_raw,
        }
    ]
    if brier_iso is not None:
        summary_rows.append({
            "Versão de Probabilidade": "Pós-Calibrada — Isotonic Regression",
            "Brier Score (menor = melhor)": brier_iso,
        })
    if brier_platt is not None:
        summary_rows.append({
            "Versão de Probabilidade": "Pós-Calibrada — Platt Scaling (Log-Odds)",
            "Brier Score (menor = melhor)": brier_platt,
        })

    return {
        "available": True,
        "reason": None,
        "brier_raw": brier_raw,
        "brier_isotonic": brier_iso,
        "brier_platt": brier_platt,
        "curve_df": curve_df,
        "summary_df": pd.DataFrame(summary_rows),
    }
