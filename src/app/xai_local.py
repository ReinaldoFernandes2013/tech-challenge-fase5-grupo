from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None


def _extract_positive_class_shap(
    shap_values: Any,
    expected_value: Any,
    n_features: int,
) -> Tuple[np.ndarray, float]:
    """Normaliza saidas de diferentes versoes do SHAP para a classe positiva (indice 1)."""
    if hasattr(shap_values, "values"):
        vals = np.asarray(shap_values.values)
        base = getattr(shap_values, "base_values", expected_value)
    else:
        vals = shap_values
        base = expected_value

    if isinstance(vals, list):
        target_arr = np.asarray(vals[1] if len(vals) > 1 else vals[0])
        if target_arr.ndim == 2:
            row_shap = target_arr[0]
        else:
            row_shap = np.asarray(target_arr).reshape(-1)[:n_features]
    else:
        arr = np.asarray(vals)
        if arr.ndim == 3:
            class_idx = 1 if arr.shape[2] > 1 else 0
            row_shap = arr[0, :, class_idx]
        elif arr.ndim == 2:
            row_shap = arr[0, :n_features]
        else:
            row_shap = arr.reshape(-1)[:n_features]

    base_arr = np.asarray(base)
    if base_arr.ndim == 0:
        base_val = float(base_arr.item())
    elif base_arr.ndim == 1:
        idx = 1 if base_arr.shape[0] > 1 else 0
        base_val = float(base_arr[idx])
    elif base_arr.ndim == 2:
        idx = 1 if base_arr.shape[1] > 1 else 0
        base_val = float(base_arr[0, idx])
    else:
        base_val = 0.0

    return np.asarray(row_shap, dtype=float).reshape(-1)[:n_features], base_val


def compute_local_shap_explanation(
    bundle: Dict[str, Any],
    X_row: pd.DataFrame,
    top_n: int = 10,
) -> Dict[str, Any]:
    """Calcula explicabilidade local (SHAP) para uma unica linha de simulacao."""
    feature_columns = list(bundle.get("feature_columns") or list(X_row.columns))
    ordered_df = X_row.reindex(columns=feature_columns).copy()

    if shap is None:
        return {
            "available": False,
            "reason": "Biblioteca 'shap' nao disponivel no ambiente.",
            "base_value": None,
            "df": pd.DataFrame(
                columns=["Variavel", "Valor na Simulacao", "Impacto SHAP", "Direcao", "abs_shap"]
            ),
        }

    model = bundle.get("model")
    if model is None or not hasattr(model, "named_steps") or "clf" not in model.named_steps:
        return {
            "available": False,
            "reason": "Modelo nao possui etapa 'clf' compativel com TreeExplainer.",
            "base_value": None,
            "df": pd.DataFrame(
                columns=["Variavel", "Valor na Simulacao", "Impacto SHAP", "Direcao", "abs_shap"]
            ),
        }

    clf = model.named_steps["clf"]
    try:
        if hasattr(model, "steps") and len(model.steps) > 1:
            X_transformed = model[:-1].transform(ordered_df)
        else:
            X_transformed = ordered_df.to_numpy()

        explainer = shap.TreeExplainer(clf)
        raw_shap = explainer.shap_values(X_transformed)
        row_shap, base_val = _extract_positive_class_shap(
            raw_shap,
            getattr(explainer, "expected_value", 0.0),
            len(feature_columns),
        )

        rows = []
        for col, val, s_val in zip(feature_columns, ordered_df.iloc[0].tolist(), row_shap.tolist()):
            direction = (
                "Aumenta Risco (t+1)" if s_val > 0
                else ("Reduz Risco (t+1)" if s_val < 0 else "Neutro")
            )
            rows.append({
                "Variavel": col,
                "Valor na Simulacao": round(float(val), 4) if pd.notna(val) else None,
                "Impacto SHAP": round(float(s_val), 4),
                "Direcao": direction,
                "abs_shap": abs(float(s_val)),
            })

        exp_df = (
            pd.DataFrame(rows)
            .sort_values("abs_shap", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        return {
            "available": True,
            "reason": None,
            "base_value": round(base_val, 4),
            "df": exp_df,
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": f"Nao foi possivel calcular SHAP local: {exc}",
            "base_value": None,
            "df": pd.DataFrame(
                columns=["Variavel", "Valor na Simulacao", "Impacto SHAP", "Direcao", "abs_shap"]
            ),
        }
