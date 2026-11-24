from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.api.main import build_feature_row, load_champion_bundle
from src.api.schemas import StudentIndicatorsInput
from src.app import xai_local
from src.app.calibration_diag import compute_calibration_diagnostics
from src.app.xai_local import _extract_positive_class_shap, compute_local_shap_explanation


def test_extract_positive_class_shap_3d_and_list_formats():
    shap_3d = np.array([
        [[-0.12, 0.12], [0.05, -0.05], [0.0, 0.0]],
    ])
    row_3d, base_3d = _extract_positive_class_shap(shap_3d, np.array([0.6, 0.4]), n_features=3)
    assert np.allclose(row_3d, np.array([0.12, -0.05, 0.0]))
    assert np.isclose(base_3d, 0.4)

    shap_list = [
        np.array([[-0.2, 0.1]]),
        np.array([[0.2, -0.1]]),
    ]
    row_list, base_list = _extract_positive_class_shap(shap_list, np.array([[0.7, 0.3]]), n_features=2)
    assert np.allclose(row_list, np.array([0.2, -0.1]))
    assert np.isclose(base_list, 0.3)


def test_compute_local_shap_with_champion_bundle():
    champion_path = Path("artifacts/champion_model.joblib")
    if not champion_path.exists():
        pytest.skip("Artefato artifacts/champion_model.joblib nao encontrado neste ambiente.")

    bundle = load_champion_bundle()
    payload = StudentIndicatorsInput(
        fase_num=3.0,
        idade=12.0,
        anos_na_ong=2.0,
        defasagem=-1.0,
        inde=6.4,
        iaa=8.0,
        ieg=6.5,
        ips=6.0,
        ida=5.8,
        ipv=6.2,
        ian=5.0,
        mat=5.5,
        por=6.0,
        pedra_ord=2.0,
        delta_inde_1y=-0.4,
    )
    x_row = build_feature_row(payload)
    res = compute_local_shap_explanation(bundle, x_row, top_n=5)

    assert res["available"] is True
    assert res["reason"] is None
    assert isinstance(res["base_value"], float)
    assert len(res["df"]) <= 5
    expected_cols = {"Variavel", "Valor na Simulacao", "Impacto SHAP", "Direcao", "abs_shap"}
    assert expected_cols.issubset(set(res["df"].columns))
    assert res["df"]["abs_shap"].is_monotonic_decreasing


def test_compute_local_shap_fallback_when_invalid_model_or_missing_shap(monkeypatch):
    invalid_bundle = {"model": object(), "feature_columns": ["inde", "mat"]}
    x_df = pd.DataFrame([{"inde": 6.0, "mat": 5.0}])
    res_invalid = compute_local_shap_explanation(invalid_bundle, x_df, top_n=5)
    assert res_invalid["available"] is False
    assert res_invalid["df"].empty

    monkeypatch.setattr(xai_local, "shap", None)
    valid_like_bundle = {"model": object(), "feature_columns": ["inde"]}
    res_no_shap = compute_local_shap_explanation(valid_like_bundle, pd.DataFrame([{"inde": 6.0}]))
    assert res_no_shap["available"] is False
    assert "shap" in res_no_shap["reason"].lower()


def test_compute_calibration_diagnostics_valid_and_edge_cases():
    rng = np.random.default_rng(42)
    y_fit = rng.integers(0, 2, size=80)
    p_fit = np.clip(0.2 + 0.6 * y_fit + rng.normal(0, 0.12, size=80), 0.01, 0.99)

    y_eval = rng.integers(0, 2, size=60)
    p_eval = np.clip(0.2 + 0.6 * y_eval + rng.normal(0, 0.12, size=60), 0.01, 0.99)

    res = compute_calibration_diagnostics(
        y_true=y_eval,
        y_proba_raw=p_eval,
        n_bins=5,
        y_fit_true=y_fit,
        y_fit_proba=p_fit,
    )
    assert res["available"] is True
    assert 0.0 <= res["brier_raw"] <= 1.0
    assert res["brier_isotonic"] is not None and 0.0 <= res["brier_isotonic"] <= 1.0
    assert res["brier_platt"] is not None and 0.0 <= res["brier_platt"] <= 1.0
    assert len(res["summary_df"]) == 3
    curve = res["curve_df"]
    assert not curve.empty
    assert np.allclose(
        curve["Diferença (Observado - Previsto)"].to_numpy(),
        (curve["Frequência Observada (Positivos)"] - curve["Probabilidade Prevista Média"]).to_numpy(),
        atol=1e-4,
    )

    res_empty = compute_calibration_diagnostics(np.array([]), np.array([]))
    assert res_empty["available"] is False

    res_single_class = compute_calibration_diagnostics(
        y_true=np.array([0, 0, 1, 0, 1, 0]),
        y_proba_raw=np.array([0.1, 0.2, 0.8, 0.3, 0.7, 0.25]),
        n_bins=3,
        y_fit_true=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        y_fit_proba=np.linspace(0.5, 0.9, 10),
    )
    assert res_single_class["available"] is True
    assert res_single_class["brier_raw"] is not None
    assert res_single_class["brier_isotonic"] is None
    assert res_single_class["brier_platt"] is None
