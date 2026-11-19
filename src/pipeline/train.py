import json
from dataclasses import asdict
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import settings
from src.pipeline.features import FEATURE_COLUMNS, build_longitudinal_datasets
from src.pipeline.evaluate import (
    find_optimal_f2_threshold,
    compute_classification_metrics,
    evaluate_quality_gate,
    save_quality_gate_report,
)


class HeuristicRiskBaseline(BaseEstimator, ClassifierMixin):
    """Baseline deterministico baseado na regra atual de corte da ONG (INDE < 6.8 ou Quartzo)."""

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        inde = X["inde"].fillna(7.0).values
        pedra = X["pedra_ord"].fillna(3.0).values
        score = np.clip((7.5 - inde) / 3.0, 0.05, 0.95)
        score = np.where((inde < 6.8) | (pedra <= 1.0), np.maximum(score, 0.65), score)
        return np.vstack([1.0 - score, score]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def get_candidate_models(seed: int):
    return {
        "baseline_heuristic": HeuristicRiskBaseline(),
        "baseline_logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=seed)),
        ]),
        "candidate_random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            )),
        ]),
        "candidate_hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.04,
            max_depth=3,
            min_samples_leaf=15,
            l2_regularization=2.0,
            class_weight="balanced",
            random_state=seed,
        ),
    }


def run_training_pipeline():
    splits, _ = build_longitudinal_datasets()
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    X_train, y_train = train_df[FEATURE_COLUMNS].astype("float64"), train_df["target"].values
    X_val, y_val = val_df[FEATURE_COLUMNS].astype("float64"), val_df["target"].values
    X_test, y_test = test_df[FEATURE_COLUMNS].astype("float64"), test_df["target"].values

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    models = get_candidate_models(settings.RANDOM_SEED)
    results = {}
    trained_pipelines = {}

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name) as run:
            model.fit(X_train, y_train)
            val_proba = model.predict_proba(X_val)[:, 1]
            opt_thr = 0.50 if model_name == "baseline_heuristic" else find_optimal_f2_threshold(y_val, val_proba)
            val_metrics = compute_classification_metrics(y_val, val_proba, opt_thr)

            test_proba = model.predict_proba(X_test)[:, 1]
            test_metrics = compute_classification_metrics(y_test, test_proba, opt_thr)

            mlflow.set_tag("model_role", "baseline" if "baseline" in model_name else "candidate")
            mlflow.set_tag("train_window", "2022->2023")
            mlflow.set_tag("val_window", "2023->2024")
            mlflow.set_tag("test_oot_window", "2024->2025")
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("optimal_threshold_f2", opt_thr)
            mlflow.log_param("n_features", len(FEATURE_COLUMNS))

            for k, v in asdict(val_metrics).items():
                mlflow.log_metric(f"val_{k}", v)
            for k, v in asdict(test_metrics).items():
                mlflow.log_metric(f"test_oot_{k}", v)

            signature = infer_signature(X_train, model.predict_proba(X_train))
            if model_name == "baseline_heuristic":
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    signature=signature,
                    input_example=X_train.head(3), serialization_format="cloudpickle",
                )
            else:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    signature=signature,
                    input_example=X_train.head(3), serialization_format="cloudpickle",
                )

            results[model_name] = {
                "run_id": run.info.run_id,
                "val": val_metrics,
                "test_oot": test_metrics,
            }
            trained_pipelines[model_name] = model

    candidates = [k for k in results if k.startswith("candidate_") or k == "baseline_logistic_regression"]
    champion_name = max(candidates, key=lambda k: results[k]["val"].pr_auc)
    champion_info = results[champion_name]
    baseline_info = results["baseline_heuristic"]

    approved, gate_checks = evaluate_quality_gate(champion_info["val"], baseline_info["val"])

    settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    gate_report = {
        "champion_model": champion_name,
        "champion_run_id": champion_info["run_id"],
        "approved": approved,
        "checks": gate_checks,
        "val_metrics": asdict(champion_info["val"]),
        "test_oot_metrics": asdict(champion_info["test_oot"]),
        "baseline_val_metrics": asdict(baseline_info["val"]),
    }
    save_quality_gate_report(gate_report, settings.ARTIFACTS_DIR / "quality_gate_report.json")

    if approved:
        bundle = {
            "model": trained_pipelines[champion_name],
            "model_name": champion_name,
            "model_version": settings.MODEL_VERSION,
            "threshold": champion_info["val"].threshold,
            "feature_columns": FEATURE_COLUMNS,
            "run_id": champion_info["run_id"],
            "val_metrics": asdict(champion_info["val"]),
            "test_oot_metrics": asdict(champion_info["test_oot"]),
        }
        joblib.dump(bundle, settings.ARTIFACTS_DIR / "champion_model.joblib")

    return results, gate_report


if __name__ == "__main__":
    results, gate_report = run_training_pipeline()
    print("\n================ COMPARATIVO DE MODELOS (VALIDACAO 2023->2024 | TESTE OOT 2024->2025) ================")
    for name, info in results.items():
        v, t = info["val"], info["test_oot"]
        print(
            f"{name:34s} | VAL PR-AUC={v.pr_auc:.4f} ROC-AUC={v.roc_auc:.4f} F2={v.f2:.4f} Rec={v.recall:.4f} Prec={v.precision:.4f} (thr={v.threshold}) "
            f"|| OOT PR-AUC={t.pr_auc:.4f} ROC-AUC={t.roc_auc:.4f} Rec={t.recall:.4f}"
        )
    print("\n================ QUALITY GATE REPORT ================")
    print(json.dumps(gate_report, indent=2, ensure_ascii=False))
