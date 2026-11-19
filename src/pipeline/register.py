import json
import mlflow
from mlflow.tracking import MlflowClient
from src.config import settings


def promote_approved_model():
    """
    Requisito 5 e 6: Registra no MLflow Model Registry apenas se aprovado no Quality Gate
    e transiciona o estagio para Production.
    """
    report_path = settings.ARTIFACTS_DIR / "quality_gate_report.json"
    if not report_path.exists():
        raise FileNotFoundError("Relatorio do Quality Gate nao encontrado. Execute train.py primeiro.")

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    if not report["approved"]:
        raise RuntimeError(f"Promocao bloqueada pelo Quality Gate: {report['checks']}")

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    run_id = report["champion_run_id"]
    model_uri = f"runs:/{run_id}/model"

    mv = mlflow.register_model(model_uri=model_uri, name=settings.MODEL_NAME)
    client.update_model_version(
        name=settings.MODEL_NAME,
        version=mv.version,
        description=(
            f"Modelo {report['champion_model']} aprovado no Quality Gate. "
            f"Val PR-AUC={report['val_metrics']['pr_auc']}, "
            f"OOT PR-AUC={report['test_oot_metrics']['pr_auc']}, "
            f"Threshold F2={report['val_metrics']['threshold']}."
        ),
    )
    client.transition_model_version_stage(
        name=settings.MODEL_NAME,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"[Model Registry] Modelo '{settings.MODEL_NAME}' v{mv.version} (Run {run_id}) promovido para PRODUCTION!")
    return mv


if __name__ == "__main__":
    promote_approved_model()
