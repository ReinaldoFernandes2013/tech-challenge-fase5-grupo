import json
from src.config import settings


def test_quality_gate_approved():
    report_path = settings.ARTIFACTS_DIR / "quality_gate_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["approved"] is True
    assert report["val_metrics"]["pr_auc"] >= settings.MIN_PR_AUC
    assert report["val_metrics"]["recall"] >= settings.MIN_RECALL


def test_drift_detector_flags_2025_restructuring():
    drift_path = settings.ARTIFACTS_DIR / "drift_audit_report.json"
    assert drift_path.exists()
    drift = json.loads(drift_path.read_text(encoding="utf-8"))
    assert drift["oot_2024_2025"]["retrain_triggered"] is True
