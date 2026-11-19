from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    PROJECT_NAME: str = "Passos Mágicos - Predição de Acompanhamento Educacional (MLOps)"
    MODEL_NAME: str = "passos_magicos_risk_classifier"
    MODEL_VERSION: str = "1.0.0"
    MODEL_STAGE: str = "Production"
    RANDOM_SEED: int = 42

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    RAW_DATA_PATH: Path = BASE_DIR / "data" / "raw" / "BASE DE DADOS PEDE - DATATHON.xlsx"
    PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    REFERENCE_DIR: Path = BASE_DIR / "data" / "reference"
    ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"

    # Quality Gate Thresholds (Requisito 6)
    MIN_PR_AUC: float = 0.42
    MIN_ROC_AUC: float = 0.72
    MIN_RECALL: float = 0.65
    MIN_MARGIN_OVER_BASELINE: float = 0.03

    # Drift Thresholds (PSI - Aula 03)
    PSI_MODERATE_THRESHOLD: float = 0.10
    PSI_SEVERE_THRESHOLD: float = 0.25

    # Databricks / MLflow
    DATABRICKS_HOST: str = ""
    DATABRICKS_TOKEN: str = ""
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    MLFLOW_EXPERIMENT_NAME: str = "passos-magicos-mlops-fase5"
    DATABRICKS_SERVING_ENDPOINT: str = ""


settings = Settings()
