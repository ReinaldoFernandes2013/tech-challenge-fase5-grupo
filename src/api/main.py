import time
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.config import settings
from src.pipeline.features import FEATURE_COLUMNS
from src.api.schemas import StudentIndicatorsInput, PredictionResponse
from src.monitoring.drift_detector import log_prediction_event


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.MODEL_VERSION,
    description=(
        "API de apoio a decisao psicopedagogica da Associacao Passos Magicos. "
        "Estima a probabilidade de necessidade de acompanhamento adicional no proximo ciclo letivo."
    ),
)


@lru_cache(maxsize=1)
def load_champion_bundle():
    model_path = settings.ARTIFACTS_DIR / "champion_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Artefato do modelo nao encontrado em {model_path}")
    return joblib.load(model_path)


def build_feature_row(payload: StudentIndicatorsInput) -> pd.DataFrame:
    tem_nota_ing = 1.0 if payload.ing is not None else 0.0
    pedra_ant = payload.pedra_anterior_ord if payload.pedra_anterior_ord is not None else np.nan
    delta_pedra = (payload.pedra_ord - payload.pedra_anterior_ord) if payload.pedra_anterior_ord is not None else 0.0
    gap_iaa_ida = payload.iaa - payload.ida
    gap_ieg_ida = payload.ieg - payload.ida
    media_psico = float(np.mean([payload.ips, payload.ipv]))

    row = {
        "fase_num": payload.fase_num,
        "idade": payload.idade,
        "anos_na_ong": payload.anos_na_ong,
        "defasagem": payload.defasagem,
        "inde": payload.inde,
        "iaa": payload.iaa,
        "ieg": payload.ieg,
        "ips": payload.ips,
        "ida": payload.ida,
        "ipv": payload.ipv,
        "ian": payload.ian,
        "mat": payload.mat,
        "por": payload.por,
        "ing": payload.ing if payload.ing is not None else np.nan,
        "tem_nota_ing": tem_nota_ing,
        "pedra_ord": payload.pedra_ord,
        "pedra_anterior_ord": pedra_ant,
        "delta_pedra_1y": delta_pedra,
        "delta_inde_1y": payload.delta_inde_1y,
        "gap_iaa_ida": gap_iaa_ida,
        "gap_ieg_ida": gap_ieg_ida,
        "media_psico": media_psico,
        "escola_publica": payload.escola_publica,
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS].astype("float64")


@app.get("/health")
def health_check():
    bundle = load_champion_bundle()
    return {
        "status": "healthy",
        "model_name": bundle["model_name"],
        "model_version": bundle["model_version"],
        "run_id": bundle["run_id"],
        "threshold": bundle["threshold"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_student_risk(payload: StudentIndicatorsInput):
    start = time.perf_counter()
    try:
        bundle = load_champion_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    X_row = build_feature_row(payload)
    model = bundle["model"]
    threshold = float(bundle["threshold"])
    proba = float(model.predict_proba(X_row)[0, 1])
    flag = proba >= threshold

    if proba >= 0.65:
        prioridade = "ALTA (Priorizar acolhimento psicopedagogico)"
    elif flag:
        prioridade = "MODERADA (Acompanhamento preventivo recomendado)"
    else:
        prioridade = "BAIXA (Manter acompanhamento regular)"

    # Extrai importancias globais do RandomForest para explicabilidade transparente
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    top_factors = []
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        top_idx = np.argsort(importances)[::-1][:5]
        for idx in top_idx:
            fname = FEATURE_COLUMNS[idx]
            top_factors.append({
                "feature": fname,
                "importance": round(float(importances[idx]), 4),
                "student_value": round(float(X_row[fname].iloc[0]), 3) if not np.isnan(X_row[fname].iloc[0]) else -1.0,
            })

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    resp = PredictionResponse(
        necessita_acompanhamento_adicional=flag,
        probabilidade_risco=round(proba, 4),
        nivel_prioridade=prioridade,
        threshold_aplicado=threshold,
        principais_fatores=top_factors,
        aviso_uso_responsavel=(
            "Sinal estatistico de apoio a investigacao pedagogica. "
            "Nao substitui avaliacao humana multidisciplinar (Human-in-the-Loop)."
        ),
        model_name=bundle["model_name"],
        model_version=bundle["model_version"],
        latency_ms=latency_ms,
    )
    log_prediction_event(payload.model_dump(), resp.model_dump())
    return resp
