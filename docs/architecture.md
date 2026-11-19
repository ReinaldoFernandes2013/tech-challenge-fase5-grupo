# Arquitetura do Pipeline MLOps — Associação Passos Mágicos

## Visão Geral
Pipeline ponta a ponta de previsão prospectiva ($t \rightarrow t+1$) de necessidade de acompanhamento psicopedagógico na **Associação Passos Mágicos**, estruturado para reprodutibilidade local (Python 3.12, MLflow, FastAPI, Streamlit, Docker, GitHub Actions) e execução em nuvem (Databricks Free Edition / Delta Lake).

## Camadas do Sistema
1. **Ingestão & Validação (`src/pipeline/data_validation.py`)**: leitura da base longitudinal PEDE (`2022–2025`), normalização de esquema e auditoria de valores ausentes e faixas de indicadores.
2. **Engenharia de Features Anti-Leakage (`src/pipeline/features.py`)**: construção estrita de features observadas no ciclo $t$ para prever o desfecho em $t+1$, evitando vazamento temporal.
3. **Treinamento, Baseline & Quality Gate (`src/pipeline/train.py`, `src/pipeline/evaluate.py`, `src/pipeline/register.py`)**: comparação contra baseline heurístico, calibração de limiar via F2-Score e promoção condicional no MLflow Model Registry.
4. **Serving & Interface (`src/api/main.py`, `src/app/streamlit_app.py`)**: API REST (`FastAPI`) com validação (`Pydantic`), auditoria de inferência e aplicação interativa (`Streamlit`) com XAI e suporte a *Human-in-the-Loop*.
5. **Observabilidade (`src/monitoring/drift_detector.py`)**: cálculo de *Population Stability Index* (PSI) e estatística KS entre ciclos para acionar alertas de investigação ou retreinamento.
