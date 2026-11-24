# Passos Mágicos — Pipeline MLOps Longitudinal (PEDE 2022–2025)

Pipeline de Machine Learning e engenharia MLOps aplicado aos dados longitudinais da **Associação Passos Mágicos (PEDE, 2022–2025)**, estruturado com **anti-leakage** estrito ($t \rightarrow t+1$), validação *out-of-time*, teste estatístico não paramétrico **KS**, auditoria de *missingness*, **MLflow Registry**, **Quality Gate**, **FastAPI**, **Streamlit** e monitoramento de **drift** via **PSI**.

---

## 1. Hipótese Científica e Desenho Anti-Leakage

* **Objetivo:** Estimar no ciclo $t$ a necessidade de acompanhamento educacional/psicopedagógico no ciclo subsequente ($t+1$), apoiando a atuação preventiva da equipe pedagógica com abordagem *Human-in-the-Loop*.
* **Prevenção de Data Leakage ($t \rightarrow t+1$):** Divisão cronológica estrita (*out-of-time*), impedindo que informações futuras ou transformações ajustadas em toda a série temporal contaminem treino ou avaliação.
* **Reestruturação PEDE 2025 e Inferência Estatística:** Mudanças metodológicas entre ciclos podem gerar *domain shift* e ausência não aleatória (*MAR/MNAR*). O pipeline audita mudanças usando estatística descritiva, teste de Kolmogorov-Smirnov (`ks_2samp`) e *Population Stability Index* (**PSI**).

---

## Trajetória Analítica até a Fase 5

A construção desta solução apoia-se na evolução progressiva ao longo das fases do programa, unindo rigor analítico, modelagem, aprendizagem profunda/NLP e engenharia de MLOps:

* **Fase 1 — Estruturação analítica e risco com CRISP-DM:** base metodológica de entendimento do problema, preparação de dados e modelagem de risco (`payflow-risk-model-crispdm`) com foco em validação e disciplina analítica.
* **Fase 2 — Pipelines e dados educacionais:** amadurecimento de fluxos de processamento e avaliação em contexto educacional (`pipeline-hibrido-alfabetizacao-brasil`).
* **Fase 3 — Machine Learning e MLOps:** consolidação de práticas de treino, avaliação, registo e operacionalização de modelos.
* **Fase 4 — Deep Learning and NLP:** aprofundamento em arquiteturas neurais e processamento de linguagem natural, ampliando a leitura de sinais complexos e não lineares.
* **Fase 5 — MLOps Longitudinal na Passos Mágicos (PEDE 2022–2025):** aplicação de desenho *anti-leakage* ($t \rightarrow t+1$), validação *out-of-time*, análise de *drift* (KS e PSI, incluindo os impactos da reestruturação de 2025), governação com *MLflow Registry* e *Quality Gate*, serviço em *FastAPI*, visualização interativa em *Streamlit* (com apoio *Human-in-the-Loop* e análise por janela e `escola_publica`) e automação de testes/qualidade.

## 2. Estrutura do Repositório

* `.github/workflows/`: integração contínua (`lint` e testes automatizados).
* `artifacts/`: modelo campeão (`.joblib`), relatórios de *quality gate* e *drift*, e logs de inferência.
* `data/raw/`, `data/processed/`, `data/reference/`: base bruta, splits temporais em Parquet e base de referência para *drift*.
* `docs/`: *Architecture*, *Data Card*, *Model Card* e *Monitoring Plan*.
* `notebooks/`: cadernos `01`, `02` e `03` (EDA/qualidade, features/experimentos e registry/quality gate/serving).
* `src/api/`, `src/app/`, `src/monitoring/`, `src/pipeline/`: API FastAPI, app Streamlit, detector de *drift* e pipeline ML.
* `tests/`: suíte de testes automatizados com `Pytest`.

---

## 3. Como Executar Localmente

### Ambiente e testes

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
PYTHONPATH=. pytest -v tests/
```

```Shell
PYTHONPATH=. uvicorn src.api.main:app --reload
PYTHONPATH=. streamlit run src.app/streamlit_app.py
```

Os três notebooks em `notebooks/` (`01_eda_and_data_quality.ipynb`,`02_feature_engineering_and_experiments.ipynb` e `03_registry_quality_gate_and_serving.ipynb`) estão preparados para execução local e para o  **Databricks Free Edition** , mantendo a lógica centralizada em `src/`.
## 4. Governança, Métricas e Limitações

* **Métricas de Avaliação:** Foco em PR-AUC, ROC-AUC, Recall e F2-Score diante do desbalanceamento de classes e do custo assimétrico de falsos negativos.

Quality Gate e MLflow: Promoção condicionada a critérios objetivos frente ao baseline heurístico, com registro de métricas e artefatos em artifacts/.

Limitações e Human-in-the-Loop: Mudanças estruturais (como a reestruturação de 2025) podem provocar drift severo (
PSI
≥
0.25
PSI≥0.25) e perda de calibração probabilística. O modelo apoia a triagem humana (Human-in-the-Loop), sem substituir avaliação pedagógica/psicossocial nem dispensar retreinamento ou recalibração periódicos.
