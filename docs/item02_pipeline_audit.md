# Auditoria Tecnica — Item 02: Pipeline de Dados e Features

## 1. Resumo da Execucao e Artefatos
- Datasets longitudinais gerados em `data/processed/` (`train_2022_2023.parquet`, `val_2023_2024.parquet`, `test_2024_2025.parquet`) e baseline em `data/reference/reference_train.parquet`.
- Notebooks `01_eda_and_data_quality.ipynb`, `02_feature_engineering_and_experiments.ipynb` e `03_registry_quality_gate_and_serving.ipynb` sincronizados com as assinaturas reais de `src/` e executados com `nbconvert`.
- Modelo campeao selecionado pelo Quality Gate: `candidate_random_forest` (`run_id`: `38efb3ba73ae49e38c401de10f20b82b`, `approved: true`), salvo em `artifacts/champion_model.joblib` e registrado no MLflow Model Registry (`passos_magicos_risk_classifier`).

## 2. Analise de Performance: Validacao (2023->2024) vs. Out-of-Time (2024->2025)
- **Validacao (`2023->2024`):** `PR-AUC = 0.4994`, `ROC-AUC = 0.8220`, `Recall = 0.7833` (threshold `0.38`), superando o baseline heuristico em `+0.0787` de `PR-AUC`.
- **Teste Out-of-Time (`2024->2025`):** `PR-AUC = 0.2298`, `ROC-AUC = 0.5508`, `Recall = 0.4622` (threshold `0.38`).
- **Diagnostico:** a queda na janela `2024->2025` reflete mudanca de regime / reestruturacao das fases em 2025 e drift distributivo severo em variaveis-chave (`anos_na_ong`, `defasagem`, `ips`, `ing`), reforçando a necessidade de retreino/recalibracao quando novas coortes entram em producao.

## 3. Escopo de `FEATURE_COLUMNS` e Colunas Intermediarias
- `FEATURE_COLUMNS` define a lista oficial de 24 features utilizadas no treinamento (`src/pipeline/train.py`) e empacotadas no bundle `artifacts/champion_model.joblib`.
- Colunas como `ipp` e `inde_anterior` participam do calculo de variaveis derivadas em `engineer_features` (`media_psico` e `delta_inde_1y`), mas permanecem fora de `FEATURE_COLUMNS` por decisao explicita de esquema.

## 4. Ambiente de Execucao (Local vs. Databricks)
- Execucao validada localmente no ambiente virtual `.venv` com tracking/registry em SQLite (`sqlite:///mlflow.db`) e artefatos locais em `artifacts/` e `data/`.
- As configuracoes de Databricks em `src/config.py` (`DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `DATABRICKS_SERVING_ENDPOINT`) estao vazias no ambiente local atual.
