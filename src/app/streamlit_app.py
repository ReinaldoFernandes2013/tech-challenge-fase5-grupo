import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import requests
import plotly.express as px
import streamlit as st

from src.config import settings
from src.api.schemas import StudentIndicatorsInput
from src.api.main import load_champion_bundle, build_feature_row
from src.app.xai_local import compute_local_shap_explanation
from src.app.calibration_diag import compute_calibration_diagnostics

st.set_page_config(
    page_title="Passos Mágicos | MLOps & Apoio Psicopedagógico",
    page_icon="🎓",
    layout="wide",
)

if "ui_theme" not in st.session_state:
    st.session_state["ui_theme"] = "Night Mode"

_, theme_col = st.columns([8, 2])
with theme_col:
    selected_theme = st.radio(
        "Modo de exibição",
        options=["Night Mode", "Light Mode"],
        index=0 if st.session_state["ui_theme"] == "Night Mode" else 1,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state["ui_theme"] = selected_theme

is_night = st.session_state["ui_theme"] == "Night Mode"
plotly_template = "plotly_dark" if is_night else "plotly_white"

if is_night:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        [data-testid="stMetric"] {
            background-color: #161b22;
            border: 1px solid #30363d;
            padding: 12px;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f8fafc;
            color: #0f172a;
        }
        .stApp p, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
            color: #0f172a;
        }
        [data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #cbd5e1;
            padding: 12px;
            border-radius: 8px;
        }
        [data-testid="stPlotlyChart"] text,
        [data-testid="stPlotlyChart"] .gtitle,
        [data-testid="stPlotlyChart"] .xtitle,
        [data-testid="stPlotlyChart"] .ytitle,
        [data-testid="stPlotlyChart"] .legendtext,
        [data-testid="stPlotlyChart"] .annotation-text {
            fill: #0f172a !important;
            color: #0f172a !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

_orig_plotly_chart = st.plotly_chart


def _themed_plotly_chart(fig, *args, **kwargs):
    fg = "#f8fafc" if is_night else "#0f172a"
    grid = "rgba(248, 250, 252, 0.14)" if is_night else "rgba(15, 23, 42, 0.14)"
    try:
        fig.layout.template = plotly_template
        fig.layout.paper_bgcolor = "rgba(0,0,0,0)"
        fig.layout.plot_bgcolor = "rgba(0,0,0,0)"
        fig.layout.font.color = fg
        if fig.layout.title and fig.layout.title.font:
            fig.layout.title.font.color = fg
        if fig.layout.legend and fig.layout.legend.font:
            fig.layout.legend.font.color = fg
        if fig.layout.legend and fig.layout.legend.title and fig.layout.legend.title.font:
            fig.layout.legend.title.font.color = fg
        fig.update_xaxes(
            color=fg,
            linecolor=fg,
            tickcolor=fg,
            gridcolor=grid,
            title_font=dict(color=fg),
            tickfont=dict(color=fg),
        )
        fig.update_yaxes(
            color=fg,
            linecolor=fg,
            tickcolor=fg,
            gridcolor=grid,
            title_font=dict(color=fg),
            tickfont=dict(color=fg),
        )
        for ann in fig.layout.annotations or []:
            if ann.font:
                ann.font.color = fg
            else:
                ann.font = dict(color=fg)
    except Exception:
        pass
    return _orig_plotly_chart(fig, *args, **kwargs)


st.plotly_chart = _themed_plotly_chart

st.title("🎓 Associação Passos Mágicos — Sistema de Apoio Psicopedagógico (MLOps)")
st.info(
    "**Diretriz de Uso Responsável (Human-in-the-Loop):** Este sistema fornece sinais estatísticos "
    "de priorização para apoiar a equipe psicopedagógica na identificação preventiva de estudantes que podem "
    "precisar de acompanhamento adicional no próximo ciclo letivo. **As previsões nunca devem ser tratadas "
    "como rótulos definitivos nem substituir a avaliação humana multidisciplinar.**"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Simulador & Triagem Individual (XAI)",
    "🛡️ Quality Gate & Governança MLflow",
    "📡 Observabilidade & Drift (PSI 2022–2025)",
    "📊 Panorama Cohort & Escola (2022–2025)",
])

with tab1:
    st.subheader("Simulação Prospectiva de Risco Educacional (Ciclo t → t+1)")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Perfil & Trajetória**")
        fase_num = st.slider("Fase Atual Normalizada (0=ALFA .. 9)", 0.0, 9.0, 3.0, 0.5)
        idade = st.slider("Idade (anos)", 6, 22, 12)
        anos_na_ong = st.slider("Anos na Passos Mágicos", 0, 10, 2)
        defasagem = st.slider("Defasagem Idade-Série", -4, 2, -1)
        pedra_label = st.selectbox("Pedra Atual (Ano t)", ["Quartzo (1)", "Ágata (2)", "Ametista (3)", "Topázio (4)"], index=1)
        pedra_ord = float(pedra_label.split("(")[1].replace(")", ""))

    with col2:
        st.markdown("**Indicadores PEDE (0 a 10)**")
        inde = st.slider("INDE (Índice Desenvolvimento Educacional)", 0.0, 10.0, 6.4, 0.1)
        ida = st.slider("IDA (Desempenho Acadêmico)", 0.0, 10.0, 5.8, 0.1)
        ieg = st.slider("IEG (Engajamento)", 0.0, 10.0, 6.5, 0.1)
        ipv = st.slider("IPV (Ponto de Virada)", 0.0, 10.0, 6.2, 0.1)
        ian = st.slider("IAN (Adequação de Nível)", 0.0, 10.0, 5.0, 0.5)

    with col3:
        st.markdown("**Dimensão Psicossocial & Notas**")
        ips = st.slider("IPS (Psicossocial)", 0.0, 10.0, 6.0, 0.1)
        iaa = st.slider("IAA (Autoavaliação)", 0.0, 10.0, 8.0, 0.1)
        mat = st.slider("Nota Matemática", 0.0, 10.0, 5.5, 0.1)
        por = st.slider("Nota Português", 0.0, 10.0, 6.0, 0.1)
        delta_inde_1y = st.slider("Variação do INDE no último ano (Δ INDE)", -4.0, 4.0, -0.4, 0.1)

    if st.button("Avaliar Prioridade de Acompanhamento", type="primary"):
        payload = StudentIndicatorsInput(
            fase_num=fase_num,
            idade=float(idade),
            anos_na_ong=float(anos_na_ong),
            defasagem=float(defasagem),
            inde=inde,
            iaa=iaa,
            ieg=ieg,
            ips=ips,
            ida=ida,
            ipv=ipv,
            ian=ian,
            mat=mat,
            por=por,
            pedra_ord=pedra_ord,
            delta_inde_1y=delta_inde_1y,
        )
        bundle = load_champion_bundle()
        X_row = build_feature_row(payload)
        api_url = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
        used_api = False
        try:
            resp = requests.post(f"{api_url}/predict", json=payload.model_dump(), timeout=3)
            if resp.ok:
                api_data = resp.json()
                proba = float(api_data["probabilidade_risco"])
                thr = float(api_data["threshold_aplicado"])
                used_api = True
            else:
                proba = float(bundle["model"].predict_proba(X_row)[0, 1])
                thr = float(bundle["threshold"])
        except Exception:
            proba = float(bundle["model"].predict_proba(X_row)[0, 1])
            thr = float(bundle["threshold"])

        m1, m2, m3 = st.columns(3)
        m1.metric("Probabilidade Estimada de Risco (t+1)", f"{proba:.1%}")
        m2.metric("Limiar Calibrado (F2-Score)", f"{thr:.1%}")
        m3.metric("Modelo em Produção", f"{bundle['model_name']} v{bundle['model_version']}")

        if proba >= thr:
            st.warning(
                "⚠️ **Sinal de Atenção Preventiva:** Estudante apresenta indicadores compatíveis com "
                "priorização para acolhimento psicopedagógico no próximo ciclo."
            )
        else:
            st.success(
                "✅ **Acompanhamento Regular:** Indicadores dentro da faixa esperada de continuidade."
            )

        clf = bundle["model"].named_steps["clf"]
        imp_df = pd.DataFrame({
            "Variável": bundle["feature_columns"],
            "Importância Global": clf.feature_importances_,
        }).sort_values("Importância Global", ascending=True).tail(10)
        fig = px.bar(
            imp_df,
            x="Importância Global",
            y="Variável",
            orientation="h",
            title="Top 10 Indicadores Determinantes no Modelo Campeão (Random Forest)",
        )
        st.plotly_chart(fig, use_container_width=True)

        shap_res = compute_local_shap_explanation(bundle, X_row, top_n=10)
        st.markdown("### Explicabilidade Local da Simulação Atual (SHAP)")
        if shap_res["available"] and not shap_res["df"].empty:
            st.caption(
                f"Valor base esperado (classe positiva): `{shap_res['base_value']}` | "
                "Valores SHAP positivos aumentam o sinal de atenção (t+1); negativos reduzem."
            )
            shap_df = shap_res["df"].sort_values("Impacto SHAP", ascending=True)
            fig_local = px.bar(
                shap_df,
                x="Impacto SHAP",
                y="Variavel",
                color="Direcao",
                orientation="h",
                hover_data=["Valor na Simulacao"],
                title="Contribuição Individual das Variáveis para Este Estudante",
            )
            fig_local.add_vline(x=0.0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_local, use_container_width=True)
            st.dataframe(
                shap_res["df"][["Variavel", "Valor na Simulacao", "Impacto SHAP", "Direcao"]],
                use_container_width=True,
            )
        else:
            st.info(f"Explicabilidade local indisponível no momento: {shap_res['reason']}")

with tab2:
    st.subheader("Auditoria de Promoção — Quality Gate Automatizado (Requisito 6)")
    qg_path = settings.ARTIFACTS_DIR / "quality_gate_report.json"
    if qg_path.exists():
        qg = json.loads(qg_path.read_text(encoding="utf-8"))
        st.success(f"Status do Quality Gate: **APROVADO** | Modelo Campeão: `{qg['champion_model']}` (Run `{qg['champion_run_id']}`)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Critérios Verificados")
            st.json(qg["checks"])
        with c2:
            st.markdown("#### Métricas de Validação (2023→2024) vs. Baseline")
            comp_df = pd.DataFrame([
                {"Modelo": "Baseline Heurístico (Regra Atual ONG)", **qg["baseline_val_metrics"]},
                {"Modelo": f"Campeão ({qg['champion_model']})", **qg["val_metrics"]},
            ])[["Modelo", "pr_auc", "roc_auc", "f2", "recall", "precision", "brier_score"]]
            st.dataframe(comp_df, use_container_width=True)

        st.markdown("---")
        st.subheader("Calibração de Probabilidades & Reliability Diagram (2023→2024 vs. OOT 2024→2025)")
        val_fp = ROOT_DIR / "data" / "processed" / "val_2023_2024.parquet"
        tst_fp = ROOT_DIR / "data" / "processed" / "test_2024_2025.parquet"
        if val_fp.exists() and tst_fp.exists():
            try:
                bundle_cal = load_champion_bundle()
                feat_cols = bundle_cal["feature_columns"]
                val_df = pd.read_parquet(val_fp)
                tst_df = pd.read_parquet(tst_fp)
                p_val_raw = bundle_cal["model"].predict_proba(val_df[feat_cols])[:, 1]
                p_tst_raw = bundle_cal["model"].predict_proba(tst_df[feat_cols])[:, 1]
                cal_res = compute_calibration_diagnostics(
                    y_true=tst_df["target"].to_numpy(),
                    y_proba_raw=p_tst_raw,
                    n_bins=8,
                    y_fit_true=val_df["target"].to_numpy(),
                    y_fit_proba=p_val_raw,
                )
                if cal_res["available"] and not cal_res["curve_df"].empty:
                    c_left_cal, c_right_cal = st.columns([1, 1.4])
                    with c_left_cal:
                        st.markdown("#### Comparativo de Brier Score (Teste OOT 2024→2025)")
                        st.dataframe(cal_res["summary_df"], use_container_width=True)
                        st.caption(
                            "Calibradores ajustados em `val_2023_2024` e avaliados fora da amostra em `test_2024_2025`, "
                            "evidenciando o efeito da calibração pós-treino e do shift de 2025."
                        )
                    with c_right_cal:
                        curve_df = cal_res["curve_df"].copy()
                        fig_cal = px.line(
                            curve_df,
                            x="Probabilidade Prevista Média",
                            y=["Frequência Observada (Positivos)", "Probabilidade Prevista Média"],
                            markers=True,
                            title="Reliability Diagram (Curva de Confiabilidade — Teste OOT 2024→2025)",
                        )
                        st.plotly_chart(fig_cal, use_container_width=True)
                    st.dataframe(cal_res["curve_df"], use_container_width=True)
                else:
                    st.info(f"Diagnóstico de calibração indisponível: {cal_res['reason']}")
            except Exception as exc:
                st.info(f"Não foi possível calcular o diagnóstico de calibração nesta execução: {exc}")
        else:
            st.info("Arquivos de validação/teste OOT não encontrados em `data/processed/` para o Reliability Diagram.")

with tab3:
    st.subheader("Monitoramento de Estabilidade Populacional (PSI) e Reestruturação 2025")
    drift_path = settings.ARTIFACTS_DIR / "drift_audit_report.json"
    if drift_path.exists():
        drift = json.loads(drift_path.read_text(encoding="utf-8"))
        oot = drift["oot_2024_2025"]
        st.error(
            f"🚨 **Gatilho de Retreinamento Acionado em 2024→2025:** {oot['severe_drift_count']} features "
            f"ultrapassaram PSI ≥ 0.25 ({', '.join(oot['severe_features'])}), refletindo a reestruturação oficial "
            "das fases e da metodologia Casa Mágica (PEDE 2025)."
        )
        rows = []
        for feat, vals in oot["features"].items():
            rows.append({
                "Feature": feat,
                "PSI (2024→2025)": vals["psi"],
                "Status": vals["status"],
                "KS Statistic": vals["ks_statistic"],
                "Média Ref (2022)": vals["ref_mean"],
                "Média Atual (2024)": vals["cur_mean"],
            })
        psi_df = pd.DataFrame(rows).sort_values("PSI (2024→2025)", ascending=False)
        fig_psi = px.bar(
            psi_df.head(12),
            x="Feature",
            y="PSI (2024→2025)",
            color="Status",
            title="Population Stability Index (PSI) por Feature — Janela 2024→2025 vs. Referência",
        )
        fig_psi.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="Limiar Severo (0.25)")
        fig_psi.add_hline(y=0.10, line_dash="dot", line_color="orange", annotation_text="Limiar Moderado (0.10)")
        st.plotly_chart(fig_psi, use_container_width=True)
        st.dataframe(psi_df, use_container_width=True)

with tab4:
    st.subheader("Panorama por Tipo de Escola e Janela Temporal (2022–2025)")
    st.caption(
        "Análise descritiva comparativa entre estudantes de escola pública e não pública nas janelas "
        "temporais processadas, sem inferência geográfica artificial."
    )

    split_specs = [
        ("2022→2023 (Treino)", ROOT_DIR / "data" / "processed" / "train_2022_2023.parquet"),
        ("2023→2024 (Validação)", ROOT_DIR / "data" / "processed" / "val_2023_2024.parquet"),
        ("2024→2025 (Teste OOT)", ROOT_DIR / "data" / "processed" / "test_2024_2025.parquet"),
    ]

    frames = []
    for label, fp in split_specs:
        if fp.exists():
            df_split = pd.read_parquet(fp).copy()
            df_split["janela"] = label
            frames.append(df_split)

    if not frames:
        st.warning("Nenhum arquivo Parquet encontrado em `data/processed/` para montar o panorama.")
    else:
        cohort_df = pd.concat(frames, ignore_index=True)

        if "escola_publica" in cohort_df.columns:
            cohort_df["tipo_escola"] = cohort_df["escola_publica"].map({
                1.0: "Escola Pública",
                1: "Escola Pública",
                0.0: "Não Pública",
                0: "Não Pública",
            }).fillna("Não Informado")
        else:
            cohort_df["tipo_escola"] = "Não Informado"

        metric_candidates = [
            c for c in ["target", "inde", "ian", "defasagem", "ieg", "ips", "ida", "ipv"]
            if c in cohort_df.columns
        ]

        if not metric_candidates:
            st.warning("Nenhuma coluna numérica de indicador/target disponível nos splits processados.")
        else:
            selected_metric = st.selectbox(
                "Indicador para comparativo por janela e tipo de escola",
                options=metric_candidates,
                index=0,
            )

            agg_df = (
                cohort_df.groupby(["janela", "tipo_escola"], as_index=False)
                .agg(
                    n_estudantes=("tipo_escola", "count"),
                    valor_medio=(selected_metric, "mean"),
                )
            )

            c_left, c_right = st.columns(2)
            with c_left:
                fig_bar = px.bar(
                    agg_df,
                    x="janela",
                    y="valor_medio",
                    color="tipo_escola",
                    barmode="group",
                    text_auto=".2f",
                    title=f"Média de `{selected_metric}` por Janela e Tipo de Escola",
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with c_right:
                fig_vol = px.bar(
                    agg_df,
                    x="janela",
                    y="n_estudantes",
                    color="tipo_escola",
                    barmode="group",
                    text_auto=True,
                    title="Volume de Estudantes por Janela e Tipo de Escola",
                )
                st.plotly_chart(fig_vol, use_container_width=True)

            heat_cols = [c for c in ["target", "inde", "ian", "defasagem", "ieg", "ips"] if c in cohort_df.columns]
            if heat_cols:
                cohort_df["estrato"] = cohort_df["janela"] + " | " + cohort_df["tipo_escola"]
                heat_df = cohort_df.groupby("estrato", as_index=True)[heat_cols].mean()
                fig_heat = px.imshow(
                    heat_df,
                    text_auto=".2f",
                    aspect="auto",
                    title="Matriz de Médias dos Indicadores por Janela e Tipo de Escola",
                )
                st.plotly_chart(fig_heat, use_container_width=True)

            st.dataframe(agg_df, use_container_width=True)
