import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import settings
from src.api.schemas import StudentIndicatorsInput
from src.api.main import load_champion_bundle, build_feature_row

st.set_page_config(
    page_title="Passos Mágicos | MLOps & Apoio Psicopedagógico",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Associação Passos Mágicos — Sistema de Apoio Psicopedagógico (MLOps)")
st.info(
    "**Diretriz de Uso Responsável (Human-in-the-Loop):** Este sistema fornece sinais estatísticos "
    "de priorização para apoiar a equipe psicopedagógica na identificação preventiva de estudantes que podem "
    "precisar de acompanhamento adicional no próximo ciclo letivo. **As previsões nunca devem ser tratadas "
    "como rótulos definitivos nem substituir a avaliação humana multidisciplinar.**"
)

tab1, tab2, tab3 = st.tabs([
    "🔍 Simulador & Triagem Individual (XAI)",
    "🛡️ Quality Gate & Governança MLflow",
    "📡 Observabilidade & Drift (PSI 2022–2025)",
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
