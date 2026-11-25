import re
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from src.config import settings
from src.pipeline.data_validation import normalize_text, validate_cohort_contract, ValidationReport


FEATURE_COLUMNS = [
    "fase_num",
    "idade",
    "anos_na_ong",
    "defasagem",
    "inde",
    "iaa",
    "ieg",
    "ips",
    "ida",
    "ipv",
    "ian",
    "mat",
    "por",
    "ing",
    "tem_nota_ing",
    "pedra_ord",
    "pedra_anterior_ord",
    "delta_pedra_1y",
    "delta_inde_1y",
    "gap_iaa_ida",
    "gap_ieg_ida",
    "media_psico",
    "escola_publica",
]

PEDRA_MAP = {
    "QUARTZO": 1,
    "AGATA": 2,
    "AMETISTA": 3,
    "TOPAZIO": 4,
}


def parse_fase_numerica(fase_val, year: int) -> float:
    """
    Normaliza Fase considerando a reestruturacao oficial de 2025
    documentada em 'PEDE_ Pontos importantes - 2025.pdf' (pagina 7).
    """
    txt = normalize_text(fase_val)
    if "ALFA" in txt:
        return 0.0
    nums = re.findall(r"\d+", txt)
    if not nums:
        return np.nan
    num = float(nums[0])
    if year == 2025:
        # Equivalencia oficial 2025 -> escala historica 0..9
        mapping_2025 = {
            1: 0.5, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0,
            6: 5.5, 7: 6.5, 8: 7.0, 9: 7.0, 10: 7.0,
            11: 7.5, 12: 8.0, 13: 9.0,
        }
        return mapping_2025.get(int(num), num)
    return num


def standardize_year_sheet(raw_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Padroniza schema heterogeneo entre abas PEDE2022..PEDE2025 sem PII."""
    df = raw_df.copy()
    df["RA"] = df["RA"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["RA"]).copy()

    inde_col = "INDE 22" if year == 2022 else f"INDE {year}"
    pedra_col = "Pedra 22" if year == 2022 else f"Pedra {year}"
    pedra_prev_col = f"Pedra {str(year - 1)[-2:]}"
    inde_prev_col = f"INDE {str(year - 1)[-2:]}" if year >= 2023 else None

    out = pd.DataFrame()
    out["RA"] = df["RA"]
    out["ano"] = year
    out["fase_num"] = df["Fase"].apply(lambda x: parse_fase_numerica(x, year))

    idade_col = "Idade 22" if year == 2022 else "Idade"
    out["idade"] = pd.to_numeric(df[idade_col], errors="coerce")
    out["ano_ingresso"] = pd.to_numeric(df["Ano ingresso"], errors="coerce")
    out["anos_na_ong"] = (year - out["ano_ingresso"]).clip(lower=0)

    defas_col = "Defas" if year == 2022 else "Defasagem"
    out["defasagem"] = pd.to_numeric(df[defas_col], errors="coerce")

    out["INDE"] = pd.to_numeric(df[inde_col], errors="coerce")
    out["IAA"] = pd.to_numeric(df["IAA"], errors="coerce")
    out["IEG"] = pd.to_numeric(df["IEG"], errors="coerce")
    out["IPS"] = pd.to_numeric(df["IPS"], errors="coerce")
    out["IPP"] = pd.to_numeric(df["IPP"], errors="coerce") if "IPP" in df.columns else np.nan
    out["IDA"] = pd.to_numeric(df["IDA"], errors="coerce")
    out["IPV"] = pd.to_numeric(df["IPV"], errors="coerce")
    out["IAN"] = pd.to_numeric(df["IAN"], errors="coerce")

    mat_col = "Matem" if year == 2022 else "Mat"
    por_col = "Portug" if year == 2022 else "Por"
    ing_col = "Inglês" if year == 2022 else "Ing"
    out["mat"] = pd.to_numeric(df[mat_col], errors="coerce")
    out["por"] = pd.to_numeric(df[por_col], errors="coerce")
    out["ing"] = pd.to_numeric(df[ing_col], errors="coerce")

    out["Pedra"] = df[pedra_col].apply(normalize_text)
    out["pedra_ord"] = out["Pedra"].map(PEDRA_MAP)

    if pedra_prev_col in df.columns:
        out["pedra_anterior_ord"] = df[pedra_prev_col].apply(normalize_text).map(PEDRA_MAP)
    else:
        out["pedra_anterior_ord"] = np.nan

    if inde_prev_col and inde_prev_col in df.columns:
        out["inde_anterior"] = pd.to_numeric(df[inde_prev_col], errors="coerce")
    else:
        out["inde_anterior"] = np.nan

    gen_norm = df["Gênero"].apply(normalize_text)
    out["genero_bin"] = np.where(gen_norm.isin(["MENINA", "FEMININO"]), 1, 0)

    inst_norm = df["Instituição de ensino"].apply(normalize_text)
    out["escola_publica"] = np.where(inst_norm.str.contains("PUBLICA"), 1, 0)

    return out


def engineer_features(df_t: pd.DataFrame) -> pd.DataFrame:
    """
    Engenharia de features deterministica e sem vazamento (usada no treino e serving).
    Recebe apenas atributos observados ate o final do ano t.
    """
    X = pd.DataFrame(index=df_t.index)
    X["fase_num"] = df_t["fase_num"]
    X["idade"] = df_t["idade"]
    X["anos_na_ong"] = df_t["anos_na_ong"]
    X["defasagem"] = df_t["defasagem"]
    X["inde"] = df_t["INDE"]
    X["iaa"] = df_t["IAA"]
    X["ieg"] = df_t["IEG"]
    X["ips"] = df_t["IPS"]
    X["ipp"] = df_t["IPP"]
    X["ida"] = df_t["IDA"]
    X["ipv"] = df_t["IPV"]
    X["ian"] = df_t["IAN"]
    X["mat"] = df_t["mat"]
    X["por"] = df_t["por"]
    X["ing"] = df_t["ing"]
    X["tem_nota_ing"] = df_t["ing"].notna().astype(int)
    X["pedra_ord"] = df_t["pedra_ord"]
    X["pedra_anterior_ord"] = df_t["pedra_anterior_ord"]
    X["delta_pedra_1y"] = (X["pedra_ord"] - X["pedra_anterior_ord"]).fillna(0.0)
    X["inde_anterior"] = df_t["inde_anterior"]
    X["delta_inde_1y"] = (X["inde"] - X["inde_anterior"]).fillna(0.0)
    # Features de dominio psicopedagogico:
    # 1) Superestimacao da autoavaliacao frente ao desempenho real (IAA - IDA)
    X["gap_iaa_ida"] = X["iaa"] - X["ida"]
    # 2) Descompasso entre engajamento de tarefas e resultado academico (IEG - IDA)
    X["gap_ieg_ida"] = X["ieg"] - X["ida"]
    # 3) Indice composto psicossocial-pedagogico
    X["media_psico"] = np.nanmean(np.vstack([X["ips"], X["ipp"].fillna(X["ips"]), X["ipv"]]), axis=0)
    X["escola_publica"] = df_t["escola_publica"]
    return X[FEATURE_COLUMNS].astype("float64")


def build_longitudinal_datasets() -> Tuple[Dict[str, pd.DataFrame], Dict[int, ValidationReport]]:
    """
    Constroi os pares longitudinais t -> t+1 sem leakage:
      - train: 2022 -> 2023
      - val:   2023 -> 2024
      - test:  2024 -> 2025
    Target Y_{t+1} = 1 se Pedra_{t+1} == 'QUARTZO' ou INDE_{t+1} < 6.5
    """
    xl = pd.ExcelFile(settings.RAW_DATA_PATH)
    cohorts = {}
    reports = {}

    for year in [2022, 2023, 2024, 2025]:
        raw = pd.read_excel(xl, f"PEDE{year}")
        std = standardize_year_sheet(raw, year)
        report = validate_cohort_contract(std, year)
        if not report.passed:
            raise ValueError(f"Contrato de dados falhou para {year}: {report.details}")
        cohorts[year] = std
        reports[year] = report

    splits = {}
    split_specs = [
        ("train", 2022, 2023),
        ("val", 2023, 2024),
        ("test", 2024, 2025),
    ]

    settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    settings.REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, y_t, y_t1 in split_specs:
        df_t = cohorts[y_t].copy()
        df_t1 = cohorts[y_t1][["RA", "INDE", "IPV", "Pedra"]].rename(
            columns={"INDE": "inde_t1", "IPV": "ipv_t1", "Pedra": "pedra_t1"}
        )
        merged = pd.merge(df_t, df_t1, on="RA", how="inner")
        merged = merged.dropna(subset=["INDE", "inde_t1", "pedra_t1"]).copy()

        X = engineer_features(merged)
        y = ((merged["pedra_t1"] == "QUARTZO") | (merged["inde_t1"] < 6.5)).astype(int)

        dataset = X.copy()
        dataset["target"] = y.values
        dataset["transition"] = f"{y_t}->{y_t1}"
        # RA anonimizado apenas para auditoria de grupo/painel, nunca entra como feature
        dataset["ra_id"] = merged["RA"].values

        out_path = settings.PROCESSED_DIR / f"{split_name}_{y_t}_{y_t1}.parquet"
        dataset.to_parquet(out_path, index=False)
        splits[split_name] = dataset

    # Salva baseline de referencia (Treino 2022->2023) para deteccao de Data Drift (PSI)
    splits["train"][FEATURE_COLUMNS + ["target"]].to_parquet(
        settings.REFERENCE_DIR / "reference_train.parquet", index=False
    )
    return splits, reports


if __name__ == "__main__":
    splits, reports = build_longitudinal_datasets()
    for yr, rep in reports.items():
        print(f"[Data Contract {yr}] Passed={rep.passed} | Rows={rep.total_rows} | Checks={rep.checks}")
    for name, df in splits.items():
        print(f"[Split {name.upper()}] Shape={df.shape} | Target Positive Rate={df['target'].mean():.2%}")
