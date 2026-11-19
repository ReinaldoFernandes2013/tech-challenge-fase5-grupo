import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List
import pandas as pd


def normalize_text(val) -> str:
    if pd.isna(val):
        return "DESCONHECIDO"
    text = str(val).strip()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    return text.upper()


@dataclass
class ValidationReport:
    year: int
    total_rows: int
    unique_ras: int
    passed: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    details: List[str] = field(default_factory=list)


def validate_cohort_contract(df: pd.DataFrame, year: int) -> ValidationReport:
    """
    Contrato de Dados declarativo (inspirado em Great Expectations - Aula 03)
    Garante schema, unicidade de RA, completude e faixas plausiveis [0, 10].
    """
    checks = {}
    details = []

    required_cols = ["RA", "INDE", "IAA", "IEG", "IPS", "IDA", "IPV", "IAN", "defasagem", "Pedra"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    checks["schema_columns_present"] = len(missing_cols) == 0
    if missing_cols:
        details.append(f"Colunas ausentes em {year}: {missing_cols}")

    unique_ras = df["RA"].nunique()
    checks["ra_uniqueness"] = unique_ras == len(df)
    if not checks["ra_uniqueness"]:
        details.append(f"Duplicidade de RA em {year}: {len(df) - unique_ras} registros")

    # Faixas plausiveis [0, 10.5] para indicadores PEDE (tolerancia para arredondamentos como 10.002)
    range_ok = True
    for ind in ["INDE", "IAA", "IEG", "IPS", "IDA", "IPV", "IAN"]:
        valid = df[ind].dropna()
        if len(valid) > 0 and ((valid < 0).any() or (valid > 10.5).any()):
            range_ok = False
            details.append(f"Indicador {ind} fora do intervalo [0, 10.5] em {year}")
    checks["indicators_plausible_range"] = range_ok

    # Completude minima de 85% nos indicadores centrais
    completeness_ok = True
    for ind in ["INDE", "IEG", "IDA", "IPV", "IAN"]:
        null_rate = df[ind].isna().mean()
        if null_rate > 0.15:
            completeness_ok = False
            details.append(f"Taxa de nulos acima do SLA (15%) para {ind} em {year}: {null_rate:.1%}")
    checks["completeness_sla"] = completeness_ok

    passed = all(checks.values())
    return ValidationReport(
        year=year,
        total_rows=len(df),
        unique_ras=unique_ras,
        passed=passed,
        checks=checks,
        details=details,
    )
