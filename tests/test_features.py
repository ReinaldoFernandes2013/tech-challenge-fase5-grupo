import pandas as pd
from src.config import settings
from src.pipeline.features import FEATURE_COLUMNS, parse_fase_numerica


def test_no_target_leakage_in_feature_columns():
    forbidden = {"target", "inde_t1", "ipv_t1", "pedra_t1", "ra_id", "transition"}
    assert forbidden.isdisjoint(set(FEATURE_COLUMNS))


def test_parse_fase_2025_restructuring():
    assert parse_fase_numerica("ALFA", 2025) == 0.0
    assert parse_fase_numerica("Fase 11 (Casa Mágica)", 2025) == 7.5
    assert parse_fase_numerica("Fase 13", 2025) == 9.0


def test_processed_splits_exist_and_match_schema():
    for fname in ["train_2022_2023.parquet", "val_2023_2024.parquet", "test_2024_2025.parquet"]:
        df = pd.read_parquet(settings.PROCESSED_DIR / fname)
        for col in FEATURE_COLUMNS + ["target"]:
            assert col in df.columns
