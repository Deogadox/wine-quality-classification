"""Carregamento da base e criação da variável alvo binária."""
import pandas as pd

from src.config import DATA_FILE, QUALITY_COL, QUALITY_THRESHOLD, TARGET_COL


def load_raw(path=DATA_FILE) -> pd.DataFrame:
    """Lê o CSV original (aceita separador ',' ou ';' e BOM UTF-8).

    Se usar o WineQT.csv do Kaggle, a coluna 'Id' é descartada automaticamente.
    """
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df.drop(columns=["Id"], errors="ignore")


def add_binary_target(df: pd.DataFrame, threshold: int = QUALITY_THRESHOLD) -> pd.DataFrame:
    """Cria a coluna alvo: 1 = alta qualidade (nota >= threshold), 0 = baixa/média."""
    out = df.copy()
    out[TARGET_COL] = (out[QUALITY_COL] >= threshold).astype(int)
    return out


def load_dataset() -> pd.DataFrame:
    """Atalho: base original + alvo binário."""
    return add_binary_target(load_raw())
