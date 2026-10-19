"""Pré-processamento: limpeza, feature engineering, divisão treino/teste e pipeline de transformação.

Decisões (justificadas no notebook 02):
- Duplicatas exatas são removidas ANTES da divisão, para evitar que o mesmo vinho
  apareça no treino e no teste (vazamento de dados).
- Não há valores faltantes, mas o pipeline inclui imputação pela mediana por segurança
  (ex.: novos lotes com alguma medida ausente).
- Variáveis de cauda longa recebem logaritmo natural; em seguida, todas são padronizadas
  (média 0, desvio 1). Tudo é ajustado SOMENTE no treino, dentro do Pipeline.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import FEATURES, QUALITY_COL, RANDOM_STATE, TARGET_COL, TEST_SIZE

# Variáveis com assimetria forte (> 1) identificadas na EDA
LOG_FEATURES = [
    "residual sugar",
    "chlorides",
    "sulphates",
    "free sulfur dioxide",
    "total sulfur dioxide",
]

# Features criadas. 'total_acidity' foi avaliada no notebook 02 e descartada
# (correlação ~0 com a qualidade e redundante com acidez fixa).
ENGINEERED_FEATURES = [
    "free_so2_ratio",
    "bound_so2",
    "volatile_fixed_ratio",
    "alcohol_x_sulphates",
]
CANDIDATE_FEATURES = ENGINEERED_FEATURES + ["total_acidity"]

ENGINEERED_LABELS_PT = {
    "free_so2_ratio": "Proporção SO₂ livre/total",
    "bound_so2": "SO₂ combinado",
    "total_acidity": "Acidez total",
    "volatile_fixed_ratio": "Razão acidez volátil/fixa",
    "alcohol_x_sulphates": "Álcool × Sulfatos",
}

# Das features criadas, as de cauda longa também recebem log
LOG_ENGINEERED = ["bound_so2", "alcohol_x_sulphates"]


def safe_log(x):
    """Logaritmo natural com proteção contra zero (todas as medidas são > 0 na base).

    Usamos log (e não log1p) porque cloretos e sulfatos são valores pequenos (< 1),
    onde log1p(x) ≈ x e quase não reduziria a assimetria.
    """
    return np.log(np.clip(x, 1e-6, None))


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas idênticas (mesmas medidas e mesma nota)."""
    return df.drop_duplicates().reset_index(drop=True)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria variáveis derivadas com base na química do vinho.

    - free_so2_ratio: fração do SO₂ que está livre (ativa como conservante).
    - bound_so2: SO₂ combinado = total − livre (forma pouco ativa, mas que pesa no aroma).
    - total_acidity: acidez fixa + volátil (acidez percebida total).
    - volatile_fixed_ratio: peso do ácido acético (defeito) dentro da acidez.
    - alcohol_x_sulphates: interação das duas variáveis positivas mais fortes.
    """
    out = df.copy()
    out["free_so2_ratio"] = out["free sulfur dioxide"] / out["total sulfur dioxide"]
    out["bound_so2"] = out["total sulfur dioxide"] - out["free sulfur dioxide"]
    out["total_acidity"] = out["fixed acidity"] + out["volatile acidity"]  # candidata (descartada)
    out["volatile_fixed_ratio"] = out["volatile acidity"] / out["fixed acidity"]
    out["alcohol_x_sulphates"] = out["alcohol"] * out["sulphates"]
    return out


def get_feature_lists(use_engineered: bool = True):
    """Retorna (todas as features, features que recebem log)."""
    feats = FEATURES + (ENGINEERED_FEATURES if use_engineered else [])
    logs = LOG_FEATURES + (LOG_ENGINEERED if use_engineered else [])
    return feats, logs


def split_data(df: pd.DataFrame, features, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Divisão treino/teste estratificada pela classe alvo."""
    X, y = df[features], df[TARGET_COL]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def build_preprocessor(features, log_features, scale: bool = True) -> ColumnTransformer:
    """Pipeline de transformação por coluna.

    log_features -> imputação (mediana) -> log -> padronização
    demais       -> imputação (mediana)          -> padronização
    """
    log_cols = [c for c in features if c in log_features]
    other_cols = [c for c in features if c not in log_features]

    log_steps = [("imputer", SimpleImputer(strategy="median")),
                 ("log", FunctionTransformer(safe_log, feature_names_out="one-to-one"))]
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        log_steps.append(("scaler", StandardScaler()))
        num_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        [("log", Pipeline(log_steps), log_cols), ("num", Pipeline(num_steps), other_cols)],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def prepare_data(df: pd.DataFrame, use_engineered: bool = True):
    """Atalho do fluxo completo: dedup -> features -> split. Retorna X_train, X_test, y_train, y_test, features, log_features."""
    data = remove_duplicates(df)
    if use_engineered:
        data = add_engineered_features(data)
    feats, logs = get_feature_lists(use_engineered)
    X_train, X_test, y_train, y_test = split_data(data, feats)
    return X_train, X_test, y_train, y_test, feats, logs
