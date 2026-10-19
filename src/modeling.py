"""Modelos, busca de hiperparâmetros, ajuste de limiar e métricas de avaliação."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.preprocessing import build_preprocessor

MODEL_NAMES_PT = {
    "baseline": "Baseline (sempre 'baixa/média')",
    "logistic": "Regressão Logística",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}

CV_SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
    "balanced_accuracy": "balanced_accuracy",
}


def cv_splitter(n_repeats: int = 3):
    """Validação cruzada estratificada 5-fold, repetida para estimativas mais estáveis."""
    return RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=RANDOM_STATE)


def get_estimators():
    """Classificadores base. Desbalanceamento tratado com pesos de classe."""
    return {
        "baseline": DummyClassifier(strategy="most_frequent"),
        "logistic": LogisticRegression(class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(class_weight="balanced_subsample", n_estimators=500,
                                                n_jobs=-1, random_state=RANDOM_STATE),
        "gradient_boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_STATE),
    }


# Grades enxutas e focadas em controlar overfitting (base pequena: ~1.100 vinhos no treino)
PARAM_GRIDS = {
    "logistic": {
        "model__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
        "model__l1_ratio": [0.0, 0.5, 1.0],   # 0 = L2 (ridge), 1 = L1 (lasso)
        "model__solver": ["saga"],
    },
    "random_forest": {
        "model__max_depth": [None, 6, 10],
        "model__min_samples_leaf": [1, 3, 5],
        "model__max_features": ["sqrt", 0.5],
    },
    "gradient_boosting": {
        "model__learning_rate": [0.03, 0.1],
        "model__max_depth": [3, 5, None],
        "model__max_iter": [200, 400],
        "model__min_samples_leaf": [10, 20],
        "model__l2_regularization": [0.0, 1.0],
    },
}


def make_pipeline(estimator, features, log_features) -> Pipeline:
    """Pré-processamento + modelo em um único objeto (sem vazamento na validação cruzada)."""
    return Pipeline([("prep", build_preprocessor(features, log_features, scale=True)), ("model", clone(estimator))])


def cross_validate_model(pipe, X, y, cv=None) -> pd.Series:
    """Médias das métricas de validação cruzada no treino."""
    cv = cv or cv_splitter()
    res = cross_validate(pipe, X, y, cv=cv, scoring=CV_SCORING, n_jobs=-1)
    out = {}
    for k in CV_SCORING:
        out[f"{k}"] = res[f"test_{k}"].mean()
        out[f"{k}_std"] = res[f"test_{k}"].std()
    return pd.Series(out)


def tune_model(name, features, log_features, X, y, scoring="average_precision"):
    """GridSearch com validação cruzada estratificada, otimizando PR-AUC (adequada a classes raras)."""
    pipe = make_pipeline(get_estimators()[name], features, log_features)
    grid = GridSearchCV(pipe, PARAM_GRIDS[name], scoring=scoring,
                        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), n_jobs=-1, refit=True)
    grid.fit(X, y)
    return grid


def find_best_threshold(pipe, X, y, beta: float = 1.0):
    """Escolhe o limiar de decisão que maximiza F-beta nas previsões out-of-fold do TREINO.

    Assim o teste continua intocado até a avaliação final.
    """
    proba = cross_val_predict(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                              method="predict_proba", n_jobs=-1)[:, 1]
    prec, rec, thr = precision_recall_curve(y, proba)
    prec, rec = prec[:-1], rec[:-1]
    fbeta = (1 + beta**2) * prec * rec / np.clip(beta**2 * prec + rec, 1e-12, None)
    i = int(np.nanargmax(fbeta))
    return float(thr[i]), float(fbeta[i]), proba


def evaluate(y_true, proba, threshold: float = 0.5) -> dict:
    """Métricas de teste para a classe positiva (alta qualidade)."""
    pred = (proba >= threshold).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y_true, proba),
        "PR-AUC": average_precision_score(y_true, proba),
        "F1": f1_score(y_true, pred, zero_division=0),
        "Precisão": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "Acurácia balanceada": balanced_accuracy_score(y_true, pred),
        "Acurácia": accuracy_score(y_true, pred),
        "Limiar": threshold,
    }
