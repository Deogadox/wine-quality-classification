"""Configurações centrais do projeto: caminhos, constantes e definição do alvo."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

DATA_FILE = DATA_DIR / "winequality-red.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Variável alvo: qualidade >= 7 -> alta qualidade (1); < 7 -> baixa/média (0)
QUALITY_COL = "quality"
TARGET_COL = "high_quality"
QUALITY_THRESHOLD = 7

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

# Nomes em português para gráficos e apresentação
FEATURE_LABELS_PT = {
    "fixed acidity": "Acidez fixa",
    "volatile acidity": "Acidez volátil",
    "citric acid": "Ácido cítrico",
    "residual sugar": "Açúcar residual",
    "chlorides": "Cloretos",
    "free sulfur dioxide": "SO₂ livre",
    "total sulfur dioxide": "SO₂ total",
    "density": "Densidade",
    "pH": "pH",
    "sulphates": "Sulfatos",
    "alcohol": "Teor alcoólico",
    "quality": "Qualidade",
}
