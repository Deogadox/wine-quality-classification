"""Estilo visual padrão e helper para salvar figuras do projeto."""
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.config import FIGURES_DIR

# Paleta (validada para daltonismo): azul = baixa/média, laranja = alta qualidade
COLOR_LOW = "#2a78d6"
COLOR_HIGH = "#eb6834"
COLOR_ACCENT = "#1baf7a"  # destaque neutro (não é classe): variáveis criadas, dados transformados
CLASS_PALETTE = {0: COLOR_LOW, 1: COLOR_HIGH}
CLASS_LABELS = {0: "Baixa/Média (< 7)", 1: "Alta (≥ 7)"}

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

# Divergente: azul (negativo) -> cinza neutro -> laranja (positivo)
DIVERGING = LinearSegmentedColormap.from_list(
    "wine_div", ["#184f95", "#6da7ec", "#f0efec", "#f2a07e", "#b8461a"]
)


def set_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 10,
        "legend.frameon": False,
        "figure.dpi": 110,
    })


def save_fig(fig, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
