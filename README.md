# 🍷 Wine Quality Classification — Tech Challenge Fase 2 (POSTECH Data Analytics)

Modelo de classificação que prevê se um vinho tinto é de **alta qualidade (nota ≥ 7)** ou **baixa/média qualidade (nota < 7)** a partir de suas características físico-químicas — apoiando enólogos e produtores na tomada de decisão durante a produção.

🎬 **Vídeo de apresentação:** https://youtu.be/bnD7dri3Oyg

📊 **Apresentação executiva (storytelling da análise):** [`presentation/apresentacao_executiva.pdf`](presentation/apresentacao_executiva.pdf) · [versão PowerPoint](presentation/apresentacao_executiva.pptx)

👥 **Grupo:** Felipe Rios Miranda · Guilherme Tavares Azevedo · Thamyres Schroeder Paula Souza

---

## Principais resultados

| | |
|---|---|
| **Excelência é rara** | Só **13,6%** dos vinhos têm nota ≥ 7 (≈ 1 em cada 7) |
| **Principal fator** | **Teor alcoólico**: a taxa de vinhos excelentes vai de **1%** (álcool < 9,5%) a **46%** (álcool ≥ 12%) |
| **Fatores que ajudam** | Álcool (×2,2 na chance por desvio-padrão) e sulfatos (×1,8) |
| **Fatores que atrapalham** | Acidez volátil (×0,67) e SO₂ total (×0,67) |
| **Modelo final** | Regressão Logística — **ROC-AUC 0,89**, **PR-AUC 0,57**, **F1 0,57** no teste |
| **Em linguagem de negócio** | De cada 2 vinhos que o modelo aponta como excelentes, **1 realmente é** (vs. 1 em 7 ao acaso) e ele encontra **~2 em cada 3** vinhos excelentes |

### Comparação dos modelos (conjunto de teste, 272 vinhos)

| Modelo | PR-AUC | ROC-AUC | F1 | Precisão | Recall |
|---|---|---|---|---|---|
| Baseline (sempre "baixa/média") | 0,136 | 0,500 | 0,000 | — | 0,000 |
| **Regressão Logística ⭐** | 0,569 | 0,892 | **0,565** | 0,500 | 0,649 |
| Random Forest | 0,574 | 0,873 | 0,489 | 0,404 | 0,622 |
| Gradient Boosting | 0,592 | 0,896 | 0,564 | 0,537 | 0,595 |

Os três modelos ficaram **estatisticamente empatados** na validação cruzada (diferenças de PR-AUC menores que 1 desvio-padrão entre folds). A Regressão Logística foi escolhida por ter o melhor F1 *out-of-fold* no treino e por ser a mais **interpretável** — escolha feita sem usar o teste e confirmada por ele.

---

## Base de dados

- **Wine Quality Dataset** (vinho tinto, *Vinho Verde* — Portugal), publicado por Cortez et al. (2009) no UCI Machine Learning Repository e redistribuído no Kaggle.
- 1.599 amostras · 11 variáveis físico-químicas · nota de qualidade (0–10) atribuída por especialistas.
- Arquivo: `data/winequality-red.csv`. Para usar o `WineQT.csv` do Kaggle, basta trocar `DATA_FILE` em `src/config.py` (a coluna `Id` é descartada automaticamente).

| Variável | Descrição |
|---|---|
| fixed acidity | Acidez fixa (g/dm³ ácido tartárico) |
| volatile acidity | Acidez volátil (g/dm³ ácido acético) |
| citric acid | Ácido cítrico (g/dm³) |
| residual sugar | Açúcar residual (g/dm³) |
| chlorides | Cloretos (g/dm³ cloreto de sódio) |
| free sulfur dioxide | SO₂ livre (mg/dm³) |
| total sulfur dioxide | SO₂ total (mg/dm³) |
| density | Densidade (g/cm³) |
| pH | pH |
| sulphates | Sulfatos (g/dm³ sulfato de potássio) |
| alcohol | Teor alcoólico (% vol.) |
| quality | Nota de qualidade (alvo original) |

**Variável alvo:** `high_quality = 1` se `quality ≥ 7`, senão `0`.

---

## Pipeline do projeto

| Etapa | Notebook | O que foi feito |
|---|---|---|
| 1. Compreensão do problema | [`01_eda.ipynb`](notebooks/01_eda.ipynb) | Contexto, definição e binarização do alvo |
| 2. Análise exploratória | [`01_eda.ipynb`](notebooks/01_eda.ipynb) | Distribuições, assimetria, boxplots por classe, teste de Mann-Whitney, correlações de Spearman **justificadas pela química do vinho**, outliers (IQR), balanceamento |
| 3. Pré-processamento | [`02_preprocessing.ipynb`](notebooks/02_preprocessing.ipynb) | Remoção de 240 duplicatas (antes do split, evitando vazamento), log nas variáveis de cauda longa, padronização, 5 features criadas e avaliadas, split estratificado 80/20 |
| 4. Modelagem | [`03_modeling.ipynb`](notebooks/03_modeling.ipynb) | Baseline + Regressão Logística + Random Forest + Gradient Boosting; pesos de classe; comparação com/sem feature engineering; GridSearchCV otimizando PR-AUC |
| 5. Avaliação | [`03_modeling.ipynb`](notebooks/03_modeling.ipynb) | Validação cruzada 5×3, ajuste do limiar de decisão no treino, curvas ROC e Precisão×Recall, matrizes de confusão, comparação no teste |
| 6. Interpretação | [`04_interpretation.ipynb`](notebooks/04_interpretation.ipynb) | Coeficientes com IC 95% (bootstrap), importância por permutação nos 3 modelos, simulações "e se", implicações para a produção |

### Decisões metodológicas importantes

- **Métrica principal: PR-AUC.** Com apenas 13,6% de vinhos excelentes, a acurácia é enganosa — o baseline que nunca aponta um vinho excelente já acerta 86%.
- **Sem vazamento de dados:** duplicatas removidas antes da divisão; todo o pré-processamento dentro de um `Pipeline` reajustado a cada fold; limiar de decisão escolhido com previsões *out-of-fold* do treino; teste usado apenas na avaliação final.
- **Feature engineering testada com evidência:** as 4 variáveis criadas (ex.: Álcool × Sulfatos, a mais correlacionada com a qualidade) **não melhoraram** os modelos de forma consistente — o modelo final usa as 11 variáveis originais (mais simples, mesmo desempenho).
- **Outliers mantidos:** 25% dos vinhos têm algum valor extremo, mas quimicamente plausível; foram tratados com transformação logarítmica em vez de removidos.

---

## Implicações para a produção

| Variável | Etapa do processo | Recomendação |
|---|---|---|
| Teor alcoólico ↑ | Colheita e fermentação | Ponto ideal de maturação da uva e fermentação completa |
| Acidez volátil ↓ | Higiene e fermentação | Controle de temperatura, higiene de tanques, menos oxigênio — principal indicador de alerta |
| Sulfatos ↑ / SO₂ total ↓ | Conservação | Proteger sem exagerar: dosagem precisa, dentro dos limites legais |
| — | Uso do modelo | Triagem de lotes para priorizar a prova dos enólogos; limiar ajustável conforme a estratégia |

> ⚠️ As recomendações se baseiam em **associações** nos dados e devem ser validadas com lotes piloto. O modelo apoia — não substitui — a avaliação sensorial dos especialistas.

**Limitações:** base pequena (37 vinhos de alta qualidade no teste), de uma única região e tipo; sem dados de uva, safra, terroir ou envelhecimento.

---

## Estrutura do repositório

```
wine-quality-classification/
├── data/
│   ├── winequality-red.csv        # Base original
│   └── processed/                 # Treino e teste (após limpeza e feature engineering)
├── notebooks/
│   ├── 01_eda.ipynb               # Compreensão do problema + EDA
│   ├── 02_preprocessing.ipynb     # Pré-processamento e feature engineering
│   ├── 03_modeling.ipynb          # Modelagem e avaliação
│   └── 04_interpretation.ipynb    # Interpretação dos resultados
├── src/
│   ├── config.py                  # Caminhos, constantes, definição do alvo
│   ├── data_loader.py             # Leitura da base e criação do alvo
│   ├── preprocessing.py           # Limpeza, features, split e pipeline de transformação
│   ├── modeling.py                # Modelos, tuning, limiar e métricas
│   └── visualization.py           # Estilo dos gráficos
├── results/
│   ├── figures/                   # 20 gráficos (EDA, modelagem e interpretação)
│   ├── models/                    # Modelo final (best_model.joblib) e artefatos
│   └── *.csv                      # Métricas, tuning, importâncias, simulações
├── presentation/
│   ├── apresentacao_executiva.pdf # Storytelling da análise (PDF) — vídeo: https://youtu.be/bnD7dri3Oyg
│   └── apresentacao_executiva.pptx
├── requirements.txt
└── README.md
```

## Como executar

```bash
git clone https://github.com/Deogadox/wine-quality-classification.git
cd wine-quality-classification

python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

jupyter notebook notebooks/
```

Execute os notebooks na ordem (01 → 04). O notebook 03 leva alguns minutos por causa da busca de hiperparâmetros.

### Usando o modelo treinado

```python
import joblib, pandas as pd

art = joblib.load("results/models/best_model.joblib")
pipe, threshold, features = art["pipeline"], art["threshold"], art["features"]

novo_lote = pd.DataFrame([{
    "fixed acidity": 8.5, "volatile acidity": 0.35, "citric acid": 0.40, "residual sugar": 2.3,
    "chlorides": 0.07, "free sulfur dioxide": 12, "total sulfur dioxide": 28, "density": 0.9960,
    "pH": 3.3, "sulphates": 0.78, "alcohol": 12.0,
}])
score = pipe.predict_proba(novo_lote[features])[:, 1][0]
print(f"Pontuação: {score:.2f} → {'ALTA qualidade' if score >= threshold else 'baixa/média'}")
```

> Execute a partir da raiz do repositório (o pipeline usa funções de `src/`).

---

## Integrantes

- **Felipe Rios Miranda** — [github.com/Deogadox](https://github.com/Deogadox)
- **Guilherme Tavares Azevedo**
- **Thamyres Schroeder Paula Souza**

Referência da base: P. Cortez, A. Cerdeira, F. Almeida, T. Matos e J. Reis. *Modeling wine preferences by data mining from physicochemical properties.* Decision Support Systems, 47(4):547–553, 2009.
