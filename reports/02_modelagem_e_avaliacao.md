# Relatório 02 — Modelagem Supervisionada e Avaliação

Fonte: [`notebooks/03_modeling.ipynb`](../notebooks/03_modeling.ipynb) ·
Código: [`src/preprocessing`](../src/preprocessing), [`src/modeling`](../src/modeling), [`src/evaluation`](../src/evaluation) ·
Tabelas: [`tabelas/metricas_modelo.json`](tabelas/metricas_modelo.json), [`tabelas/classification_report.txt`](tabelas/classification_report.txt)

---

## 1. Objetivo analítico

Prever, para cada aluno, se ele será classificado como **alfabetizado (1)** ou
**não alfabetizado (0)** a partir de variáveis educacionais, territoriais e de
porte — sem usar a proficiência, que é o próprio critério de definição do alvo.

## 2. Pipeline

Todo o pré-processamento está **dentro** do objeto `Pipeline` do Scikit-Learn,
o que garante que as estatísticas (mediana da imputação, média e desvio do
scaler, categorias do one-hot) sejam aprendidas somente no conjunto de treino:

```
Pipeline
├── preprocessor (ColumnTransformer)
│   ├── num  →  SimpleImputer(median) → StandardScaler → float32
│   └── cat  →  SimpleImputer('missing') → OneHotEncoder(handle_unknown='ignore')
└── classifier → XGBClassifier
```

| Grupo | Colunas |
| --- | --- |
| Numéricas (6) | `ano`, `qtd_alunos_escola`, `qtd_alunos_municipio`, `qtd_escolas_municipio`, `w_peso_aluno`, `ref_meta_taxa_ano` |
| Categóricas (5) | `serie`, `rede`, `sigla_uf`, `nome_regiao`, `nome_municipio` |

Após o one-hot (puxado pelos 5.173 municípios), a matriz de entrada do modelo
tem cerca de **5.214 colunas**.

## 3. Tratamento de data leakage

| Coluna removida | Motivo |
| --- | --- |
| `leak_proficiencia` | **Define o target** (`alfabetizado := proficiencia >= 743`). Mantê-la produziria acurácia artificial próxima de 100% |
| `y_alfabetizado_label` | O próprio alvo em formato texto |
| `id_aluno`, `id_escola`, `id_municipio` | Identificadores que memorizam em vez de generalizar |

Além da remoção de colunas, o segundo vetor de vazamento — estatísticas
calculadas sobre a base inteira — é neutralizado pela integração do
pré-processamento à pipeline (§2).

## 4. Estratégia de validação

- **Split estratificado 80/20** com `random_state=42`: 134.232 linhas de treino
  e 33.559 de teste, preservando a proporção 59/41 do alvo nos dois conjuntos.
- **`RandomizedSearchCV`** com 15 combinações × 3 folds = 45 ajustes,
  otimizando `roc_auc`. A validação cruzada refaz o pré-processamento a cada
  fold, medindo generalização e não memorização.
- Busca aleatória em vez de exaustiva pelo custo computacional: cada ajuste leva
  de 20 a 30 segundos por causa da largura da matriz.

### Hiperparâmetros selecionados

| Parâmetro | Valor |
| --- | ---: |
| `n_estimators` | 205 |
| `max_depth` | 8 |
| `learning_rate` | 0,095 |
| `subsample` | 0,827 |
| `colsample_bytree` | 0,902 |

`subsample` e `colsample_bytree` abaixo de 1 fazem cada árvore ver apenas parte
das linhas e das colunas — é o principal mecanismo de controle de overfitting
usado aqui, junto com a profundidade limitada.

## 5. Resultados no conjunto de teste

| Métrica | Valor |
| --- | ---: |
| **ROC-AUC** | **0,6667** |
| **F1-Score ponderado** | **0,6100** |
| Acurácia | 0,6350 |

| Classe | Precisão | Recall | F1 | Suporte |
| --- | ---: | ---: | ---: | ---: |
| 0 — Não alfabetizado | 0,591 | 0,345 | **0,436** | 13.713 |
| 1 — Alfabetizado | 0,649 | 0,835 | **0,730** | 19.846 |

**Matriz de confusão**

| | Previsto: Não | Previsto: Sim |
| --- | ---: | ---: |
| **Real: Não** | 4.735 | 8.978 |
| **Real: Sim** | 3.271 | 16.575 |

![Curva ROC](../images/curva_roc.png)

## 6. Interpretação dos resultados

**O modelo discrimina de forma moderada.** Um AUC de 0,667 significa que, ao
sortear um aluno alfabetizado e um não alfabetizado, o modelo atribui
probabilidade maior ao primeiro em cerca de 67% das vezes — bem acima do acaso
(50%), longe de um classificador forte (>0,80).

**O desempenho é assimétrico entre as classes.** O modelo acerta 83,5% dos
alunos alfabetizados, mas apenas 34,5% dos não alfabetizados. Traduzindo para
o problema real: **8.978 alunos que não estão alfabetizados seriam classificados
como alfabetizados** — exatamente o erro mais caro do ponto de vista de política
pública, porque deixa de sinalizar quem precisa de intervenção.

**Esse teto era esperado, e a causa é a base, não o algoritmo.** As features
disponíveis descrevem *onde* o aluno estuda (região, UF, município, rede) e o
*tamanho* da rede — nenhuma delas descreve o aluno, a família, a escola ou a
prática pedagógica. Dois alunos do mesmo município, na mesma rede, com o mesmo
porte de escola são, para o modelo, idênticos; na realidade, podem ter
trajetórias completamente diferentes. O modelo captura o **efeito territorial**
da alfabetização e nada além disso.

### Como melhorar o recall da classe 0

O limiar de decisão padrão (0,5) favorece a classe majoritária. Para uso em
política pública, em que identificar o aluno em risco vale mais que evitar
falsos alarmes, baixar o limiar aumentaria o recall da classe 0 ao custo de
precisão. As probabilidades já são exportadas pelo pipeline, então esse ajuste
não exige retreino.

## 7. Escolha do algoritmo — justificativa

| Alternativa | Por que não foi o modelo principal |
| --- | --- |
| Regressão Logística | A EDA mostrou correlação linear quase nula das numéricas (\|ρ\| < 0,05); o sinal está em interações entre categóricas |
| Random Forest | Boa opção, mas boosting costuma superar bagging em dados tabulares com sinal fraco |
| **XGBoost** | **Escolhido**: lida nativamente com interações não lineares, alta cardinalidade e valores faltantes, com controle explícito de overfitting |

Continua em [Relatório 03 — Interpretabilidade](03_interpretabilidade_shap.md).
