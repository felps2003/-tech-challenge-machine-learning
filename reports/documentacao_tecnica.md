# Documentação Técnica

Tech Challenge — Fase 3 · Predição e Inteligência Analítica para Alfabetização no Brasil

Este documento descreve as decisões de engenharia e modelagem do projeto. Os
resultados numéricos estão nos relatórios [01](01_analise_exploratoria.md),
[02](02_modelagem_e_avaliacao.md), [03](03_interpretabilidade_shap.md) e
[04](04_municipios_em_risco.md); as tabelas brutas estão em [`tabelas/`](tabelas).

---

## 1. Arquitetura do repositório

```
tech-challenge-machine-learning/
├── data/raw/          # base analítica (ABT) extraída da camada Gold — não versionada
├── notebooks/         # registro narrativo da análise (01 a 05)
├── src/               # o mesmo pipeline em código reaproveitável
│   ├── config.py           # caminhos e semente única do projeto
│   ├── preprocessing/      # ABT -> X, y, split e ColumnTransformer
│   ├── modeling/           # pipeline final e busca de hiperparâmetros
│   ├── evaluation/         # métricas, curva ROC e risco municipal
│   ├── visualization/      # todas as figuras de images/
│   └── run_pipeline.py     # execução ponta a ponta
├── reports/           # relatórios analíticos + tabelas de resultados
├── images/            # figuras geradas pelo pipeline
├── requirements.txt
├── README.md
└── .gitignore
```

A divisão de responsabilidades é intencional:

- **notebooks/** contam a história da análise, com as saídas preservadas;
- **src/** é a versão executável e testável do mesmo fluxo, sem estado de kernel;
- **reports/** e **images/** guardam somente artefatos gerados — nada é editado à mão.

Isso significa que `python -m src.run_pipeline` reconstrói integralmente as
pastas `images/` e `reports/tabelas/`.

---

## 2. Origem dos dados

A base analítica (ABT) vem da camada Gold construída na Fase 2, no BigQuery.
O notebook `01_data_extraction.ipynb` cria três objetos:

| Objeto | Descrição |
| --- | --- |
| `gold.alunos_features` | ABT em nível de aluno, particionada por `ano` e clusterizada por `sigla_uf` e `rede` |
| `governanca.dicionario_features_fase3` | Dicionário de colunas com o papel de cada variável |
| `gold.alunos_features_amostra` | Amostra determinística de ~5% (`MOD(ABS(FARM_FINGERPRINT(id_aluno)), 20) = 0`) |

A amostragem por hash do `id_aluno` é determinística: a mesma consulta devolve
sempre as mesmas linhas, o que preserva a replicabilidade mesmo quando a tabela
completa é reprocessada. A amostra resultante tem **167.791 alunos e 17 colunas**,
e é ela que alimenta todo o restante do projeto (`data/raw/abt_alunos_alfabetizacao.parquet`).

### Dicionário de features

| Coluna | Papel | Observação |
| --- | --- | --- |
| `id_aluno` | identificador | Chave do aluno; rastreabilidade |
| `id_escola` | identificador | Chave INEP da escola |
| `id_municipio` | identificador | Código IBGE de 7 dígitos (~5.570 níveis) |
| `y_alfabetizado` | **target** | Alvo binário 1 = Sim / 0 = Não |
| `y_alfabetizado_label` | target | Alvo original decodificado ("Sim"/"Não") |
| `ano` | split | Define os anos disponíveis na base |
| `serie` | feature categórica | Baixa cardinalidade — one-hot |
| `rede` | feature categórica | Estadual / Municipal / (demais) — one-hot |
| `sigla_uf` | feature categórica | 26 níveis presentes na amostra — one-hot |
| `nome_regiao` | feature categórica | 5 níveis — one-hot |
| `nome_municipio` | rótulo / feature | 5.173 níveis na amostra |
| `qtd_alunos_escola` | feature numérica | Porte da escola no ano |
| `qtd_alunos_municipio` | feature numérica | Porte do município no ano |
| `qtd_escolas_municipio` | feature numérica | Tamanho da rede escolar do município |
| `w_peso_aluno` | peso amostral | Peso da avaliação; **não deveria ser feature** (ver §5) |
| `ref_meta_taxa_ano` | referência | Meta municipal de alfabetização do ano |
| `leak_proficiencia` | **bloqueada** | Define o target (`alfabetizado := proficiencia >= 743`) |

---

## 3. Tratamento de data leakage

O vazamento mais grave da base é direto: `leak_proficiencia` é a variável a
partir da qual o target foi derivado por uma regra de corte. Mantê-la produziria
um modelo com acurácia próxima de 100% e valor preditivo nulo.

O bloqueio acontece em um único lugar — `COLS_TO_DROP`, em
[`src/preprocessing/pipeline.py`](../src/preprocessing/pipeline.py) — usado tanto
pelos notebooks quanto pelo pipeline executável:

```python
COLS_TO_DROP = [
    'id_aluno',        # identificador, não generaliza
    'id_escola',       # identificador, não generaliza
    'id_municipio',    # redundante com nome_municipio
    'y_alfabetizado_label',  # o próprio target em texto
    'leak_proficiencia',     # define o target — vazamento direto
]
```

O segundo tipo de vazamento, mais sutil, é o das **estatísticas de
pré-processamento**: calcular mediana de imputação ou média/desvio do
`StandardScaler` sobre a base inteira contamina o conjunto de teste com
informação que o modelo não teria em produção. Por isso a imputação, a
padronização e o one-hot estão **dentro** da `Pipeline` do Scikit-Learn, e não
antes dela. Com isso, o `.fit()` aprende as estatísticas apenas do treino, e a
validação cruzada refaz esse cálculo a cada fold.

---

## 4. Pipeline de pré-processamento e modelagem

```
Pipeline
├── preprocessor (ColumnTransformer)
│   ├── num: SimpleImputer(median) -> StandardScaler -> float32
│   └── cat: SimpleImputer('missing') -> OneHotEncoder(handle_unknown='ignore')
└── classifier: XGBClassifier
```

Decisões e por quês:

- **Imputação por mediana** nas numéricas: `ref_meta_taxa_ano` é nula para
  aproximadamente metade dos alunos (municípios sem meta cadastrada para o ano),
  e as variáveis de porte têm distribuição fortemente assimétrica — a mediana é
  mais robusta que a média nesse cenário.
- **`handle_unknown='ignore'`** no one-hot: municípios presentes no teste e
  ausentes do treino viram um vetor de zeros em vez de quebrar a predição.
- **`StandardScaler`** não é necessário para árvores, mas é mantido porque torna
  a mesma matriz utilizável por modelos lineares em comparações futuras.
- **Conversão para `float32`**: o one-hot de `nome_municipio` gera ~5.200
  colunas; em `float64` a matriz de treino ocuparia cerca de 5,6 GB. Como o
  XGBoost opera internamente em `float32`, a conversão é feita ainda dentro da
  pipeline e corta o pico de memória pela metade sem alterar o resultado.
- **Predição em lotes**: `predict_proba_em_lotes()` processa a base completa em
  blocos de 20 mil linhas, o que evita materializar uma matriz de 167.791 ×
  5.214 de uma só vez. O resultado é idêntico ao de uma chamada única.

### Escolha do algoritmo

A EDA mostrou correlação de Spearman praticamente nula entre o target e as
variáveis numéricas de porte (entre -0,04 e 0,02), enquanto as variáveis
categóricas (região, UF, rede) concentram a maior parte do sinal. Esse é um
cenário de interações não lineares entre variáveis de alta cardinalidade —
exatamente onde modelos de boosting em árvore superam modelos lineares. Daí a
escolha do **XGBoost**.

### Validação e busca de hiperparâmetros

- Split estratificado 80/20 (`stratify=y`, `random_state=42`), preservando a
  proporção 59/41 do target nos dois conjuntos.
- `RandomizedSearchCV` com 15 combinações, `cv=3` e `scoring='roc_auc'` —
  45 ajustes no total. O `RandomizedSearch` foi preferido ao `GridSearch` pelo
  custo: cada ajuste leva de 20 a 30 segundos por causa da largura da matriz.
- Os melhores hiperparâmetros encontrados estão fixos em `BEST_PARAMS`
  ([`src/modeling/train.py`](../src/modeling/train.py)), de modo que o pipeline
  reproduza o modelo final em um único treino. A busca continua disponível em
  `run_hyperparameter_search()` para quem quiser refazê-la.

---

## 5. Limitações conhecidas

1. **`w_peso_aluno` como feature.** O dicionário de governança classifica essa
   coluna como peso amostral ("nunca feature"), mas ela entrou no conjunto de
   treino. O SHAP mostra que o modelo a usa de forma relevante. Como o peso é
   função do desenho amostral da avaliação, e não uma característica do aluno,
   parte do poder preditivo observado pode não se sustentar em produção. Remover
   essa coluna e reavaliar é o primeiro item de trabalho futuro.
2. **`ref_meta_taxa_ano` como feature.** A meta municipal é derivada da taxa-base
   histórica do município e é usada, no relatório 04, como referência de
   comparação. Usá-la como feature e depois comparar a predição contra ela cria
   uma circularidade parcial — o modelo já "viu" a meta ao estimar a taxa.
3. **`serie` é constante** na amostra (um único valor), portanto não contribui
   com informação; permanece na pipeline apenas para compatibilidade com bases
   futuras que incluam outras séries.
4. **Municípios com poucos alunos na amostra** (5% do total) produzem estimativas
   instáveis: há municípios representados por 1 ou 2 alunos no ranking de risco.
   O relatório 04 sinaliza o `total_alunos` justamente para permitir esse filtro.
5. **Ausência de variáveis socioeconômicas e pedagógicas.** A ABT contém apenas
   variáveis estruturais (território, rede, porte). Renda, escolaridade dos
   responsáveis, infraestrutura escolar e formação docente — os fatores que a
   literatura aponta como determinantes — não estão presentes, o que explica o
   teto de desempenho observado.

---

## 6. Reprodutibilidade

```bash
# 1. Ambiente
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Extração da ABT (exige credenciais GCP e o .env com GCP_PROJECT_ID)
jupyter notebook notebooks/01_data_extraction.ipynb

# 3. Pipeline completo: regenera images/ e reports/tabelas/
python -m src.run_pipeline
```

Pontos que garantem a replicabilidade:

- semente única (`RANDOM_STATE = 42` em `src/config.py`) usada no split, no
  classificador e na amostragem do SHAP;
- amostra da ABT definida por hash determinístico no BigQuery;
- versões fixadas em `requirements.txt`, correspondentes ao ambiente que gerou
  os resultados publicados;
- `reports/tabelas/resumo_execucao.json` registra data da execução,
  hiperparâmetros e todos os números citados nos relatórios.

O arquivo `data/raw/abt_alunos_alfabetizacao.parquet` não é versionado
(`.gitignore`), porque a entrega prevê que a ABT seja reconstruída a partir da
camada Gold. Sem ele, execute primeiro o notebook 01.
