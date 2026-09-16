"""
Módulo de visualização: gera e salva na pasta `images/` todos os gráficos
usados nos notebooks, no README e nos relatórios de `reports/`.

Cada função recebe os dados já prontos, devolve o caminho do arquivo gerado
e não depende de estado de notebook — o que torna as figuras reprodutíveis
por qualquer pessoa que rode `python -m src.run_pipeline`.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sem janela: permite gerar figuras via script

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap

from src.config import IMAGES_DIR

DPI = 150

# Nomes dos arquivos gerados (mantidos iguais aos já referenciados no projeto)
ARQ_TARGET = "eda_distribuicao_variavel_target.png"
ARQ_TAXA_REDE = "taxa_alfabetizacao_por_rede_de_ensino.png"
ARQ_TAXA_REGIAO = "taxa_alfabetizacao_por_regiao.png"
ARQ_DIST_NUMERICAS = "distribuicao_qdt_alunos_escola_municipio.png"
ARQ_CORRELACAO = "matriz_de_correlacao.png"
ARQ_ROC = "curva_roc.png"
ARQ_FEATURE_IMPORTANCE = "feature_importance.png"
ARQ_SHAP_BAR = "shap_summary_bar.png"
ARQ_SHAP_BEESWARM = "shap_summary_beeswarm.png"
ARQ_RISCO_REGIAO = "risco_por_regiao.png"
ARQ_MUNICIPIOS_RISCO = "municipios_em_risco.png"

COLS_NUMERICAS_PORTE = ['qtd_alunos_escola', 'qtd_alunos_municipio', 'qtd_escolas_municipio']


def configurar_estilo() -> None:
    """Aplica o tema visual usado em todas as figuras do projeto."""
    sns.set_theme(style='whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)


def _salvar(nome_arquivo: str) -> Path:
    """Salva a figura corrente em `images/` e fecha o buffer."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    caminho = IMAGES_DIR / nome_arquivo
    plt.savefig(caminho, dpi=DPI, bbox_inches='tight')
    plt.close('all')
    return caminho


# ---------------------------------------------------------------------------
# 1. Análise exploratória (notebook 02)
# ---------------------------------------------------------------------------

def plot_distribuicao_target(df: pd.DataFrame) -> Path:
    """Contagem de alunos alfabetizados x não alfabetizados."""
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x='y_alfabetizado', hue='y_alfabetizado',
                       palette='Set2', legend=False)
    plt.title('Distribuição da Variável Target (y_alfabetizado)')
    plt.xlabel('Alfabetizado (0 = Não, 1 = Sim)')
    plt.ylabel('Quantidade de Alunos')
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom')
    return _salvar(ARQ_TARGET)


def plot_taxa_por_rede(df: pd.DataFrame) -> Path:
    """Taxa média de alfabetização por rede de ensino."""
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='rede', y='y_alfabetizado', hue='rede',
                palette='Pastel1', errorbar=None, legend=False)
    plt.title('Taxa de Alfabetização por Rede de Ensino')
    plt.xlabel('Rede de ensino')
    plt.ylabel('Proporção de Alfabetizados')
    return _salvar(ARQ_TAXA_REDE)


def plot_taxa_por_regiao(df: pd.DataFrame) -> Path:
    """Taxa média de alfabetização por região do país."""
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x='nome_regiao', y='y_alfabetizado', hue='nome_regiao',
                palette='Pastel2', errorbar=None, legend=False)
    plt.title('Taxa de Alfabetização por Região')
    plt.xlabel('Região')
    plt.ylabel('Proporção de Alfabetizados')
    return _salvar(ARQ_TAXA_REGIAO)


def plot_distribuicoes_numericas(df: pd.DataFrame, colunas: list = None) -> Path:
    """Histogramas das variáveis de porte (escola e município)."""
    colunas = colunas or COLS_NUMERICAS_PORTE
    fig, axes = plt.subplots(1, len(colunas), figsize=(18, 5))
    for ax, col in zip(axes, colunas):
        sns.histplot(df[col], bins=30, ax=ax, kde=True)
        ax.set_title(f'Distribuição de {col}')
    plt.tight_layout()
    return _salvar(ARQ_DIST_NUMERICAS)


def plot_matriz_correlacao(df: pd.DataFrame, colunas: list = None) -> Path:
    """Matriz de correlação de Spearman entre o target e as variáveis numéricas."""
    colunas = colunas or (COLS_NUMERICAS_PORTE + ['w_peso_aluno', 'ref_meta_taxa_ano'])
    plt.figure(figsize=(8, 6))
    corr = df[['y_alfabetizado'] + colunas].astype(float).corr(method='spearman')
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de Correlação (Spearman)')
    return _salvar(ARQ_CORRELACAO)


# ---------------------------------------------------------------------------
# 2. Avaliação do modelo (notebook 03)
# ---------------------------------------------------------------------------

def plot_curva_roc(fpr, tpr, roc_auc: float) -> Path:
    """Curva ROC do modelo no conjunto de teste."""
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Taxa de Falso Positivo')
    plt.ylabel('Taxa de Verdadeiro Positivo')
    plt.title('Curva ROC - Predição de Alfabetização')
    plt.legend(loc="lower right")
    return _salvar(ARQ_ROC)


# ---------------------------------------------------------------------------
# 3. Interpretabilidade (notebook 04)
# ---------------------------------------------------------------------------

def plot_feature_importance(feat_imp_df: pd.DataFrame, top_n: int = 20) -> Path:
    """Top N variáveis pelo ganho interno do XGBoost."""
    dados = feat_imp_df.head(top_n)
    plt.figure(figsize=(10, 7))
    sns.barplot(data=dados, x='importance', y='feature', hue='feature',
                palette='Blues_r', legend=False)
    plt.title(f'Top {top_n} Variáveis Mais Importantes (XGBoost Feature Importance)')
    plt.xlabel('Importância')
    plt.ylabel('Variável')
    plt.tight_layout()
    return _salvar(ARQ_FEATURE_IMPORTANCE)


def plot_shap_bar(shap_values, amostra: pd.DataFrame, max_display: int = 15) -> Path:
    """Importância global das variáveis segundo o SHAP."""
    plt.figure()
    shap.summary_plot(shap_values, amostra, plot_type='bar',
                      max_display=max_display, show=False)
    plt.title('SHAP - Importância Global das Variáveis')
    plt.tight_layout()
    return _salvar(ARQ_SHAP_BAR)


def plot_shap_beeswarm(shap_values, amostra: pd.DataFrame, max_display: int = 15) -> Path:
    """Direção e magnitude do impacto de cada variável (beeswarm)."""
    plt.figure()
    shap.summary_plot(shap_values, amostra, max_display=max_display, show=False)
    plt.title('SHAP - Direção e Magnitude do Impacto das Variáveis')
    plt.tight_layout()
    return _salvar(ARQ_SHAP_BEESWARM)


# ---------------------------------------------------------------------------
# 4. Risco de não atingir metas municipais (notebook 05)
# ---------------------------------------------------------------------------

def plot_risco_por_regiao(risco_regiao: pd.DataFrame) -> Path:
    """Proporção de municípios em risco de não atingir a meta, por região."""
    dados = risco_regiao.sort_values('pct_em_risco', ascending=False)
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=dados, x='nome_regiao', y='pct_em_risco',
                     hue='nome_regiao', palette='Reds_r', legend=False)
    plt.title('Proporção de Municípios em Risco de Não Atingir a Meta por Região')
    plt.xlabel('Região')
    plt.ylabel('% Municípios em Risco')
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%',
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom')
    plt.tight_layout()
    return _salvar(ARQ_RISCO_REGIAO)


def plot_top_municipios_em_risco(top_risco: pd.DataFrame, top_n: int = 15) -> Path:
    """Municípios com maior déficit em relação à meta de alfabetização."""
    dados = top_risco.head(top_n)
    labels = dados['nome_municipio'] + ' (' + dados['sigla_uf'] + ')'
    plt.figure(figsize=(11, 7))
    plt.barh(labels, dados['gap_meta'], color='#d62728')
    plt.axvline(0, color='black', linewidth=1, linestyle='--')
    plt.gca().invert_yaxis()
    plt.xlabel('Gap em relação à meta (taxa prevista % - meta %)')
    plt.title(f'Top {top_n} Municípios em Maior Risco de Não Atingir a Meta de Alfabetização')
    plt.tight_layout()
    return _salvar(ARQ_MUNICIPIOS_RISCO)
