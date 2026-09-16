"""
Módulo de avaliação: cálculo e exibição de métricas de performance.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(pipeline, X_test, y_test, verbose: bool = True) -> dict:
    """
    Calcula as principais métricas de avaliação do modelo sobre o conjunto de teste.

    Args:
        pipeline: Pipeline treinada (preprocessador + classificador).
        X_test: Features do conjunto de teste.
        y_test: Target real do conjunto de teste.
        verbose: Se True, imprime o relatório no console.

    Returns:
        Dicionário com as métricas: roc_auc, f1_weighted e o relatório completo.
    """
    y_proba = predict_proba_em_lotes(pipeline, X_test)
    y_pred = (y_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred,
                                   target_names=['Não alfabetizado', 'Alfabetizado'])
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    matriz = confusion_matrix(y_test, y_pred)

    if verbose:
        print("--- Relatório de Classificação ---")
        print(report)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(f"F1-Score Ponderado: {f1:.4f}")

    return {
        'roc_auc': roc_auc,
        'f1_weighted': f1,
        'classification_report': report,
        'classification_report_dict': report_dict,
        'confusion_matrix': matriz,
        'y_pred': y_pred,
        'y_proba': y_proba
    }


def predict_proba_em_lotes(pipeline, X: pd.DataFrame, tamanho_lote: int = 20_000) -> np.ndarray:
    """
    Prediz a probabilidade da classe positiva processando a base em lotes.

    O One-Hot de `nome_municipio` gera uma matriz muito larga (~5.200 colunas).
    Transformar as 167 mil linhas de uma vez consumiria vários GB de RAM, então
    a predição é feita por blocos. O resultado é idêntico ao de uma chamada única.
    """
    partes = []
    for inicio in range(0, len(X), tamanho_lote):
        lote = X.iloc[inicio:inicio + tamanho_lote]
        partes.append(pipeline.predict_proba(lote)[:, 1])
    return np.concatenate(partes)


def get_roc_curve_data(y_test, y_proba) -> tuple:
    """Retorna os dados da curva ROC para plotagem."""
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    return fpr, tpr, thresholds


def taxa_prevista_por_grupo(df_analise: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """
    Agrega a probabilidade média prevista de alfabetização por uma variável
    categórica (região, rede, UF), usada para responder às perguntas de negócio.
    """
    agg = (
        df_analise
        .groupby(coluna)
        .agg(taxa_prevista=('prob_alfabetizado', 'mean'),
             total_alunos=('prob_alfabetizado', 'count'))
        .sort_values('taxa_prevista')
        .reset_index()
    )
    agg['taxa_prevista_pct'] = agg['taxa_prevista'] * 100
    return agg


def predict_municipality_risk(pipeline, df_full: pd.DataFrame,
                              cols_to_drop: list, target_col: str) -> pd.DataFrame:
    """
    Estima a taxa de alfabetização prevista por município e identifica
    quais estão em risco de não atingir a meta estabelecida.

    Args:
        pipeline: Pipeline treinada.
        df_full: DataFrame com todos os dados (antes de remover colunas).
        cols_to_drop: Lista de colunas a remover antes da predição.
        target_col: Nome da coluna target.

    Returns:
        DataFrame com taxa prevista, meta e flag de risco por município.
    """
    df_model = df_full.drop(columns=cols_to_drop)
    df_model[target_col] = df_model[target_col].astype(int)
    X_full = df_model.drop(columns=[target_col])

    df_model['prob_alfabetizado'] = predict_proba_em_lotes(pipeline, X_full)

    municipio_risco = df_model.groupby(['nome_municipio', 'sigla_uf', 'nome_regiao']).agg(
        taxa_prevista=('prob_alfabetizado', 'mean'),
        meta=('ref_meta_taxa_ano', 'mean'),
        total_alunos=('prob_alfabetizado', 'count')
    ).dropna(subset=['meta']).reset_index()

    municipio_risco['taxa_prevista_pct'] = municipio_risco['taxa_prevista'] * 100
    municipio_risco['gap_meta'] = municipio_risco['taxa_prevista_pct'] - municipio_risco['meta']
    municipio_risco['em_risco'] = municipio_risco['gap_meta'] < 0

    return municipio_risco.sort_values('gap_meta').reset_index(drop=True)


def risco_por_regiao(municipio_risco: pd.DataFrame) -> pd.DataFrame:
    """Consolida a proporção de municípios em risco por região."""
    agg = municipio_risco.groupby('nome_regiao').agg(
        total=('em_risco', 'count'),
        em_risco=('em_risco', 'sum')
    ).reset_index()
    agg['pct_em_risco'] = agg['em_risco'] / agg['total'] * 100
    return agg.sort_values('pct_em_risco', ascending=False).reset_index(drop=True)
