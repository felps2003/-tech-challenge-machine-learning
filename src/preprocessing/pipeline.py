"""
Módulo de pré-processamento: preparação da base analítica para modelagem.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE, TEST_SIZE


# Colunas que não devem entrar no modelo por causarem data leakage ou serem identificadores
COLS_TO_DROP = [
    'id_aluno',
    'id_escola',
    'id_municipio',
    'y_alfabetizado_label',
    'leak_proficiencia'
]

TARGET_COL = 'y_alfabetizado'


def load_data(path: str) -> pd.DataFrame:
    """Carrega a base analítica no formato Parquet."""
    return pd.read_parquet(path)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Remove colunas de data leakage e identificadores,
    e separa features (X) da variável alvo (y).
    """
    df_model = df.drop(columns=COLS_TO_DROP)
    df_model[TARGET_COL] = df_model[TARGET_COL].astype(int)

    X = df_model.drop(columns=[TARGET_COL])
    y = df_model[TARGET_COL]

    return X, y


def split_train_test(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Divide a base em treino e teste de forma estratificada.

    A estratificação preserva a proporção 59/41 entre alfabetizados e não
    alfabetizados nos dois conjuntos, e a semente fixa garante replicabilidade.
    """
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )


def _to_float32(X):
    """Converte a matriz transformada para float32 (usado dentro da pipeline)."""
    return np.asarray(X, dtype=np.float32)


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Constrói o ColumnTransformer com pipelines distintas para
    variáveis numéricas (imputação por mediana + StandardScaler)
    e categóricas (imputação por constante + OneHotEncoder).

    O One-Hot de `nome_municipio` gera ~5.200 colunas. Como o XGBoost opera
    internamente em float32, a matriz é convertida para float32 ainda dentro
    da pipeline: o resultado do modelo é o mesmo e o pico de memória cai pela
    metade (de ~5,6 GB para ~2,8 GB no conjunto de treino).
    """
    num_cols = X_train.select_dtypes(include=['int64', 'float64', 'Int64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'string']).columns.tolist()

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('to_float32', FunctionTransformer(_to_float32, feature_names_out='one-to-one'))
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False,
                                  dtype=np.float32))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer,
                      num_cols: list, cat_cols: list) -> list:
    """
    Recupera os nomes das colunas depois do pré-processamento
    (numéricas na ordem original + colunas geradas pelo One-Hot).
    """
    ohe_names = (
        preprocessor
        .named_transformers_['cat']
        .named_steps['encoder']
        .get_feature_names_out(cat_cols)
        .tolist()
    )
    return list(num_cols) + ohe_names
