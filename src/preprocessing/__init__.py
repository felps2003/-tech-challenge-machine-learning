"""Pré-processamento da base analítica."""

from .pipeline import (
    COLS_TO_DROP,
    TARGET_COL,
    build_preprocessor,
    load_data,
    prepare_features,
    split_train_test,
)
