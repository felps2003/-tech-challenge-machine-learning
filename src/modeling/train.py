"""
Módulo de modelagem: treinamento e busca de hiperparâmetros.
"""

import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform


# Melhores hiperparâmetros encontrados pelo RandomizedSearchCV
BEST_PARAMS = {
    'n_estimators': 205,
    'max_depth': 8,
    'learning_rate': 0.095,
    'subsample': 0.827,
    'colsample_bytree': 0.902,
    'random_state': 42
}


def build_model_pipeline(preprocessor, params: dict = None) -> Pipeline:
    """
    Monta a pipeline final unindo o pré-processador ao classificador XGBoost.

    Args:
        preprocessor: ColumnTransformer já configurado.
        params: Dicionário de hiperparâmetros do XGBoost.
                Se None, usa os melhores parâmetros encontrados no tuning.

    Returns:
        Pipeline do Scikit-Learn pronta para treino.
    """
    if params is None:
        params = BEST_PARAMS

    classifier = xgb.XGBClassifier(**params)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    return pipeline


def run_hyperparameter_search(pipeline: Pipeline, X_train, y_train,
                               n_iter: int = 15, cv: int = 3) -> RandomizedSearchCV:
    """
    Executa o RandomizedSearchCV para otimização de hiperparâmetros do XGBoost.

    Args:
        pipeline: Pipeline base com preprocessador e classificador.
        X_train: Features de treino.
        y_train: Target de treino.
        n_iter: Número de combinações a testar.
        cv: Número de folds para validação cruzada.

    Returns:
        Objeto RandomizedSearchCV já ajustado.
    """
    param_dist = {
        'classifier__n_estimators': randint(100, 300),
        'classifier__max_depth': randint(3, 10),
        'classifier__learning_rate': uniform(0.01, 0.2),
        'classifier__subsample': uniform(0.6, 0.4),
        'classifier__colsample_bytree': uniform(0.6, 0.4)
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=2
    )

    search.fit(X_train, y_train)
    return search
