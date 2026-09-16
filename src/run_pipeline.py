"""
Pipeline reproduzível de ponta a ponta.

Executa, na ordem, o mesmo fluxo dos notebooks 02 a 05 e grava os resultados
nas pastas exigidas pelo edital:

    images/          -> todas as figuras (EDA, ROC, feature importance, SHAP, risco)
    reports/tabelas/ -> métricas, importâncias e rankings em CSV/JSON/TXT

Uso (a partir da raiz do repositório):

    python -m src.run_pipeline

Os notebooks permanecem como o registro narrativo da análise; este script é a
versão executável e versionada do mesmo pipeline, para que qualquer pessoa
reproduza os números do README sem abrir o Jupyter.
"""

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import shap

from src import config
from src.evaluation.metrics import (
    evaluate_model,
    get_roc_curve_data,
    predict_municipality_risk,
    predict_proba_em_lotes,
    risco_por_regiao,
    taxa_prevista_por_grupo,
)
from src.modeling.train import BEST_PARAMS, build_model_pipeline
from src.preprocessing.pipeline import (
    COLS_TO_DROP,
    TARGET_COL,
    build_preprocessor,
    get_feature_names,
    load_data,
    prepare_features,
    split_train_test,
)
from src.visualization import plots

SHAP_SAMPLE_SIZE = 1500
TOP_N = 15
# Mínimo de alunos na amostra para considerar a estimativa municipal confiável
MIN_ALUNOS_ROBUSTO = 30


def _etapa(titulo: str) -> None:
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)


def main() -> dict:
    inicio = time.time()
    config.ensure_dirs()
    plots.configurar_estilo()
    resumo: dict = {
        "executado_em": datetime.now().isoformat(timespec="seconds"),
        "hiperparametros": BEST_PARAMS,
    }

    # ------------------------------------------------------------------
    _etapa("1/6 - Carregando a base analítica (camada Gold)")
    df = load_data(config.ABT_PATH)
    print(f"Linhas: {df.shape[0]} | Colunas: {df.shape[1]}")

    dist_target = df[TARGET_COL].value_counts(normalize=True).sort_index() * 100
    resumo["base"] = {
        "linhas": int(df.shape[0]),
        "colunas": int(df.shape[1]),
        "municipios_distintos": int(df["nome_municipio"].nunique()),
        "pct_alfabetizados": round(float(dist_target.loc[1]), 2),
        "pct_nao_alfabetizados": round(float(dist_target.loc[0]), 2),
    }
    print(f"Alfabetizados: {resumo['base']['pct_alfabetizados']}% | "
          f"Não alfabetizados: {resumo['base']['pct_nao_alfabetizados']}%")

    # ------------------------------------------------------------------
    _etapa("2/6 - Análise exploratória: gerando figuras e tabelas")
    df_eda = df.drop(columns=COLS_TO_DROP)

    plots.plot_distribuicao_target(df)
    plots.plot_taxa_por_rede(df_eda)
    plots.plot_taxa_por_regiao(df_eda)
    plots.plot_distribuicoes_numericas(df_eda)
    plots.plot_matriz_correlacao(df_eda)

    taxa_real_regiao = (
        df_eda.groupby("nome_regiao")[TARGET_COL]
        .agg(taxa_real="mean", alunos="count")
        .sort_values("taxa_real")
        .reset_index()
    )
    taxa_real_regiao["taxa_real_pct"] = taxa_real_regiao["taxa_real"] * 100
    taxa_real_regiao.to_csv(config.TABLES_DIR / "eda_taxa_real_por_regiao.csv",
                            index=False, encoding="utf-8")

    taxa_real_rede = (
        df_eda.groupby("rede")[TARGET_COL]
        .agg(taxa_real="mean", alunos="count")
        .sort_values("taxa_real")
        .reset_index()
    )
    taxa_real_rede["taxa_real_pct"] = taxa_real_rede["taxa_real"] * 100
    taxa_real_rede.to_csv(config.TABLES_DIR / "eda_taxa_real_por_rede.csv",
                          index=False, encoding="utf-8")

    cols_corr = plots.COLS_NUMERICAS_PORTE + ["w_peso_aluno", "ref_meta_taxa_ano"]
    corr = df_eda[[TARGET_COL] + cols_corr].astype(float).corr(method="spearman")
    corr.to_csv(config.TABLES_DIR / "eda_matriz_correlacao.csv", encoding="utf-8")

    resumo["eda"] = {
        "correlacao_spearman_com_target": {
            c: round(float(corr.loc[TARGET_COL, c]), 4) for c in cols_corr
        },
        "taxa_real_por_regiao_pct": {
            r["nome_regiao"]: round(float(r["taxa_real_pct"]), 2)
            for _, r in taxa_real_regiao.iterrows()
        },
        # O par (taxa, alunos) é importante: a rede Privada aparece com 1 único
        # aluno na amostra, então sua taxa não deve ser lida como um resultado.
        "taxa_real_por_rede_pct": {
            r["rede"]: {"taxa_pct": round(float(r["taxa_real_pct"]), 2),
                        "alunos": int(r["alunos"])}
            for _, r in taxa_real_rede.iterrows()
        },
    }
    print("Figuras da EDA geradas em images/")

    # ------------------------------------------------------------------
    _etapa("3/6 - Treinando a pipeline (pré-processamento + XGBoost)")
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    print(f"Treino: {X_train.shape[0]} linhas | Teste: {X_test.shape[0]} linhas")

    preprocessor = build_preprocessor(X_train)
    num_cols = X_train.select_dtypes(include=["int64", "float64", "Int64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    print(f"Numéricas: {num_cols}")
    print(f"Categóricas: {cat_cols}")

    model_pipeline = build_model_pipeline(preprocessor)
    model_pipeline.fit(X_train, y_train)
    print("Modelo treinado.")

    resumo["split"] = {
        "linhas_treino": int(X_train.shape[0]),
        "linhas_teste": int(X_test.shape[0]),
        "features_numericas": num_cols,
        "features_categoricas": cat_cols,
    }

    # ------------------------------------------------------------------
    _etapa("4/6 - Avaliando o modelo no conjunto de teste")
    metricas = evaluate_model(model_pipeline, X_test, y_test)

    fpr, tpr, _ = get_roc_curve_data(y_test, metricas["y_proba"])
    plots.plot_curva_roc(fpr, tpr, metricas["roc_auc"])

    (config.TABLES_DIR / "classification_report.txt").write_text(
        metricas["classification_report"], encoding="utf-8"
    )

    rel = metricas["classification_report_dict"]
    resumo["metricas"] = {
        "roc_auc": round(float(metricas["roc_auc"]), 4),
        "f1_weighted": round(float(metricas["f1_weighted"]), 4),
        "acuracia": round(float(rel["accuracy"]), 4),
        "classe_0_nao_alfabetizado": {
            k: round(float(v), 4) for k, v in rel["0"].items()
        },
        "classe_1_alfabetizado": {
            k: round(float(v), 4) for k, v in rel["1"].items()
        },
        "matriz_confusao": metricas["confusion_matrix"].tolist(),
    }
    with open(config.TABLES_DIR / "metricas_modelo.json", "w", encoding="utf-8") as f:
        json.dump(resumo["metricas"] | {"hiperparametros": BEST_PARAMS}, f,
                  indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    _etapa("5/6 - Interpretabilidade: Feature Importance e SHAP")
    feature_names = get_feature_names(
        model_pipeline.named_steps["preprocessor"], num_cols, cat_cols
    )
    importances = model_pipeline.named_steps["classifier"].feature_importances_
    feat_imp = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    feat_imp.head(20).to_csv(config.TABLES_DIR / "feature_importance_top20.csv",
                             index=False, encoding="utf-8")
    plots.plot_feature_importance(feat_imp)

    amostra_shap = X_test.sample(SHAP_SAMPLE_SIZE, random_state=config.RANDOM_STATE)
    amostra_transformada = pd.DataFrame(
        model_pipeline.named_steps["preprocessor"].transform(amostra_shap),
        columns=feature_names,
    )
    print(f"Calculando SHAP values em {SHAP_SAMPLE_SIZE} linhas...")
    explainer = shap.TreeExplainer(model_pipeline.named_steps["classifier"])
    shap_values = explainer.shap_values(amostra_transformada)

    plots.plot_shap_bar(shap_values, amostra_transformada)
    plots.plot_shap_beeswarm(shap_values, amostra_transformada)

    shap_imp = (
        pd.DataFrame({
            "feature": feature_names,
            "shap_medio_absoluto": np.abs(shap_values).mean(axis=0),
        })
        .sort_values("shap_medio_absoluto", ascending=False)
        .reset_index(drop=True)
    )
    shap_imp.head(20).to_csv(config.TABLES_DIR / "shap_importance_top20.csv",
                             index=False, encoding="utf-8")

    resumo["interpretabilidade"] = {
        "top10_feature_importance": feat_imp.head(10).to_dict(orient="records"),
        "top10_shap": shap_imp.head(10).to_dict(orient="records"),
    }
    print("Figuras de interpretabilidade geradas.")

    # ------------------------------------------------------------------
    _etapa("6/6 - Aplicação estratégica: risco de não atingir as metas")
    df_analise = X_test.copy()
    df_analise["y_real"] = y_test.values
    df_analise["prob_alfabetizado"] = metricas["y_proba"]
    df_analise["y_pred"] = metricas["y_pred"]

    taxa_regiao = taxa_prevista_por_grupo(df_analise, "nome_regiao")
    taxa_rede = taxa_prevista_por_grupo(df_analise, "rede")
    taxa_uf = taxa_prevista_por_grupo(df_analise, "sigla_uf")
    taxa_regiao.to_csv(config.TABLES_DIR / "taxa_prevista_por_regiao.csv",
                       index=False, encoding="utf-8")
    taxa_rede.to_csv(config.TABLES_DIR / "taxa_prevista_por_rede.csv",
                     index=False, encoding="utf-8")
    taxa_uf.to_csv(config.TABLES_DIR / "taxa_prevista_por_uf.csv",
                   index=False, encoding="utf-8")

    municipio_risco = predict_municipality_risk(
        model_pipeline, df, COLS_TO_DROP, TARGET_COL
    )
    municipio_risco.to_csv(config.TABLES_DIR / "municipios_taxa_prevista_vs_meta.csv",
                           index=False, encoding="utf-8")

    top_risco = municipio_risco[municipio_risco["em_risco"]].head(TOP_N)
    top_risco.to_csv(config.TABLES_DIR / f"top{TOP_N}_municipios_em_risco.csv",
                     index=False, encoding="utf-8")

    # Recorte robusto: municípios com massa amostral suficiente para a estimativa
    # ser confiável. O ranking geral inclui municípios com 1 ou 2 alunos na
    # amostra de 5%, cuja taxa prevista é instável demais para orientar política.
    robustos = municipio_risco[municipio_risco["total_alunos"] >= MIN_ALUNOS_ROBUSTO]
    robustos_em_risco = robustos[robustos["em_risco"]]
    robustos_em_risco.to_csv(
        config.TABLES_DIR / "municipios_em_risco_amostra_robusta.csv",
        index=False, encoding="utf-8"
    )

    regiao_risco = risco_por_regiao(municipio_risco)
    regiao_risco.to_csv(config.TABLES_DIR / "risco_por_regiao.csv",
                        index=False, encoding="utf-8")

    plots.plot_risco_por_regiao(regiao_risco)
    plots.plot_top_municipios_em_risco(top_risco, top_n=TOP_N)

    resumo["metas"] = {
        "municipios_com_meta": int(len(municipio_risco)),
        "municipios_em_risco": int(municipio_risco["em_risco"].sum()),
        "pct_em_risco": round(float(municipio_risco["em_risco"].mean() * 100), 1),
        "taxa_prevista_por_regiao_pct": {
            r["nome_regiao"]: round(float(r["taxa_prevista_pct"]), 2)
            for _, r in taxa_regiao.iterrows()
        },
        "taxa_prevista_por_rede_pct": {
            r["rede"]: round(float(r["taxa_prevista_pct"]), 2)
            for _, r in taxa_rede.iterrows()
        },
        "risco_por_regiao_pct": {
            r["nome_regiao"]: round(float(r["pct_em_risco"]), 1)
            for _, r in regiao_risco.iterrows()
        },
        "meta_media_por_regiao": {
            r: round(float(v), 2)
            for r, v in municipio_risco.groupby("nome_regiao")["meta"].mean().items()
        },
        f"top{TOP_N}_municipios": top_risco[
            ["nome_municipio", "sigla_uf", "nome_regiao", "taxa_prevista_pct",
             "meta", "gap_meta", "total_alunos"]
        ].round(2).to_dict(orient="records"),
        "amostra_robusta": {
            "min_alunos": MIN_ALUNOS_ROBUSTO,
            "municipios_avaliados": int(len(robustos)),
            "municipios_em_risco": int(len(robustos_em_risco)),
            "pct_em_risco": round(float(len(robustos_em_risco) / len(robustos) * 100), 1),
            f"top{TOP_N}_municipios": robustos_em_risco.head(TOP_N)[
                ["nome_municipio", "sigla_uf", "nome_regiao", "taxa_prevista_pct",
                 "meta", "gap_meta", "total_alunos"]
            ].round(2).to_dict(orient="records"),
        },
    }

    print(f"Municípios com meta: {resumo['metas']['municipios_com_meta']}")
    print(f"Municípios em risco: {resumo['metas']['municipios_em_risco']} "
          f"({resumo['metas']['pct_em_risco']}%)")

    # ------------------------------------------------------------------
    resumo["duracao_segundos"] = round(time.time() - inicio, 1)
    with open(config.TABLES_DIR / "resumo_execucao.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)

    _etapa(f"Concluído em {resumo['duracao_segundos']}s")
    print(f"Figuras  -> {config.IMAGES_DIR}")
    print(f"Tabelas  -> {config.TABLES_DIR}")
    return resumo


if __name__ == "__main__":
    main()
