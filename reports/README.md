# Relatórios

Documentação analítica do Tech Challenge Fase 3. Os relatórios consolidam os
resultados dos notebooks; as tabelas em [`tabelas/`](tabelas) e as figuras em
[`../images/`](../images) são geradas por `python -m src.run_pipeline`.

| Relatório | Conteúdo |
| --- | --- |
| [01 — Análise Exploratória](01_analise_exploratoria.md) | Distribuições, correlações e decisões derivadas da EDA |
| [02 — Modelagem e Avaliação](02_modelagem_e_avaliacao.md) | Pipeline, data leakage, validação e métricas |
| [03 — Interpretabilidade](03_interpretabilidade_shap.md) | Feature Importance, SHAP e perguntas de negócio |
| [04 — Municípios em Risco](04_municipios_em_risco.md) | Taxa prevista versus meta e priorização |
| [Documentação Técnica](documentacao_tecnica.md) | Arquitetura, dicionário de dados, decisões e reprodutibilidade |

## Tabelas geradas

| Arquivo | Conteúdo |
| --- | --- |
| `resumo_execucao.json` | Todos os números citados nos relatórios, com data da execução |
| `metricas_modelo.json` | ROC-AUC, F1, acurácia, matriz de confusão e hiperparâmetros |
| `classification_report.txt` | Relatório de classificação do Scikit-Learn |
| `eda_taxa_real_por_regiao.csv` · `eda_taxa_real_por_rede.csv` | Taxas observadas na base |
| `eda_matriz_correlacao.csv` | Correlação de Spearman entre numéricas e alvo |
| `feature_importance_top20.csv` | Top 20 pelo ganho interno do XGBoost |
| `shap_importance_top20.csv` | Top 20 por \|SHAP\| médio |
| `taxa_prevista_por_regiao.csv` · `_rede.csv` · `_uf.csv` | Taxa prevista agregada |
| `municipios_taxa_prevista_vs_meta.csv` | 4.890 municípios: taxa prevista, meta, gap e flag de risco |
| `top15_municipios_em_risco.csv` | Maiores déficits absolutos |
| `municipios_em_risco_amostra_robusta.csv` | Municípios em risco com ≥ 30 alunos na amostra |
