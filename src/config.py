"""
Caminhos e constantes compartilhadas pelo projeto.

Centralizar os caminhos aqui evita que cada notebook/script use caminhos
relativos diferentes ('../data' vs 'data') e garante que todos os artefatos
sejam gravados sempre nas mesmas pastas exigidas pelo edital.
"""

from pathlib import Path

# Raiz do repositório (src/config.py -> src -> raiz)
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
IMAGES_DIR = ROOT_DIR / "images"
REPORTS_DIR = ROOT_DIR / "reports"
TABLES_DIR = REPORTS_DIR / "tabelas"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# Base analítica extraída da camada Gold (notebook 01)
ABT_PATH = RAW_DATA_DIR / "abt_alunos_alfabetizacao.parquet"

# Semente única usada em split, modelo e amostragem do SHAP
RANDOM_STATE = 42
TEST_SIZE = 0.2


def ensure_dirs() -> None:
    """Cria as pastas de saída caso ainda não existam."""
    for d in (IMAGES_DIR, REPORTS_DIR, TABLES_DIR, RAW_DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)
