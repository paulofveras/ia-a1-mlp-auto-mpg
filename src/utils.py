"""Funções auxiliares: configuração, sementes e versão do código."""
import random
import subprocess

import numpy as np
import yaml

NOME_EXPERIMENTO = "a1-mlp-auto-mpg"
URI_RASTREAMENTO = "sqlite:///mlflow.db"


def carregar_config(caminho: str) -> dict:
    """Lê um arquivo YAML de configuração."""
    with open(caminho, "r", encoding="utf-8") as arquivo:
        return yaml.safe_load(arquivo)


def fixar_sementes(semente: int) -> None:
    """Fixa as sementes de todas as bibliotecas para a execução ser reproduzível."""
    import torch

    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)


def hash_do_commit() -> str:
    """Retorna o hash curto do commit atual do Git (versão do código usada na run)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "sem-git"
