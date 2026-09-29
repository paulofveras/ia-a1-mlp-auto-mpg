"""Gráficos salvos como artefatos de cada run."""
import matplotlib

matplotlib.use("Agg")  # gera as imagens sem abrir janela
import matplotlib.pyplot as plt
import numpy as np


def curva_aprendizado(historico: dict, melhor_epoca: int, caminho: str, titulo: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(historico["epoca"], historico["train_loss"], label="treino (train_loss)")
    ax.plot(historico["epoca"], historico["val_loss"], label="validação (val_loss)")
    if melhor_epoca > 0:
        ax.axvline(melhor_epoca, color="gray", linestyle="--", label=f"melhor época ({melhor_epoca})")
    ax.set_xlabel("Época")
    ax.set_ylabel("Perda MSE (mpg padronizado, escala log)")
    ax.set_yscale("log")
    ax.set_title(titulo)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def previsto_vs_real(y_real, y_prev, caminho: str, titulo: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_real, y_prev, alpha=0.7)
    limites = [min(np.min(y_real), np.min(y_prev)), max(np.max(y_real), np.max(y_prev))]
    ax.plot(limites, limites, "k--", label="previsão perfeita")
    ax.set_xlabel("mpg real")
    ax.set_ylabel("mpg previsto")
    ax.set_title(titulo)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def residuos(y_real, y_prev, caminho: str, titulo: str) -> None:
    erro = np.asarray(y_real) - np.asarray(y_prev)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_prev, erro, alpha=0.7)
    ax.axhline(0, color="k", linestyle="--")
    ax.set_xlabel("mpg previsto")
    ax.set_ylabel("resíduo (real - previsto)")
    ax.set_title(titulo)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
