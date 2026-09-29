"""Laço de treino, previsão e métricas."""
import copy
import math

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset


def calcular_metricas(mpg_real, mpg_previsto) -> dict:
    """RMSE, MAE e R² na unidade original (milhas por galão)."""
    if not np.all(np.isfinite(mpg_previsto)):
        return {"rmse": math.nan, "mae": math.nan, "r2": math.nan}
    return {
        "rmse": float(np.sqrt(mean_squared_error(mpg_real, mpg_previsto))),
        "mae": float(mean_absolute_error(mpg_real, mpg_previsto)),
        "r2": float(r2_score(mpg_real, mpg_previsto)),
    }


def prever(modelo, X, media_y: float, desvio_y: float):
    """Retorna (previsão padronizada, previsão em mpg)."""
    modelo.eval()
    with torch.no_grad():
        padronizada = modelo(torch.from_numpy(X)).numpy()
    return padronizada, padronizada * desvio_y + media_y


def treinar(modelo, dados: dict, cfg_treino: dict, semente: int, ao_fim_da_epoca=None) -> dict:
    """Treina a MLP e guarda os pesos da época com menor RMSE de validação."""
    gerador = torch.Generator().manual_seed(semente)
    carregador = DataLoader(
        TensorDataset(torch.from_numpy(dados["X_treino"]), torch.from_numpy(dados["y_treino"])),
        batch_size=int(cfg_treino["tamanho_lote"]),
        shuffle=True,
        generator=gerador,
    )
    criterio = torch.nn.MSELoss()
    otimizador = torch.optim.Adam(
        modelo.parameters(),
        lr=float(cfg_treino["taxa_aprendizado"]),
        weight_decay=float(cfg_treino["l2"]),  # regularização L2
    )

    historico = {"epoca": [], "train_loss": [], "val_loss": []}
    melhor = {"epoca": 0, "rmse": math.inf, "train_loss": math.nan, "val_loss": math.nan,
              "pesos": copy.deepcopy(modelo.state_dict())}
    status = "concluido"

    for epoca in range(1, int(cfg_treino["epocas"]) + 1):
        modelo.train()
        soma, total = 0.0, 0
        for xb, yb in carregador:
            otimizador.zero_grad()
            perda = criterio(modelo(xb), yb)
            perda.backward()
            otimizador.step()
            soma += perda.item() * len(xb)
            total += len(xb)
        train_loss = soma / total

        prev_pad, prev_mpg = prever(modelo, dados["X_val"], dados["media_y"], dados["desvio_y"])
        val_loss = float(np.mean((prev_pad - dados["y_val"]) ** 2))
        metricas_val = calcular_metricas(dados["mpg_val"], prev_mpg)

        historico["epoca"].append(epoca)
        historico["train_loss"].append(train_loss)
        historico["val_loss"].append(val_loss)

        if ao_fim_da_epoca:
            ao_fim_da_epoca(epoca, train_loss, val_loss, metricas_val)

        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            status = "divergiu"
            print(f"  Época {epoca}: perda não finita — treino interrompido (divergência).")
            break

        if metricas_val["rmse"] < melhor["rmse"]:
            melhor.update(epoca=epoca, rmse=metricas_val["rmse"], train_loss=train_loss,
                          val_loss=val_loss, pesos=copy.deepcopy(modelo.state_dict()))

    modelo.load_state_dict(melhor["pesos"])  # volta ao melhor momento do treino
    return {"historico": historico, "melhor": melhor, "status": status}
