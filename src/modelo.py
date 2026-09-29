"""Rede neural Multi-Layer Perceptron (MLP) em PyTorch para regressão."""
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_entradas: int, camadas_ocultas: list, dropout: float = 0.0):
        super().__init__()
        camadas = []
        anterior = n_entradas
        for n_neuronios in camadas_ocultas:
            camadas.append(nn.Linear(anterior, n_neuronios))
            camadas.append(nn.ReLU())
            if dropout > 0:
                camadas.append(nn.Dropout(dropout))
            anterior = n_neuronios
        camadas.append(nn.Linear(anterior, 1))  # saída linear: um valor (mpg padronizado)
        self.rede = nn.Sequential(*camadas)

    def forward(self, x):
        return self.rede(x).squeeze(-1)
