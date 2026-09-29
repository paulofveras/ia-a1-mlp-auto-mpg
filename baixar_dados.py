"""Baixa o Auto MPG do UCI Machine Learning Repository e salva em data/auto-mpg.csv.

Rode uma única vez e faça commit do CSV, para o repositório não depender da internet.
"""
import os

import pandas as pd

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
COLUNAS = ["mpg", "cylinders", "displacement", "horsepower", "weight",
           "acceleration", "model_year", "origin", "car_name"]

if __name__ == "__main__":
    df = pd.read_csv(URL, names=COLUNAS, na_values="?", sep=r"\s+")
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/auto-mpg.csv", index=False)
    print(f"Salvo em data/auto-mpg.csv: {len(df)} linhas, "
          f"{int(df['horsepower'].isna().sum())} com horsepower ausente.")
