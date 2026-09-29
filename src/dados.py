"""Carga do Auto MPG, divisão treino/validação/teste e pré-processamento.

Ordem obrigatória para evitar vazamento de informação:
1. dividir os dados;
2. ajustar os scalers SOMENTE com o conjunto de treino;
3. aplicar os scalers já ajustados na validação e no teste.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ALVO = "mpg"
NUMERICAS = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
CATEGORICA = "origin"  # 1 = EUA, 2 = Europa, 3 = Japão
CATEGORIAS_ORIGEM = [1, 2, 3]
NOMES_ORIGEM = ["origem_eua", "origem_europa", "origem_japao"]


def carregar_csv(caminho: str) -> pd.DataFrame:
    """Lê o CSV e remove as linhas com valores ausentes (6 linhas sem horsepower)."""
    df = pd.read_csv(caminho, na_values="?")
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
    df = df.dropna(subset=[ALVO] + NUMERICAS + [CATEGORICA]).reset_index(drop=True)
    return df


def _montar_entradas(df_parte: pd.DataFrame, scaler_x: StandardScaler) -> np.ndarray:
    """Padroniza as colunas numéricas e transforma origin em 3 colunas 0/1 (one-hot)."""
    numericas = scaler_x.transform(df_parte[NUMERICAS].values)
    origem = np.stack(
        [(df_parte[CATEGORICA].values == k).astype(np.float32) for k in CATEGORIAS_ORIGEM],
        axis=1,
    )
    return np.hstack([numericas, origem]).astype(np.float32)


def preparar_dados(caminho: str, semente: int, frac_validacao: float, frac_teste: float) -> dict:
    """Divide os dados e ajusta o pré-processamento só no treino.

    A divisão é estratificada por faixas (quintis) de mpg, para que treino,
    validação e teste tenham distribuições de consumo parecidas.
    """
    df = carregar_csv(caminho)
    faixas = pd.qcut(df[ALVO], q=5, labels=False).values
    indices = np.arange(len(df))

    frac_restante = frac_validacao + frac_teste
    idx_treino, idx_restante = train_test_split(
        indices, test_size=frac_restante, random_state=semente, stratify=faixas
    )
    idx_val, idx_teste = train_test_split(
        idx_restante,
        test_size=frac_teste / frac_restante,
        random_state=semente,
        stratify=faixas[idx_restante],
    )

    treino, val, teste = df.iloc[idx_treino], df.iloc[idx_val], df.iloc[idx_teste]

    # Scalers ajustados APENAS com o treino (evita vazamento de informação)
    scaler_x = StandardScaler().fit(treino[NUMERICAS].values)
    scaler_y = StandardScaler().fit(treino[[ALVO]].values)

    def alvo_padronizado(parte):
        return scaler_y.transform(parte[[ALVO]].values).ravel().astype(np.float32)

    resumo = {
        "arquivo": caminho,
        "linhas_validas": int(len(df)),
        "semente": semente,
        "divisao": {"treino": int(len(treino)), "validacao": int(len(val)), "teste": int(len(teste))},
        "estratificacao": "quintis de mpg",
        "mpg_por_conjunto": {
            nome: {"media": round(float(p[ALVO].mean()), 3), "desvio": round(float(p[ALVO].std()), 3)}
            for nome, p in (("treino", treino), ("validacao", val), ("teste", teste))
        },
        "entradas": NUMERICAS + NOMES_ORIGEM,
        "scaler_entradas": {
            col: {"media": round(float(m), 4), "desvio": round(float(s), 4)}
            for col, m, s in zip(NUMERICAS, scaler_x.mean_, scaler_x.scale_)
        },
        "scaler_alvo": {
            "media": round(float(scaler_y.mean_[0]), 4),
            "desvio": round(float(scaler_y.scale_[0]), 4),
        },
        "observacao": "scalers ajustados somente no conjunto de treino",
    }

    return {
        "X_treino": _montar_entradas(treino, scaler_x),
        "X_val": _montar_entradas(val, scaler_x),
        "X_teste": _montar_entradas(teste, scaler_x),
        "y_treino": alvo_padronizado(treino),
        "y_val": alvo_padronizado(val),
        "y_teste": alvo_padronizado(teste),
        "mpg_val": val[ALVO].values.astype(np.float32),
        "mpg_teste": teste[ALVO].values.astype(np.float32),
        "media_y": float(scaler_y.mean_[0]),
        "desvio_y": float(scaler_y.scale_[0]),
        "df_completo": df,
        "resumo": resumo,
    }
