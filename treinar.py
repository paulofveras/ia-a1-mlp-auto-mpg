"""Executa UMA configuração de treino e registra tudo no MLflow.

Uso:
    python treinar.py --config configs/run_a_referencia.yaml
"""
import argparse
import json
import math
import os
import tempfile

import mlflow
import torch

from src.dados import preparar_dados
from src.graficos import curva_aprendizado, previsto_vs_real, residuos
from src.modelo import MLP
from src.treino import calcular_metricas, prever, treinar
from src.utils import (NOME_EXPERIMENTO, URI_RASTREAMENTO, carregar_config,
                       fixar_sementes, hash_do_commit)


def _registrar_metricas(metricas: dict, passo: int) -> None:
    """Registra só valores finitos (o MLflow não aceita bem NaN/infinito)."""
    finitas = {k: v for k, v in metricas.items() if v is not None and math.isfinite(v)}
    if finitas:
        mlflow.log_metrics(finitas, step=passo)


def executar(caminho_config: str) -> str:
    cfg = carregar_config(caminho_config)
    semente = int(cfg["semente"])
    c_dados, c_modelo, c_treino = cfg["dados"], cfg["modelo"], cfg["treino"]

    fixar_sementes(semente)
    mlflow.set_tracking_uri(URI_RASTREAMENTO)
    mlflow.set_experiment(NOME_EXPERIMENTO)

    print(f"\n=== {cfg['nome_run']} ===")
    print(f"Pergunta: {cfg['pergunta']}")

    with mlflow.start_run(run_name=cfg["nome_run"]) as run:
        # ---------- Identidade da run: configuração + código ----------
        mlflow.set_tags({
            "pergunta": cfg["pergunta"],
            "mlflow.note.content": cfg["pergunta"],  # aparece como descrição na interface
            "commit_git": hash_do_commit(),
            "arquivo_config": caminho_config,
            "tarefa": "regressao (mpg)",
        })
        mlflow.log_artifact(caminho_config, artifact_path="config")

        with mlflow.start_span(name="pipeline_treino") as raiz:
            raiz.set_inputs({"config": caminho_config, "run": cfg["nome_run"]})

            # ---------- Etapa 1: preparar dados ----------
            with mlflow.start_span(name="preparar_dados") as span:
                span.set_inputs({"arquivo": c_dados["caminho"], "semente": semente,
                                 "frac_validacao": c_dados["frac_validacao"],
                                 "frac_teste": c_dados["frac_teste"]})
                dados = preparar_dados(c_dados["caminho"], semente,
                                       float(c_dados["frac_validacao"]), float(c_dados["frac_teste"]))
                span.set_outputs(dados["resumo"]["divisao"])

            n_entradas = dados["X_treino"].shape[1]
            divisao = dados["resumo"]["divisao"]
            mlflow.log_params({
                "taxa_aprendizado": c_treino["taxa_aprendizado"],
                "l2": c_treino["l2"],
                "epocas": c_treino["epocas"],
                "tamanho_lote": c_treino["tamanho_lote"],
                "otimizador": "Adam",
                "funcao_perda": "MSELoss",
                "arquitetura": f"{n_entradas} -> {' -> '.join(map(str, c_modelo['camadas_ocultas']))} -> 1",
                "ativacao": "ReLU",
                "dropout": c_modelo["dropout"],
                "semente": semente,
                "split": f"{divisao['treino']}/{divisao['validacao']}/{divisao['teste']} (treino/val/teste)",
                "estratificacao": "quintis de mpg",
                "dataset": "Auto MPG (UCI)",
            })

            try:  # registra o dataset na aba "Datasets" da run
                conjunto = mlflow.data.from_pandas(dados["df_completo"], source=c_dados["caminho"],
                                                   name="auto-mpg", targets="mpg")
                mlflow.log_input(conjunto, context="treino")
            except Exception as erro:
                print(f"  Aviso: dataset não registrado ({erro})")

            # ---------- Etapa 2: treinar ----------
            modelo = MLP(n_entradas, c_modelo["camadas_ocultas"], float(c_modelo["dropout"]))

            def ao_fim_da_epoca(epoca, train_loss, val_loss, m):
                _registrar_metricas({"train_loss": train_loss, "val_loss": val_loss,
                                     "val_rmse": m["rmse"], "val_mae": m["mae"], "val_r2": m["r2"]}, epoca)
                if epoca == 1 or epoca % 25 == 0:
                    print(f"  Época {epoca:3d} | train_loss {train_loss:.4f} | "
                          f"val_loss {val_loss:.4f} | val_rmse {m['rmse']:.3f} mpg")

            with mlflow.start_span(name="treinar") as span:
                span.set_inputs({"taxa_aprendizado": c_treino["taxa_aprendizado"], "l2": c_treino["l2"],
                                 "epocas": c_treino["epocas"], "tamanho_lote": c_treino["tamanho_lote"]})
                resultado = treinar(modelo, dados, c_treino, semente, ao_fim_da_epoca)
                melhor = resultado["melhor"]
                span.set_outputs({"status": resultado["status"], "melhor_epoca": melhor["epoca"],
                                  "epocas_executadas": len(resultado["historico"]["epoca"])})

            mlflow.set_tag("status_treino", resultado["status"])

            # ---------- Etapa 3: validar o melhor modelo ----------
            with mlflow.start_span(name="validar") as span:
                _, prev_val = prever(modelo, dados["X_val"], dados["media_y"], dados["desvio_y"])
                m_val = calcular_metricas(dados["mpg_val"], prev_val)
                gap = melhor["val_loss"] - melhor["train_loss"]
                _registrar_metricas({
                    "melhor_epoca": melhor["epoca"],
                    "melhor_val_rmse": m_val["rmse"],
                    "melhor_val_mae": m_val["mae"],
                    "melhor_val_r2": m_val["r2"],
                    "gap_val_menos_treino": gap,
                }, passo=0)
                span.set_outputs({k: (round(v, 4) if math.isfinite(v) else None) for k, v in m_val.items()})

            # ---------- Artefatos ----------
            with tempfile.TemporaryDirectory() as pasta:
                titulo = f"{cfg['nome_run']} (lr={c_treino['taxa_aprendizado']}, L2={c_treino['l2']})"
                curva_aprendizado(resultado["historico"], melhor["epoca"],
                                  os.path.join(pasta, "curva_aprendizado.png"), titulo)
                if math.isfinite(m_val["rmse"]):
                    previsto_vs_real(dados["mpg_val"], prev_val,
                                     os.path.join(pasta, "previsto_vs_real_validacao.png"),
                                     f"Validação – {cfg['nome_run']}")
                    residuos(dados["mpg_val"], prev_val, os.path.join(pasta, "residuos_validacao.png"),
                             f"Resíduos na validação – {cfg['nome_run']}")
                mlflow.log_artifacts(pasta, artifact_path="graficos")

                with open(os.path.join(pasta, "resumo_preprocessamento.json"), "w", encoding="utf-8") as f:
                    json.dump(dados["resumo"], f, ensure_ascii=False, indent=2)
                mlflow.log_artifact(os.path.join(pasta, "resumo_preprocessamento.json"),
                                    artifact_path="preprocessamento")

                caminho_pesos = os.path.join(pasta, "melhor_modelo.pt")
                torch.save(modelo.state_dict(), caminho_pesos)
                mlflow.log_artifact(caminho_pesos, artifact_path="checkpoint")

            try:
                mlflow.pytorch.log_model(modelo, name="modelo")
            except Exception as erro:
                print(f"  Aviso: log_model falhou ({erro}); os pesos continuam em checkpoint/")

            raiz.set_outputs({"melhor_val_rmse": m_val["rmse"] if math.isfinite(m_val["rmse"]) else None,
                              "status": resultado["status"]})

        print(f"  Melhor época: {melhor['epoca']} | val_rmse {m_val['rmse']:.3f} | "
              f"val_mae {m_val['mae']:.3f} | val_r2 {m_val['r2']:.3f}")
        print(f"  Run ID: {run.info.run_id}")
        return run.info.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina a MLP com uma configuração YAML.")
    parser.add_argument("--config", required=True, help="caminho do arquivo YAML em configs/")
    args = parser.parse_args()
    executar(args.config)
