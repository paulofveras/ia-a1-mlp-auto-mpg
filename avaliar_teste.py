"""Avalia a run ESCOLHIDA no conjunto de teste, uma única vez.

O teste fica reservado até o fim: a escolha da melhor run é feita só com a validação.
Este script recria exatamente a mesma divisão dos dados (mesma config e semente),
carrega os pesos da melhor época e registra as métricas de teste na própria run.

Uso:
    python avaliar_teste.py --run-name A-referencia
"""
import argparse
import math
import os
import tempfile

import mlflow
import torch

from src.dados import preparar_dados
from src.graficos import previsto_vs_real, residuos
from src.modelo import MLP
from src.treino import calcular_metricas, prever
from src.utils import NOME_EXPERIMENTO, URI_RASTREAMENTO, carregar_config, fixar_sementes


def buscar_run(nome: str):
    runs = mlflow.search_runs(
        experiment_names=[NOME_EXPERIMENTO],
        filter_string=f"tags.mlflow.runName = '{nome}'",
        order_by=["attributes.start_time DESC"],
        output_format="list",
    )
    if not runs:
        raise SystemExit(f"Nenhuma run chamada '{nome}' no experimento '{NOME_EXPERIMENTO}'.")
    return runs[0]  # a mais recente com esse nome


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação final no teste da run escolhida.")
    parser.add_argument("--run-name", required=True, help="nome da run escolhida, ex.: A-referencia")
    parser.add_argument("--forcar", action="store_true",
                        help="permite reavaliar uma run já testada (evite: o teste deve ser usado uma vez)")
    args = parser.parse_args()

    mlflow.set_tracking_uri(URI_RASTREAMENTO)
    mlflow.set_experiment(NOME_EXPERIMENTO)
    run = buscar_run(args.run_name)
    run_id = run.info.run_id

    if run.data.tags.get("avaliado_no_teste") == "sim" and not args.forcar:
        raise SystemExit("Esta run já foi avaliada no teste. O teste deve ser usado uma única vez.")

    with tempfile.TemporaryDirectory() as pasta:
        # Recupera a configuração e os pesos salvos na própria run
        caminho_cfg_run = run.data.tags["arquivo_config"]
        pasta_cfg = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="config", dst_path=pasta)
        cfg = carregar_config(os.path.join(pasta_cfg, os.path.basename(caminho_cfg_run)))
        caminho_pesos = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="checkpoint/melhor_modelo.pt", dst_path=pasta)

        fixar_sementes(int(cfg["semente"]))

        with mlflow.start_run(run_id=run_id):
            with mlflow.start_span(name="testar") as span:
                span.set_inputs({"run": args.run_name, "run_id": run_id})
                c_dados = cfg["dados"]
                dados = preparar_dados(c_dados["caminho"], int(cfg["semente"]),
                                       float(c_dados["frac_validacao"]), float(c_dados["frac_teste"]))

                modelo = MLP(dados["X_teste"].shape[1], cfg["modelo"]["camadas_ocultas"],
                             float(cfg["modelo"]["dropout"]))
                modelo.load_state_dict(torch.load(caminho_pesos, map_location="cpu"))

                _, prev_teste = prever(modelo, dados["X_teste"], dados["media_y"], dados["desvio_y"])
                m = calcular_metricas(dados["mpg_teste"], prev_teste)
                span.set_outputs({k: (round(v, 4) if math.isfinite(v) else None) for k, v in m.items()})

            mlflow.log_metrics({"test_rmse": m["rmse"], "test_mae": m["mae"], "test_r2": m["r2"]})
            mlflow.set_tag("avaliado_no_teste", "sim")

            pasta_graf = os.path.join(pasta, "graficos_teste")
            os.makedirs(pasta_graf)
            previsto_vs_real(dados["mpg_teste"], prev_teste,
                             os.path.join(pasta_graf, "previsto_vs_real_teste.png"),
                             f"Teste – {args.run_name}")
            residuos(dados["mpg_teste"], prev_teste, os.path.join(pasta_graf, "residuos_teste.png"),
                     f"Resíduos no teste – {args.run_name}")
            mlflow.log_artifacts(pasta_graf, artifact_path="teste")

    print(f"\nTeste da run {args.run_name} ({dados['resumo']['divisao']['teste']} amostras):")
    print(f"  RMSE {m['rmse']:.3f} mpg | MAE {m['mae']:.3f} mpg | R² {m['r2']:.3f}")
