"""Executa todas as configurações do experimento com um único comando.

Uso:
    python rodar_experimentos.py                       # roda A, B e C
    python rodar_experimentos.py --incluir-incidente   # roda A, B, C e D
"""
import argparse

from treinar import executar

CONFIGS_PRINCIPAIS = [
    "configs/run_a_referencia.yaml",
    "configs/run_b_taxa_menor.yaml",
    "configs/run_c_com_l2.yaml",
]
CONFIG_INCIDENTE = "configs/run_d_incidente.yaml"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roda as runs do experimento em sequência.")
    parser.add_argument("--incluir-incidente", action="store_true",
                        help="também roda a run D (taxa de aprendizado exagerada)")
    args = parser.parse_args()

    configs = CONFIGS_PRINCIPAIS + ([CONFIG_INCIDENTE] if args.incluir_incidente else [])
    for caminho in configs:
        executar(caminho)

    print("\nConcluído. Abra a interface com:")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("e acesse http://localhost:5000")
