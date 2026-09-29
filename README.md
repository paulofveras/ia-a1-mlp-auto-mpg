# Treinamento observável de uma MLP com MLflow — Auto MPG

Avaliação A1 da disciplina **Inteligência Artificial** (Sistemas de Informação — UNITINS, 2026/2).
Autor: **Paulo Fernando Mourão Veras Filho**.

Pipeline de treino, validação e teste de uma rede **Multi-Layer Perceptron (MLP) em PyTorch** para
prever o consumo de combustível de veículos, com rastreamento completo de cada execução no **MLflow**:
parâmetros, métricas por época, artefatos e traces das etapas.

> Vídeo de apresentação: **[link do YouTube — adicionar após a gravação]**

---

## 1. Problema

Ao treinar várias versões de uma rede neural mudando hiperparâmetros, é fácil perder o contexto de
cada resultado: qual configuração gerou qual métrica, onde está o gráfico, quais pesos correspondem
ao melhor momento do treino. Este projeto organiza o treino como um pipeline em que **cada execução é
uma run do MLflow**, ligando o desempenho às condições que o produziram
(configuração + dados + código + métricas + artefatos).

## 2. Dataset

- **Nome:** Auto MPG — UCI Machine Learning Repository (<https://archive.ics.uci.edu/dataset/9/auto+mpg>)
- **Tarefa:** regressão — prever `mpg` (consumo em milhas por galão)
- **Linhas:** 398 originais; 6 sem `horsepower` foram removidas → **392 amostras**
- **Entradas (9):**
  - numéricas, padronizadas: `cylinders`, `displacement`, `horsepower`, `weight`, `acceleration`, `model_year`
  - `origin` (1 = EUA, 2 = Europa, 3 = Japão) transformada em 3 colunas 0/1 (one-hot), por ser categórica
- **Removida:** `car_name` (texto livre, sem valor preditivo para a MLP)

## 3. Decisões metodológicas

| Decisão | Como foi feito | Por quê |
|---|---|---|
| Divisão | 70% treino / 15% validação / 15% teste (274/59/59) | Validação para escolher; teste reservado |
| Estratificação | Por quintis de `mpg` | Os três conjuntos ficam com consumo parecido |
| Pré-processamento | `StandardScaler` ajustado **só no treino** e aplicado em validação e teste | Evita vazamento de informação |
| Alvo | `mpg` padronizado no treino; métricas convertidas de volta para mpg | Treino estável; métricas interpretáveis |
| Reprodutibilidade | Sementes fixas (`random`, `numpy`, `torch`) e ordem dos lotes com gerador semeado | Mesma config → mesmo resultado |
| Melhor momento | Guarda os pesos da época com menor RMSE de validação (checkpoint) | Identifica o ponto de melhor generalização |
| Teste | Usado **uma única vez**, só na run escolhida (o script bloqueia reavaliação) | Estimativa honesta do desempenho final |

**Critério de escolha da melhor run (definido antes de rodar):**
1. menor `melhor_val_rmse`;
2. desempate: menor distância entre as curvas de treino e validação (`gap_val_menos_treino`) e curvas mais estáveis.

## 4. Experimentos

Arquitetura (`9 → 32 → 16 → 1`, ReLU), divisão, semente (42), lote (32) e épocas (150) são iguais em
todas as runs. Cada run muda uma coisa e responde a uma pergunta:

| Run | Taxa | L2 | Pergunta |
|---|---|---|---|
| A-referencia | 0.01 | 0 | Qual é o desempenho de base? |
| B-taxa-menor | 0.001 | 0 | Uma taxa menor gera curvas mais estáveis e menor erro de validação? |
| C-com-L2 | 0.01 | 1e-5 | A regularização L2 reduz a distância entre treino e validação? |
| D-incidente-taxa-alta | 0.5 | 0 | O que uma taxa exagerada causa nas curvas? (diagnóstico) |

## 5. O que cada run registra no MLflow

- **Parâmetros:** taxa, L2, épocas, lote, otimizador, função de perda, arquitetura, ativação, dropout, semente, split, estratificação, dataset
- **Tags:** pergunta da run, hash do commit do Git (versão do código), arquivo de configuração, status do treino
- **Métricas por época:** `train_loss`, `val_loss` (MSE no alvo padronizado), `val_rmse`, `val_mae`, `val_r2` (em mpg)
- **Métricas finais:** `melhor_epoca`, `melhor_val_rmse`, `melhor_val_mae`, `melhor_val_r2`, `gap_val_menos_treino`
- **Artefatos:** curva de aprendizado, previsto × real e resíduos da validação, resumo do pré-processamento (JSON), config usada, pesos da melhor época e o modelo PyTorch
- **Dataset:** registrado na aba *Datasets* da run
- **Trace:** span `pipeline_treino` com os filhos `preparar_dados`, `treinar` e `validar` (entradas, saídas, duração e erros)

## 6. Como reproduzir

Pré-requisitos: [Miniconda](https://docs.conda.io/en/latest/miniconda.html) e Git.

```bash
# 1. Clonar e entrar na pasta
git clone https://github.com/SEU-USUARIO/ia-a1-mlp-auto-mpg.git
cd ia-a1-mlp-auto-mpg

# 2. Criar e ativar o ambiente isolado
conda env create -f environment.yml
conda activate ia-a1-mlp

# 3. (Só se data/auto-mpg.csv não existir) baixar o dataset
python baixar_dados.py

# 4. Rodar as runs A, B e C com um único comando (acrescente --incluir-incidente para a D)
python rodar_experimentos.py --incluir-incidente

# 5. Abrir a interface do MLflow em http://localhost:5000
mlflow ui --backend-store-uri sqlite:///mlflow.db

# 6. Depois de escolher a melhor run pela validação, avaliar no teste (uma única vez)
python avaliar_teste.py --run-name NOME-DA-RUN-ESCOLHIDA
```

Para rodar só uma configuração: `python treinar.py --config configs/run_a_referencia.yaml`.
Para criar um novo experimento, basta copiar um YAML de `configs/` e mudar os valores — o código não precisa ser editado.

## 7. Estrutura

```
├── configs/                 # uma configuração YAML por run
├── data/auto-mpg.csv        # dataset versionado
├── docs/img/                # prints usados neste README
├── src/
│   ├── dados.py             # carga, divisão estratificada, scalers, one-hot
│   ├── modelo.py            # classe MLP em PyTorch
│   ├── treino.py            # laço de treino, checkpoint e métricas
│   ├── graficos.py          # curva, previsto × real, resíduos
│   └── utils.py             # config, sementes, hash do commit
├── treinar.py               # executa UMA configuração com rastreamento
├── rodar_experimentos.py    # executa todas as configurações
├── avaliar_teste.py         # avaliação única no teste da run escolhida
├── baixar_dados.py          # baixa o CSV do UCI
├── environment.yml / requirements.txt
```

## 8. Resultados

> Preencher após rodar os experimentos.

| Run | Melhor época | val RMSE (mpg) | val MAE (mpg) | val R² | Gap val − treino |
|---|---|---|---|---|---|
| A-referencia | | | | | |
| B-taxa-menor | | | | | |
| C-com-L2 | | | | | |
| D-incidente-taxa-alta | | | | | |

![Comparação das curvas no MLflow](docs/img/comparacao_curvas.png)

**Run escolhida:** _____ — justificativa: _____

**Teste (uma única avaliação):** RMSE ___ mpg | MAE ___ mpg | R² ___

**Ressalva:** a validação tem apenas 59 amostras; diferenças pequenas de RMSE entre runs podem ser ruído
estatístico e não devem ser tratadas como conclusivas.

## 9. Diagnóstico do incidente (run D)

- **Sintoma:** _descrever o que aparece nas curvas da run D (oscilação, explosão ou curva travada)._
- **Hipótese:** taxa de aprendizado alta demais faz cada atualização "passar do ponto" do mínimo (ou zera neurônios ReLU).
- **Nova run / verificação:** comparação com a run A, idêntica exceto pela taxa: _descrever o resultado._

## 10. Uso de IA generativa

Conforme exigido pela atividade, foi utilizada IA generativa (Claude, da Anthropic) como apoio para:
_descrever aqui, com suas palavras, em quais etapas a IA ajudou (ex.: estrutura do pipeline, código, revisão do README) e o que foi feito por você (execução, análise dos resultados, escolha da run, gravação do vídeo)._
