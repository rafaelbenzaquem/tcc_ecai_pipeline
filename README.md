# Pipeline de Análise de Distúrbios de Qualidade de Energia com Aprendizado de Máquina Não Supervisionado

**Trabalho de Conclusão de Curso — Rafael Benzaquem Neto**  
Programa ECAI 4.0 — Universidade Federal de Roraima (UFRR)

---

## Visão Geral

Este projeto implementa um **pipeline completo de análise automática de distúrbios elétricos** em sistemas de medição inteligente (smart metering), com foco em afundamentos de tensão (*voltage sags*) classificados segundo a norma **IEEE 1159:2019**.

O pipeline é totalmente não supervisionado: a partir de sinais brutos trifásicos em formato MATLAB, extrai representações latentes via **Autoencoder Convolucional (CAE)**, agrupa eventos via **K-Means** e rotula os agrupamentos semanticamente de acordo com faixas de tensão normalizadas. O resultado é um conjunto de eventos elétricos categorizados e prontos para auditoria técnica.

### Problema

Redes elétricas de alta tensão estão sujeitas a distúrbios transitórios (afundamentos, sobretensões, interrupções) que comprometem equipamentos industriais e a qualidade do fornecimento. A análise manual de milhares de registros de perturbação é inviável; este trabalho propõe uma solução automatizada baseada em aprendizado de máquina.

### Solução

```
Sinais brutos (.mat)
    └─► Pré-processamento        → tensores normalizados (p.u.)
        └─► Autoencoder CAE      → vetores latentes de 16 dimensões
            └─► K-Means          → 4 agrupamentos + detecção de anomalias
                └─► Curadoria Semântica → rótulos IEEE 1159:2019
```

---

## Estrutura do Repositório

```
pipeline-tcc/
│
├── 0_dados_brutos/                    # Dados de entrada (arquivos .mat)
│   ├── Data from Sub3/Results/        # 9 arquivos MATLAB — Sujeito 3
│   └── Data from Sub8/Results/        # Arquivos MATLAB — Sujeito 8
│
├── 0_preprocessamento_tcc.ipynb       # Estágio 1: Pré-processamento (v1)
├── 0_preprocessamento_tcc_v2.ipynb    # Estágio 1: Pré-processamento (v2 — atual)
│
├── 1_dados_preprocessados/            # Saídas do pré-processamento (v1)
├── 1_dados_preprocessados_v2/         # Saídas do pré-processamento (v2)
│   ├── X_cae_input.npy                # Tensor de entrada do CAE (54 MB)
│   ├── dataset_X_pu.npy               # Tensões em p.u. (107 MB)
│   ├── dataset_X_raw_kv.npy           # Tensões brutas em kV (107 MB)
│   └── dataset_metadata.csv           # Metadados por evento (2.5 MB)
│
├── 1_cae_autoencoder_tcc.ipynb        # Estágio 2: Treinamento do CAE
│
├── 2_dados_cae/                       # Saídas do CAE (v1)
├── 2_dados_cae_v2/                    # Saídas do CAE (v2 — atual)
│   ├── cae_model.keras                # Modelo completo (4.1 MB)
│   ├── encoder_model.keras            # Somente encoder (574 KB)
│   ├── Z_latente.npy                  # Vetores latentes 12.092 × 16
│   ├── historico_treino.json          # Curvas de loss/MAE
│   └── metricas_reconstrucao.csv      # RMSE por evento
│
├── 2_k_means_clustering_tcc.ipynb     # Estágio 3: Clusterização
│
├── 3_dados_clustering/                # Saídas da clusterização (v1)
├── 3_dados_clustering_v2/             # Saídas da clusterização (v2 — atual)
│   ├── labels_kmeans.npy              # Rótulos de cluster por evento
│   ├── centroides_clusters.csv        # Centróides dos clusters
│   ├── anomalias.csv                  # Eventos anômalos (P90)
│   ├── Z_pca2d.npy                    # Projeção PCA 2D
│   ├── Z_tsne2d.npy                   # Projeção t-SNE 2D
│   └── Z_tsne3d.npy                   # Projeção t-SNE 3D
│
├── 3_curadoria_semantica_v2.ipynb     # Estágio 4: Curadoria semântica
│
├── 4_dados_curadoria/                 # Saídas da curadoria (v1)
├── 4_dados_curadoria_v2/              # Saídas da curadoria (v2 — atual)
│   ├── eventos_rotulados_hibrido.csv  # Todos os eventos com rótulos IEEE
│   ├── anomalias_semanticas.csv       # Anomalias com análise semântica
│   ├── curadoria_clusters.csv         # Perfil por cluster
│   ├── curadoria_metadata.json        # Metadados da curadoria
│   ├── relatorio_curadoria_semantica.txt  # Relatório técnico
│   └── painel_curadoria.png           # Painel visual de resultados
│
├── requirements.txt                   # Dependências Python
└── Rafael_Neto_PROJETO_TCC_ECAI.docx  # Documento de projeto
```

---

## Estágios do Pipeline

### Estágio 0 — Pré-processamento

**Notebook**: `0_preprocessamento_tcc_v2.ipynb`

Converte os arquivos MATLAB brutos em tensores normalizados prontos para deep learning.

#### Dados de Entrada

- **Formato**: Arquivos `.mat` com estrutura de detecção (`matAux`)
- **Conteúdo**: Sinais de tensão trifásicos (fases A, B e C) de eventos elétricos
- **Sujeitos**: Sub3 e Sub8 (medições de campo de alta tensão)
- **Total de arquivos processados**: 327 arquivos → **12.092 eventos distintos**

#### Parâmetros de Referência

| Parâmetro | Valor | Justificativa |
|---|---|---|
| Frequência nominal | 60 Hz | Rede brasileira |
| Amostras por ciclo | 64 | Resolução do sinal |
| Janela de análise | 6 ciclos (384 amostras) | Norma IEEE 1159:2019 |
| Tensão nominal de pico | 95,5 kV | Referência p.u. |
| Faixa válida de tensão | 80–115 kV | Filtro de qualidade |

#### Processamento

1. Leitura dos arquivos `.mat` e extração de formas de onda
2. Inferência automática da taxa de amostragem (amostras/ciclo)
3. Cálculo de tensão RMS por ciclo (janela deslizante)
4. Normalização para valores em por unidade: `V_pu = V_raw / V_nominal`
5. Extração de janela: 6 ciclos centrados no início do afundamento
6. Validações de qualidade (faixa de tensão, completude de dados)
7. Exportação de metadados por evento (V_rms_min, duração, fase afetada etc.)

#### Dados de Saída

| Arquivo | Forma | Descrição |
|---|---|---|
| `dataset_X_pu.npy` | (12092, 384, 3) | Tensões normalizadas em p.u. |
| `dataset_X_raw_kv.npy` | (12092, 384, 3) | Tensões brutas em kV |
| `X_cae_input.npy` | (12092, 384, 3) | Tensor formatado para o CAE |
| `dataset_metadata.csv` | (12092, N) | Metadados tabulares por evento |

---

### Estágio 1 — Autoencoder Convolucional (CAE)

**Notebook**: `1_cae_autoencoder_tcc.ipynb`

Aprende uma representação compacta (espaço latente 16D) de cada forma de onda elétrica sem supervisão.

#### Arquitetura

```
Entrada: (384, 3) — timesteps × fases

ENCODER
  ├── Conv1D(64,  kernel=7, padding='same') → ReLU → MaxPool(2)  [→ 192, 64]
  ├── Conv1D(128, kernel=5, padding='same') → ReLU → MaxPool(2)  [→  96, 128]
  └── Conv1D(256, kernel=3, padding='same') → ReLU → MaxPool(2)  [→  48, 256]
       │
       └── Flatten → Dense(16)  ← ESPAÇO LATENTE (gargalo)

DECODER
  ├── Dense → Reshape(48, 256)
  ├── UpSample(2) → Conv1D(256, kernel=3, padding='same') → ReLU  [→  96, 256]
  ├── UpSample(2) → Conv1D(128, kernel=5, padding='same') → ReLU  [→ 192, 128]
  └── UpSample(2) → Conv1D(64,  kernel=7, padding='same') → ReLU  [→ 384,  64]
       │
       └── Conv1D(3, kernel=1, activation='linear')  → Saída: (384, 3)
```

#### Hiperparâmetros de Treinamento

| Parâmetro | Valor |
|---|---|
| Dimensão latente | 16 |
| Épocas máximas | 100 |
| Batch size | 16 |
| Otimizador | Adam |
| Taxa de aprendizado inicial | 1e-3 |
| Taxa de aprendizado final | 1e-7 |
| Agendamento de LR | Decaimento exponencial (9 etapas) |
| Divisão validação | 15% |
| Early stopping (paciência) | 15 épocas |
| Função de perda | MSE |

#### Resultados do Treinamento

| Métrica | Treinamento | Validação |
|---|---|---|
| Loss final (MSE) | 0,000023 | 0,000104 |
| MAE final | 0,003197 | 0,003161 |

- Convergência em torno das épocas 30–40
- Excelente generalização: `val_loss ≈ train_loss`
- RMSE mediano de reconstrução < 0,05 p.u.
- ~10% dos eventos apresentam erro elevado (candidatos a anomalias)

#### Dados de Saída

| Arquivo | Descrição |
|---|---|
| `cae_model.keras` | Autoencoder completo |
| `encoder_model.keras` | Somente o encoder (extração de features) |
| `Z_latente.npy` | Vetores latentes — (12092, 16) |
| `historico_treino.json` | Loss/MAE por época |
| `metricas_reconstrucao.csv` | RMSE por evento |

---

### Estágio 2 — Clusterização K-Means

**Notebook**: `2_k_means_clustering_tcc.ipynb`

Particiona os vetores latentes em grupos semânticos e detecta anomalias por distância intra-cluster.

#### Metodologia

1. **Seleção de K**: Método do cotovelo (Elbow) para k = 2 a 8
2. **Treinamento**: K-Means com `n_init=50` (inicializações robustas) e `max_iter=500`
3. **Avaliação**: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score
4. **Detecção de anomalias**: Pontos com distância ao centróide > percentil 90 (P90)
5. **Visualização**: PCA 2D, t-SNE 2D e t-SNE 3D

#### Hiperparâmetros

| Parâmetro | Valor |
|---|---|
| K ótimo escolhido | 4 (método do cotovelo) |
| K alternativo (Silhouette) | 7 |
| n_init | 50 |
| max_iter | 500 |
| Perplexidade t-SNE | 15 |
| Iterações t-SNE | 1500 |
| Limiar de anomalia | Percentil 90 (distância intra-cluster) |
| Random state | 42 |

#### Resultados da Clusterização

| Métrica | Valor |
|---|---|
| Silhouette Score | 0,48 |
| Davies-Bouldin Index | 0,88 |
| Calinski-Harabasz Score | 7.862,70 |
| Total de eventos clusterizados | 11.928 |
| Anomalias detectadas | 1.194 (≈ 10%) |

**Distribuição por cluster:**

| Cluster | Eventos | Proporção |
|---|---|---|
| 0 | 2.190 | 18,4% |
| 1 | 3.187 | 26,7% |
| 2 | 2.365 | 19,8% |
| 3 | 4.186 | 35,1% |

#### Dados de Saída

| Arquivo | Descrição |
|---|---|
| `labels_kmeans.npy` | Rótulo de cluster por evento |
| `centroides_clusters.csv` | Centróides no espaço latente |
| `anomalias.csv` | Eventos flagrados (distância > P90) |
| `Z_pca2d.npy` | Projeção PCA 2D |
| `Z_tsne2d.npy` | Projeção t-SNE 2D |
| `Z_tsne3d.npy` | Projeção t-SNE 3D |

---

### Estágio 3 — Curadoria Semântica

**Notebook**: `3_curadoria_semantica_v2.ipynb`

Mapeia os clusters para categorias da norma **IEEE 1159:2019** com base nas características de tensão RMS e dominância de fase, gerando rótulos informativos e um relatório técnico.

#### Classificação IEEE 1159:2019 (Afundamentos)

| Categoria | Faixa V_rms | Descrição |
|---|---|---|
| Afundamento Severo | 0,10 – 0,50 p.u. | Distúrbio crítico |
| Afundamento Moderado | 0,50 – 0,70 p.u. | Distúrbio intermediário |
| Afundamento Leve | 0,70 – 0,90 p.u. | Distúrbio suave |

#### Resultados da Curadoria

Todos os 4 clusters foram classificados como **Afundamento Moderado** (0,50–0,70 p.u.), com diferenciação por fase dominante:

| Cluster | Eventos | Rótulo | V_rms_min médio | Fase dominante | Duração média |
|---|---|---|---|---|---|
| 0 | 3.065 (25,3%) | Sag Moderado — Fase B | 0,564 ± 0,071 p.u. | B | 29,0 ms |
| 1 | 3.374 (27,9%) | Sag Moderado — Fase A | 0,573 ± 0,013 p.u. | A | 29,4 ms |
| 2 | 2.729 (22,6%) | Sag Moderado — Fase B | 0,557 ± 0,095 p.u. | B | 28,9 ms |
| 3 | 2.924 (24,2%) | Sag Moderado — Fase A | 0,563 ± 0,074 p.u. | A | 29,0 ms |

**Principais observações:**
- A separação entre clusters se dá pela **fase elétrica afetada** (A vs. B) e por **variações sutis na tensão de pré-falta** (ΔV_rms ≈ 0,016 p.u.)
- O Cluster 2 apresenta **maior variabilidade** (σ = 0,095 p.u.), indicando diversidade interna
- ~10% dos eventos foram sinalizados como anomalias semânticas para revisão manual

#### Dados de Saída

| Arquivo | Descrição |
|---|---|
| `eventos_rotulados_hibrido.csv` | Todos os eventos com rótulo IEEE e metadados |
| `anomalias_semanticas.csv` | Anomalias com análise semântica detalhada |
| `curadoria_clusters.csv` | Perfil estatístico por cluster |
| `curadoria_metadata.json` | Metadados da execução da curadoria |
| `relatorio_curadoria_semantica.txt` | Relatório técnico textual |
| `painel_curadoria.png` | Painel visual consolidado |

---

## Tecnologias Utilizadas

| Biblioteca | Versão | Uso |
|---|---|---|
| TensorFlow / Keras | ≥ 2.x | Autoencoder convolucional (CAE) |
| scikit-learn | ≥ 1.x | K-Means, PCA, t-SNE, métricas |
| NumPy | — | Manipulação de arrays e tensores |
| Pandas | — | Metadados e exportação CSV |
| Matplotlib | — | Visualizações e painéis |
| SciPy | — | Processamento de sinais (RMS, FFT) |

---

## Como Executar

### Pré-requisitos

```bash
python -m venv env
source env/bin/activate         # Linux/macOS
# env\Scripts\activate          # Windows

pip install -r requirements.txt
```

> **Nota sobre GPU**: O arquivo `requirements.txt` inclui `tensorflow[and-cuda]`. Para executar sem GPU, substitua por `tensorflow`.

### Execução Sequencial dos Notebooks

Execute os notebooks na ordem numérica indicada pelos prefixos:

```
1. 0_preprocessamento_tcc_v2.ipynb   → gera dados em 1_dados_preprocessados_v2/
2. 1_cae_autoencoder_tcc.ipynb        → gera modelos e vetores em 2_dados_cae_v2/
3. 2_k_means_clustering_tcc.ipynb     → gera clusters em 3_dados_clustering_v2/
4. 3_curadoria_semantica_v2.ipynb     → gera rótulos em 4_dados_curadoria_v2/
```

### Requisitos de Armazenamento

| Componente | Tamanho |
|---|---|
| Dados pré-processados (tensores) | ~268 MB |
| Modelos CAE + Encoder | ~4,7 MB |
| Vetores latentes | ~756 KB |
| Saídas de clustering | ~3,8 MB |
| Saídas de curadoria | ~5,8 MB |
| **Total estimado** | **~1,5 GB** |

---

## Resultados Principais

| Estágio | Indicador | Resultado |
|---|---|---|
| Pré-processamento | Eventos válidos processados | 12.092 |
| CAE — Treinamento | MAE de reconstrução (validação) | 0,003161 |
| CAE — Treinamento | MSE de reconstrução (validação) | 0,000104 |
| Clusterização | Silhouette Score | 0,48 |
| Clusterização | Anomalias detectadas | 1.194 (≈ 10%) |
| Curadoria | Clusters com rótulo IEEE 1159 | 4 / 4 |
| Curadoria | Categoria predominante | Sag Moderado (0,50–0,70 p.u.) |

---

## Decisões de Projeto

- **Aprendizado não supervisionado**: nenhum rótulo manual foi utilizado em nenhuma etapa do pipeline, tornando a abordagem aplicável a novos conjuntos de dados sem anotação prévia.
- **Gargalo latente de 16 dimensões**: dimensão escolhida empiricamente para balancear capacidade de representação e separabilidade dos clusters.
- **Curadoria híbrida**: combinação de K-Means (agrupamento estrutural) com regras IEEE 1159 (interpretação de domínio), evitando tanto a caixa-preta pura quanto a dependência exclusiva de heurísticas manuais.
- **Inicialização robusta do K-Means**: `n_init=50` para reduzir sensibilidade à inicialização aleatória de centróides.
- **Anomalias por percentil**: limiar adaptativo (P90 por cluster) em vez de limiar fixo global, respeitando a heterogeneidade entre grupos.
- **Decaimento exponencial de LR**: 9 etapas de redução da taxa de aprendizado para convergência estável sem mínimos locais grosseiros.
- **Reprodutibilidade**: `random_state=42` em todas as etapas estocásticas.

---

## Referências Normativas

- **IEEE Std 1159-2019** — *IEEE Recommended Practice for Monitoring Electric Power Quality*. Classificação de distúrbios de qualidade de energia elétrica, incluindo faixas de afundamento de tensão, duração e categorias de severidade.

---

## Autor

**Rafael Benzaquem Neto**  
Programa ECAI 4.0 — Especialização em Ciência Aplicada e Inteligência Artificial  
Universidade Federal de Roraima (UFRR)
