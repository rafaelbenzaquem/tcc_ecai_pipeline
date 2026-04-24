# Pipeline de Análise de Distúrbios de Qualidade de Energia com Aprendizado de Máquina Não Supervisionado

**Trabalho de Conclusão de Curso — Rafael Benzaquem Neto**  
Programa ECAI 4.0 — Universidade Federal de Roraima (UFRR)

---

## Visão Geral

Este projeto implementa um **pipeline completo de análise automática de distúrbios elétricos** em sistemas de medição inteligente (smart metering), classificados segundo a norma **IEEE 1159:2019**.

O pipeline é totalmente não supervisionado: a partir de sinais brutos trifásicos em formato MATLAB, extrai representações latentes via **Autoencoder Convolucional (CAE)**, agrupa eventos via **K-Means** e rotula os agrupamentos semanticamente de acordo com faixas de tensão normalizadas. O resultado é um conjunto de eventos elétricos categorizados e prontos para auditoria técnica.

### Problema

Redes elétricas de alta tensão estão sujeitas a distúrbios transitórios (afundamentos, sobretensões, interrupções) que comprometem equipamentos industriais e a qualidade do fornecimento. A análise manual de milhares de registros de perturbação é inviável; este trabalho propõe uma solução automatizada baseada em aprendizado de máquina.

### Solução

```
Sinais brutos (.mat)
    └─► Pré-processamento        → tensores normalizados (p.u.)
        └─► Autoencoder CAE      → vetores latentes de 16 dimensões
            └─► K-Means          → 8 agrupamentos + detecção de anomalias
                └─► Curadoria Semântica → rótulos IEEE 1159:2019
```

---

## Estrutura do Repositório

```
pipeline-tcc/
│
├── 0_dados_brutos/                        # Dados de entrada (arquivos .mat)
│
├── 0_preprocessamento.ipynb              # Módulo 1 — Pré-processamento
├── 1_cae_autoencoder.ipynb               # Módulo 2 — Autoencoder Convolucional
├── 2_k_means_clustering.ipynb            # Módulo 3 — Clusterização K-Means
├── 3_curadoria_semantica.ipynb           # Módulo 4 — Curadoria Semântica
│
├── 1_dados_preprocessados/
│   └── exec_YYYYMMDD_HHMM_SPC_WS_NC_N/  # Uma pasta por execução
│       ├── X_cae_input.npy               # Tensor de entrada do CAE (N, 384, 3)
│       ├── dataset_X_pu.npy              # Tensões normalizadas em p.u.
│       ├── dataset_X_raw_kv.npy          # Tensões brutas em kV
│       ├── dataset_metadata.csv          # Metadados por evento
│       └── estatisticas_eventos.csv      # Estatísticas e classe heurística
│
├── 2_dados_cae/
│   └── exec_YYYYMMDD_HHMM_LD_F0_FN_EP/  # Uma pasta por execução
│       ├── cae_model.keras               # Modelo CAE completo
│       ├── encoder_model.keras           # Somente o encoder
│       ├── Z_latente.npy                 # Vetores latentes (N, 16)
│       ├── historico_treino.json         # Loss/MAE por época
│       ├── metricas_reconstrucao.csv     # RMSE/MAPE por evento e fase
│       └── *.png                         # Curvas de treinamento e visualizações
│
├── 3_dados_clustering/
│   └── exec_YYYYMMDD_HHMM_KMIN_KMAX_NINIT/  # Uma pasta por execução
│       ├── labels_kmeans.npy             # Rótulo de cluster por evento
│       ├── centroides_clusters.csv       # Centróides no espaço latente
│       ├── anomalias.csv                 # Eventos anômalos (P90)
│       ├── metricas_clustering.json      # Silhouette, Davies-Bouldin, CH
│       ├── Z_pca2d.npy / Z_tsne2d.npy   # Projeções 2D
│       ├── Z_tsne3d.npy                  # Projeção t-SNE 3D
│       └── *.png                         # Gráficos de elbow, silhueta, t-SNE
│
├── 4_dados_curadoria/
│   └── exec_YYYYMMDD_HHMM/               # Uma pasta por execução
│       ├── eventos_rotulados_hibrido.csv  # Eventos com rótulo IEEE e metadados
│       ├── anomalias_semanticas.csv       # Anomalias com análise semântica
│       ├── curadoria_clusters.csv         # Perfil estatístico por cluster
│       ├── curadoria_metadata.json        # Metadados da execução
│       ├── relatorio_curadoria_semantica.txt  # Relatório técnico textual
│       └── *.png                          # Painel visual, heatmap, formas de onda
│
├── requirements.txt                       # Dependências Python
├── Rafael_Neto_PROJETO_TCC_ECAI.pdf       # Documento de projeto
└── Rafael_Neto_TCC_FINAL_ECAI.pdf         # TCC final
```

> Cada módulo cria automaticamente um subdiretório com timestamp e hiperparâmetros no nome (ex: `exec_20260412_1800_16_32_128_100`), garantindo rastreabilidade completa entre execuções.

---

## Estágios do Pipeline

### Módulo 1 — Pré-processamento

**Notebook**: `0_preprocessamento.ipynb`  
**Execução de referência**: `1_dados_preprocessados/exec_20260408_1953_60_64_6_384/`

Converte os arquivos MATLAB brutos em tensores normalizados prontos para deep learning.

#### Dados de Entrada

- **Formato**: Arquivos `.mat` com estrutura de detecção (`matAux`)
- **Conteúdo**: Sinais de tensão trifásicos (fases A, B e C) de eventos elétricos
- **Total de arquivos processados**: 327 arquivos → **11.928 eventos distintos**

#### Parâmetros de Referência

| Parâmetro | Valor | Justificativa |
|---|---|---|
| Frequência nominal | 60 Hz | Rede brasileira |
| Amostras por ciclo | 64 | Resolução do sinal |
| Janela de análise | 6 ciclos (384 amostras) | Norma IEEE 1159:2019 |
| Tensão nominal de pico | 95,5 kV | Referência p.u. |

#### Processamento

1. Leitura dos arquivos `.mat` e extração de formas de onda trifásicas
2. Inferência automática da taxa de amostragem (amostras/ciclo)
3. Normalização para valores em por unidade: `V_pu = V_raw / V_nominal`
4. Extração de janela centrada: 6 ciclos em torno do início do distúrbio
5. Classificação heurística preliminar por V_rms
6. Exportação de metadados por evento (V_rms_min, duração, fase afetada etc.)

#### Dados de Saída

| Arquivo | Forma | Descrição |
|---|---|---|
| `X_cae_input.npy` | (11928, 384, 3) | Tensor de entrada para o CAE |
| `dataset_X_pu.npy` | (11928, 384, 3) | Tensões normalizadas em p.u. |
| `dataset_X_raw_kv.npy` | (11928, 384, 3) | Tensões brutas em kV |
| `dataset_metadata.csv` | (11928, 21) | Metadados tabulares por evento |
| `estatisticas_eventos.csv` | (11928, 8) | Classe heurística e estatísticas |

**Distribuição heurística dos eventos:**

| Classe | Eventos | Proporção |
|---|---|---|
| Sag (afundamento) | 11.917 | 99,9% |
| Interrupção | 11 | 0,1% |

---

### Módulo 2 — Autoencoder Convolucional (CAE)

**Notebook**: `1_cae_autoencoder.ipynb`  
**Execução de referência**: `2_dados_cae/exec_20260412_1800_16_32_128_100/`

Aprende uma representação compacta (espaço latente 16D) de cada forma de onda elétrica sem supervisão.

#### Arquitetura

```
Entrada: (384, 3) — timesteps × fases

ENCODER
  ├── Conv1D(32,  kernel=5, padding='same') → ReLU → MaxPool(2)  [→ 192, 32]
  ├── Conv1D(64,  kernel=5, padding='same') → ReLU → MaxPool(2)  [→  96, 64]
  └── Conv1D(128, kernel=3, padding='same') → ReLU → MaxPool(2)  [→  48, 128]
       │
       └── Flatten → Dense(16)  ← ESPAÇO LATENTE (gargalo)

DECODER
  ├── Dense → Reshape(48, 128)
  ├── UpSample(2) → Conv1D(128, kernel=3, padding='same') → ReLU  [→  96, 128]
  ├── UpSample(2) → Conv1D(64,  kernel=5, padding='same') → ReLU  [→ 192,  64]
  └── UpSample(2) → Conv1D(32,  kernel=5, padding='same') → ReLU  [→ 384,  32]
       │
       └── Conv1D(3, kernel=1, activation='linear')  → Saída: (384, 3)
```

#### Hiperparâmetros de Treinamento

| Parâmetro | Valor |
|---|---|
| Dimensão latente | 16 |
| Filtros encoder | [32, 64, 128] |
| Kernels Conv1D | [5, 5, 3] |
| Épocas | 100 |
| Batch size | 16 |
| Otimizador | Adam (lr inicial = 1e-3) |
| Divisão validação | 15% |
| Early stopping (paciência) | 15 épocas |
| Função de perda | MSE |

#### Resultados do Treinamento

| Métrica | Valor |
|---|---|
| Loss final — treino (MSE) | 0,000025 |
| Loss final — validação (MSE) | 0,000093 |
| Melhor val_loss | 0,000093 |
| RMSE médio de reconstrução | 0,004189 p.u. |
| RMSE mediano de reconstrução | 0,003573 p.u. |
| MAPE médio (todos os canais) | 1,18% |
| MAPE mediano (todos os canais) | 0,93% |

#### Dados de Saída

| Arquivo | Descrição |
|---|---|
| `cae_model.keras` | Autoencoder completo |
| `encoder_model.keras` | Somente o encoder (extração de features) |
| `Z_latente.npy` | Vetores latentes — (11928, 16) |
| `historico_treino.json` | Loss/MAE por época |
| `log_treinamento.csv` | Log detalhado por época |
| `metricas_reconstrucao.csv` | RMSE e MAPE por evento e por fase (Va, Vb, Vc) |
| `curvas_treinamento.png` | Curvas de loss e MAE |
| `distribuicao_rmse.png` | Histograma do erro de reconstrução |
| `latente_pca2d.png` | PCA 2D do espaço latente (diagnóstico) |

---

### Módulo 3 — Clusterização K-Means

**Notebook**: `2_k_means_clustering.ipynb`  
**Execução de referência**: `3_dados_clustering/exec_20260412_1904_2_8_50/`

Particiona os vetores latentes em grupos semânticos e detecta anomalias por distância intra-cluster.

#### Metodologia

1. **Padronização**: Z-score dos vetores latentes antes do K-Means
2. **Seleção de K**: Método do cotovelo (Elbow) para k = 2 a 8
3. **Treinamento**: K-Means++ com `n_init=50` inicializações robustas
4. **Avaliação**: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score
5. **Detecção de anomalias**: Eventos com distância ao centróide > percentil 90 (P90)
6. **Visualização**: PCA 2D, t-SNE 2D e t-SNE 3D

#### Hiperparâmetros

| Parâmetro | Valor |
|---|---|
| K ótimo (Elbow + Silhouette) | 8 |
| Faixa avaliada | k = 2 a 8 |
| n_init | 50 |
| Perplexidade t-SNE | 15 |
| Iterações t-SNE | 1.500 |
| Limiar de anomalia | Percentil 90 (distância intra-cluster) |

#### Resultados da Clusterização

| Métrica | Valor |
|---|---|
| K ótimo | 8 |
| Silhouette Score | 0,4981 |
| Davies-Bouldin Index | 0,6381 |
| Calinski-Harabasz Score | 11.333,32 |
| Total de eventos clusterizados | 11.928 |
| Anomalias detectadas (P90) | 1.196 (≈ 10,0%) |

**Distribuição por cluster:**

| Cluster | Eventos | Proporção |
|---|---|---|
| 0 | 1.764 | 14,8% |
| 1 | 2.428 | 20,4% |
| 2 | 1.102 | 9,2% |
| 3 | 981 | 8,2% |
| 4 | 993 | 8,3% |
| 5 | 1.169 | 9,8% |
| 6 | 1.693 | 14,2% |
| 7 | 1.798 | 15,1% |

#### Dados de Saída

| Arquivo | Descrição |
|---|---|
| `labels_kmeans.npy` | Rótulo de cluster por evento |
| `centroides_clusters.csv` | Centróides no espaço latente (16D) |
| `anomalias.csv` | Eventos flagrados (distância > P90) |
| `metricas_clustering.json` | Silhouette, Davies-Bouldin, Calinski-Harabasz |
| `Z_pca2d.npy` / `pca2d_clusters.png` | Projeção PCA 2D |
| `Z_tsne2d.npy` / `tsne2d_clusters.png` | Projeção t-SNE 2D |
| `Z_tsne3d.npy` / `tsne3d_clusters.png` | Projeção t-SNE 3D |
| `elbow_method.png` | Curva do cotovelo |
| `silhueta_amostras.png` | Coeficiente de silhueta por amostra |
| `heatmap_centroides.png` | Heatmap dos centróides no espaço latente |

---

### Módulo 4 — Curadoria Semântica

**Notebook**: `3_curadoria_semantica.ipynb`  
**Execução de referência**: `4_dados_curadoria/exec_20260412_1910/`

Mapeia os clusters para categorias da norma **IEEE 1159:2019** com base nas características de tensão RMS e dominância de fase, gerando rótulos informativos e um relatório técnico.

#### Classificação IEEE 1159:2019 (Afundamentos de Tensão)

| Categoria | Faixa V_rms | Descrição |
|---|---|---|
| Interrupção | 0,00 – 0,10 p.u. | Tensão praticamente nula |
| Afundamento Severo | 0,10 – 0,50 p.u. | Distúrbio crítico |
| Afundamento Moderado | 0,50 – 0,70 p.u. | Distúrbio intermediário |
| Afundamento Leve | 0,70 – 0,90 p.u. | Distúrbio suave |
| Tensão Normal | 0,90 – 1,10 p.u. | Operação normal |
| Swell Leve | 1,10 – 1,40 p.u. | Sobretensão leve |

#### Resultados da Curadoria

Todos os 8 clusters foram classificados como **Sag Moderado** (0,50–0,70 p.u.), com diferenciação pela fase elétrica dominante (B ou C). O ΔV_rms entre clusters é de apenas **0,0076 p.u.**, confirmando que os grupos capturam variações sutis dentro da mesma categoria IEEE — subcategorias de severidade e fase afetada.

| Cluster | Eventos | % | Rótulo | V_rms_min médio | V_rms_pré-falta | Fase | Duração |
|---|---|---|---|---|---|---|---|
| 0 | 1.928 | 15,9% | Sag Moderado — Fase B | 0,6957 ± 0,0748 p.u. | 0,7008 p.u. | B | 28,8 ms |
| 1 | 2.428 | 20,1% | Sag Moderado — Fase B | 0,6933 ± 0,0838 p.u. | 0,6994 p.u. | B | 28,9 ms |
| 2 | 1.102 | 9,1%  | Sag Moderado — Fase B | 0,6924 ± 0,0883 p.u. | 0,6987 p.u. | B | 29,0 ms |
| 3 | 981  | 8,1%  | Sag Moderado — Fase C | 0,6880 ± 0,1041 p.u. | 0,6960 p.u. | C | 29,3 ms |
| 4 | 993  | 8,2%  | Sag Moderado — Fase B | 0,6950 ± 0,0767 p.u. | 0,7001 p.u. | B | 29,6 ms |
| 5 | 1.169 | 9,7% | Sag Moderado — Fase B | 0,6922 ± 0,0890 p.u. | 0,6989 p.u. | B | 28,6 ms |
| 6 | 1.693 | 14,0% | Sag Moderado — Fase B | 0,6933 ± 0,0849 p.u. | 0,6981 p.u. | B | 29,4 ms |
| 7 | 1.798 | 14,9% | Sag Moderado — Fase C | 0,6948 ± 0,0787 p.u. | 0,7006 p.u. | C | 29,3 ms |

**Principais observações:**
- A separação entre clusters ocorre predominantemente pela **fase elétrica afetada** (B vs. C) e por **variações sutis na tensão de pré-falta** (ΔV_rms ≈ 0,0076 p.u.)
- O Cluster 3 apresenta **maior variabilidade** (σ = 0,1041 p.u.), indicando maior heterogeneidade interna
- **1.196 eventos** (≈ 9,9%) foram sinalizados como anomalias semânticas para revisão manual

#### Dados de Saída

| Arquivo | Descrição |
|---|---|
| `eventos_rotulados_hibrido.csv` | Todos os eventos com rótulo IEEE e metadados |
| `anomalias_semanticas.csv` | Anomalias com análise semântica detalhada |
| `curadoria_clusters.csv` | Perfil estatístico por cluster |
| `curadoria_metadata.json` | Metadados da execução (k, Silhouette, ΔV_rms etc.) |
| `relatorio_curadoria_semantica.txt` | Relatório técnico textual completo |
| `painel_curadoria.png` | Painel visual consolidado (4 visualizações) |
| `formas_onda_representativas.png` | Formas de onda do evento central por cluster |
| `heatmap_perfil_clusters.png` | Heatmap de perfil por cluster |

---

## Tecnologias Utilizadas

| Biblioteca | Uso |
|---|---|
| TensorFlow / Keras | Autoencoder convolucional (CAE) |
| scikit-learn | K-Means, PCA, t-SNE, métricas de clustering |
| NumPy | Manipulação de arrays e tensores |
| Pandas | Metadados e exportação CSV |
| Matplotlib | Visualizações e painéis |
| SciPy | Processamento de sinais (RMS, FFT) |
| pathlib | Gerenciamento dinâmico de caminhos |
| ipywidgets | Seleção interativa de execuções nos notebooks |

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

Execute os notebooks na ordem indicada pelos prefixos:

```
1. 0_preprocessamento.ipynb       → gera pasta em 1_dados_preprocessados/exec_.../
2. 1_cae_autoencoder.ipynb        → gera pasta em 2_dados_cae/exec_.../
3. 2_k_means_clustering.ipynb     → gera pasta em 3_dados_clustering/exec_.../
4. 3_curadoria_semantica.ipynb    → gera pasta em 4_dados_curadoria/exec_.../
```

Em cada notebook, a **Célula 3** exibe um seletor interativo (`ipywidgets.Dropdown`) para escolher a pasta de entrada gerada pelo módulo anterior. O diretório de saída é criado automaticamente com timestamp e hiperparâmetros no nome.

### Rastreabilidade entre Execuções

Os diretórios de saída seguem o padrão:

| Módulo | Padrão do diretório |
|---|---|
| Pré-processamento | `exec_{YYYYMMDD_HHMM}_{SPC}_{WS}_{NC}_{N}` |
| CAE | `exec_{YYYYMMDD_HHMM}_{LATENT_DIM}_{F_min}_{F_max}_{EPOCHS}` |
| K-Means | `exec_{YYYYMMDD_HHMM}_{K_MIN}_{K_MAX}_{N_INIT}` |
| Curadoria | `exec_{YYYYMMDD_HHMM}` |

---

## Resultados Principais

| Estágio | Indicador | Resultado |
|---|---|---|
| Pré-processamento | Arquivos .mat processados | 327 |
| Pré-processamento | Eventos extraídos | 11.928 |
| Pré-processamento | Classe predominante (heurística) | Sag (99,9%) |
| CAE | Loss treino final (MSE) | 0,000025 |
| CAE | Loss validação final (MSE) | 0,000093 |
| CAE | RMSE mediano de reconstrução | 0,003573 p.u. |
| CAE | MAPE mediano (todos os canais) | 0,93% |
| K-Means | K ótimo | 8 |
| K-Means | Silhouette Score | 0,4981 |
| K-Means | Davies-Bouldin Index | 0,6381 |
| K-Means | Calinski-Harabasz Score | 11.333,32 |
| K-Means | Anomalias detectadas (P90) | 1.196 (≈ 10,0%) |
| Curadoria | Clusters rotulados (IEEE 1159) | 8 / 8 |
| Curadoria | Categoria predominante | Sag Moderado (0,50–0,70 p.u.) |
| Curadoria | ΔV_rms entre clusters | 0,0076 p.u. |

---

## Decisões de Projeto

- **Aprendizado não supervisionado**: nenhum rótulo manual foi utilizado em nenhuma etapa do pipeline, tornando a abordagem aplicável a novos conjuntos de dados sem anotação prévia.
- **Gargalo latente de 16 dimensões**: dimensão escolhida empiricamente para balancear capacidade de representação e separabilidade dos clusters.
- **K=8 vs. K=4**: o aumento para 8 clusters (Silhouette=0,498; DBI=0,638; CH=11.333) superou o desempenho de K=4, capturando subcategorias de fase afetada (B vs. C) dentro da mesma classe IEEE.
- **Curadoria híbrida**: combinação de K-Means (agrupamento estrutural) com regras IEEE 1159 (interpretação de domínio), evitando tanto a caixa-preta pura quanto a dependência exclusiva de heurísticas manuais.
- **Inicialização robusta do K-Means**: `n_init=50` para reduzir sensibilidade à inicialização aleatória de centróides.
- **Anomalias por percentil**: limiar adaptativo (P90 por cluster) em vez de limiar fixo global, respeitando a heterogeneidade entre grupos.
- **Rastreabilidade por timestamp**: cada execução gera um subdiretório próprio com hiperparâmetros no nome, eliminando sobrescrita acidental de resultados.
- **Reprodutibilidade**: `random_state=42` em todas as etapas estocásticas.

---

## Referências Normativas

- **IEEE Std 1159-2019** — *IEEE Recommended Practice for Monitoring Electric Power Quality*. Classificação de distúrbios de qualidade de energia elétrica, incluindo faixas de afundamento de tensão, duração e categorias de severidade.

---

## Autor

**Rafael Benzaquem Neto**  
Programa ECAI 4.0 — Especialização em Ciência Aplicada e Inteligência Artificial  
Universidade Federal de Roraima (UFRR)
