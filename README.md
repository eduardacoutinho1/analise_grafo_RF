# Análise de Grafo Bipartido + Random Forest para Classificação de Sinais Preditivos

## Visão geral

Este repositório contém a implementação de uma análise híbrida (teoria de grafos + aprendizado de máquina supervisionado) para distinguir **sinais preditivos reais** de **ruído semântico** em correspondências entre mensagens do Telegram e incidentes de cibersegurança registrados no Hackmageddon.

O problema parte de uma base de matches semânticos já produzida (`matches_semanticos.csv`) e usa um subconjunto qualificado com evidências adicionais (`sinais_qualificados.csv`) como ground truth para treinar um classificador.

## Estrutura do repositório

```
analise-grafo-rf/
├── README.md                  → este arquivo
├── requirements.txt           → bibliotecas Python necessárias
├── scripts/
│   └── grafo_rf_analise.py    → script principal (executável)
├── inputs/
│   ├── matches_semanticos.csv      → 6.018 matches semânticos completos
│   ├── sinais_qualificados.csv     → 136 sinais com evidência adicional (ground truth)
│   └── cruzamento_diario.csv       → cruzamento diário das três fontes OSINT
└── outputs/
    ├── tabelas/
    │   ├── features_grafo.csv               → matches enriquecidos com features de grafo
    │   ├── rf_avaliacao.csv                 → métricas do classificador
    │   ├── rf_importancia_features.csv      → importância relativa de cada feature
    │   └── rf_criterio_selecao.csv          → razões sinal/ruído por feature
    └── graficos/
        ├── plot_grafo_bipartido.png         → visualização do grafo (sinais qualificados)
        ├── plot_importancia.png             → importância das features
        ├── plot_matriz_confusao.png         → matriz de confusão no conjunto de teste
        └── plot_roc.png                     → curva ROC
```

## Metodologia

### Etapa 1 — Construção do grafo bipartido

Foi construído um grafo bipartido **G = (E ∪ M, A)** onde:
- **E** = conjunto de eventos do Hackmageddon que possuem ao menos um match semântico
- **M** = conjunto de mensagens do Telegram que possuem ao menos um match semântico
- **A** = arestas conectando evento e mensagem sempre que há match em `matches_semanticos.csv`, ponderadas pela similaridade

A biblioteca utilizada foi `networkx`.

### Etapa 2 — Engenharia de features de grafo

Para cada match, foram derivadas três features adicionais a partir da estrutura do grafo:

| Feature | Descrição | Hipótese |
|---|---|---|
| `event_degree` | Grau do nó-evento (quantas mensagens distintas matcharam com ele) | Eventos discutidos por múltiplas fontes têm mais chance de serem reais |
| `message_degree` | Grau do nó-mensagem (quantos eventos distintos ela tocou) | Hipótese inicial: alto grau = ruído (genérico). Resultado refutou (ver "Achados") |
| `message_breadth` | Quantidade de tipos de ataque distintos cobertos por uma mensagem | Mensagens específicas tendem a cobrir poucos tipos |

### Etapa 3 — Classificação supervisionada com Random Forest

Os 136 registros em `sinais_qualificados.csv` foram tomados como classe positiva (sinal real). Como negativos, foi feita amostragem aleatória estratificada dos demais 2.719 matches do tipo `preditivo` sem evidência adicional, na razão 3:1 (negativos:positivos).

Configuração do modelo:
- 200 árvores (`n_estimators=200`)
- Profundidade máxima 8 (`max_depth=8`)
- `class_weight="balanced"` para mitigar desbalanceio remanescente
- Validação cruzada k-fold com k=5
- Seed fixa (42) para reprodutibilidade

Features usadas: `similarity`, `lead_time`, `event_degree`, `message_degree`, `message_breadth`, `attack_norm` (codificado por `LabelEncoder`).

Split: 80% treino / 20% teste, estratificado por classe.

## Resultados

### Métricas do classificador

| Métrica | Valor | Interpretação |
|---|---|---|
| AUC-ROC | 0.877 | Discriminação muito acima do aleatório (0.5) |
| Recall | 0.81 | 81% dos sinais reais foram identificados |
| Precisão | 0.51 | Aproximadamente metade dos sinais flagrados são verdadeiros positivos |
| F1-Score | 0.63 | Média harmônica entre precisão e recall |
| CV F1 (5-fold) | 0.68 ± 0.03 | Resultado estável em diferentes divisões dos dados |

### Importância das features

```
similarity         55.9%  ███████████████████████████
lead_time          16.8%  ████████
message_degree     11.9%  █████
attack_encoded      7.6%  ███
event_degree        3.9%  █
message_breadth     3.9%  █
```

As três features derivadas do grafo somam **~19,7%** da decisão do modelo, justificando a inclusão da análise de grafo no pipeline.

### Achado metodológico

A hipótese inicial de que **alto `message_degree` indica ruído** (mensagens genéricas que casam com muitos eventos) **foi refutada pelos dados**:

| Feature | Média (sinais reais) | Média (ruído) | Razão |
|---|---|---|---|
| `similarity` | 0.604 | 0.535 | 1.13× |
| `lead_time` | 8.47 | 7.20 | 1.18× |
| `message_degree` | 12.68 | 8.00 | **1.58×** |
| `message_breadth` | 2.16 | 1.90 | 1.14× |
| `event_degree` | 4.96 | 4.58 | 1.08× |

Sinais reais apresentam grau de mensagem 58% maior que o ruído. A interpretação consistente com os dados é: mensagens que mencionam ameaças específicas (ex: CVEs, famílias de ransomware) tendem a corresponder a múltiplos eventos relacionados à **mesma** ameaça — formando clusters densos e não dispersos. A feature `message_breadth` (1.14×) reforça essa interpretação: o grau alto não é acompanhado por dispersão temática.

### Critério de seleção derivado

Com base nas razões sinal/ruído observadas, uma heurística inicial para classificação de novos matches seria:

- `similarity ≥ 0.60`
- `lead_time` entre 7 e 14 dias
- `message_degree ≥ 10`

Essa heurística é uma aproximação das fronteiras de decisão aprendidas pelo Random Forest e pode ser usada como filtro inicial em pipelines de detecção precoce.

## Como reproduzir

### Pré-requisitos

- Python 3.11 ou superior
- Bibliotecas listadas em `requirements.txt`

### Instalação

```bash
pip install -r requirements.txt
```

### Execução

1. Ajuste a variável `BASE_DIR` no início do script `scripts/grafo_rf_analise.py` para o caminho local do repositório.
2. Execute:

```bash
python scripts/grafo_rf_analise.py
```

## Limitações reconhecidas

- A amostragem negativa (3:1) é uma escolha metodológica que afeta as métricas finais. Razões diferentes (1:1, ou uso de todos os negativos com pesos de classe) podem ser exploradas.
- O modelo foi treinado apenas com matches do tipo `preditivo`; matches `repercussao` e `simultaneo` não foram incluídos.
- A configuração do Random Forest não passou por busca de hiperparâmetros sistemática (GridSearch ou RandomizedSearch). Os valores usados são padrões razoáveis baseados na literatura.
- Outros algoritmos (XGBoost, Logistic Regression, SVM) não foram comparados nesta primeira versão.


## Reprodutibilidade

- Seed fixa (`RANDOM_SEED = 42`) em todas as etapas estocásticas
- Versão do Python documentada
- Bibliotecas com versões em `requirements.txt`
- Inputs preservados em `inputs/` no estado em que foram utilizados

## Referências

- Saeed, M. H., & Huang, H. (2025). *SENTINEL: A Multi-Modal Early Detection Framework for Emerging Cyber Threats using Telegram.* arXiv preprint arXiv:2512.21380.
- Pipeline de coleta do Hackmageddon: [github.com/gabrierys/hackmageddon](https://github.com/gabrierys/hackmageddon)
- Coleta de dados do Telegram: [github.com/lyMartins/Analise-Database-Cybersec](https://github.com/lyMartins/Analise-Database-Cybersec)

## Licença

Este trabalho faz parte de pesquisa acadêmica em andamento no Synapse Lab / UNIFOR.
