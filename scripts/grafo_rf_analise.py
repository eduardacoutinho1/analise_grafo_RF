"""
Análise de Grafo + Random Forest — Classificação de Sinais Preditivos
Eduarda — Maio 2026

Combina análise de grafo bipartido (eventos ↔ mensagens) com
classificador Random Forest para distinguir predições reais (sinais)
de coincidências semânticas (ruído).

Inputs:
  - matches_semanticos.csv (todos os matches semânticos)
  - sinais_qualificados.csv (136 preditivos com evidência adicional = ground truth)

Outputs:
  - features_grafo.csv (features extraídas do grafo para cada match)
  - rf_avaliacao.csv (métricas do classificador)
  - rf_importancia_features.csv (qual feature mais ajuda a separar sinal de ruído)
  - rf_criterio_selecao.csv (regras derivadas para classificar novos matches)
  - plot_grafo_bipartido.png (visualização do grafo)
  - plot_importancia.png (gráfico de importância das features)
  - plot_matriz_confusao.png (matriz de confusão do modelo)
  - plot_roc.png (curva ROC)
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, roc_auc_score, precision_recall_curve)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# CAMINHOS
BASE_DIR = r"C:\Users\user\Documents\pesquisa\cybersecurity\analise_cruzada"

# Pastas de entrada e saída
INPUT_DIR = BASE_DIR  # onde estão matches_semanticos.csv e sinais_qualificados.csv
OUTPUT_DIR = os.path.join(BASE_DIR, "output_grafo_rf")

MATCHES_CSV = os.path.join(INPUT_DIR, "matches_semanticos.csv")
SINAIS_CSV = os.path.join(INPUT_DIR, "sinais_qualificados.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Seed para reprodutibilidade
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def print_secao(titulo):
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)


# 1. CARREGAR DADOS
print_secao("1. CARREGANDO DADOS")

matches = pd.read_csv(MATCHES_CSV)
sinais = pd.read_csv(SINAIS_CSV)

print(f"   matches_semanticos: {len(matches)} matches totais")
print(f"      preditivo:   {(matches['tipo'] == 'preditivo').sum()}")
print(f"      repercussao: {(matches['tipo'] == 'repercussao').sum()}")
print(f"      simultaneo:  {(matches['tipo'] == 'simultaneo').sum()}")
print(f"   sinais_qualificados: {len(sinais)} preditivos com evidência (ground truth positivos)")


# 2. CONSTRUIR GRAFO BIPARTIDO
print_secao("2. CONSTRUINDO GRAFO BIPARTIDO (EVENTOS ↔ MENSAGENS)")

# Nós: event_id (eventos do Hackmageddon) e tg_msg_id (mensagens do Telegram)
# Arestas: cada match em matches_semanticos
# Peso da aresta: similaridade

G = nx.Graph()

# Adiciona nós de eventos (com atributo 'tipo' = 'event')
for event_id in matches["event_id"].unique():
    G.add_node(event_id, tipo="event")

# Adiciona nós de mensagens (com prefixo "M_" para evitar colisão de IDs)
for tg_msg_id in matches["tg_msg_id"].unique():
    G.add_node(f"M_{tg_msg_id}", tipo="message")

# Adiciona arestas (cada match)
for _, row in matches.iterrows():
    G.add_edge(
        row["event_id"],
        f"M_{row['tg_msg_id']}",
        similarity=row["similarity"],
        tipo_match=row["tipo"],
        delta_days=row["delta_days"],
    )

print(f"   Nós totais: {G.number_of_nodes()}")
print(f"      Eventos:   {sum(1 for n, d in G.nodes(data=True) if d['tipo'] == 'event')}")
print(f"      Mensagens: {sum(1 for n, d in G.nodes(data=True) if d['tipo'] == 'message')}")
print(f"   Arestas (matches): {G.number_of_edges()}")
print(f"   Densidade: {nx.density(G):.6f}")


# 3. EXTRAIR FEATURES DO GRAFO PARA CADA MATCH
print_secao("3. EXTRAINDO FEATURES DO GRAFO")

# Para cada nó, calcula:
# - degree: quantas arestas (matches) ele tem
event_degrees = {n: G.degree(n) for n, d in G.nodes(data=True) if d["tipo"] == "event"}
message_degrees = {n: G.degree(n) for n, d in G.nodes(data=True) if d["tipo"] == "message"}

# Para cada mensagem, calcular "breadth" = quantos tipos diferentes de ataque ela conecta
# (mensagem que cobre muitos tipos diferentes = mais provável ser genérica/ruído)
message_breadth = {}
for msg_node in [n for n, d in G.nodes(data=True) if d["tipo"] == "message"]:
    eventos_conectados = list(G.neighbors(msg_node))
    tipos_ataque = matches[matches["event_id"].isin(eventos_conectados)]["attack_norm"].unique()
    message_breadth[msg_node] = len(tipos_ataque)

# Adiciona features ao dataframe matches
matches["event_degree"] = matches["event_id"].map(event_degrees)
matches["message_degree"] = matches["tg_msg_id"].map(lambda x: message_degrees.get(f"M_{x}", 0))
matches["message_breadth"] = matches["tg_msg_id"].map(lambda x: message_breadth.get(f"M_{x}", 0))

print(f"   Features adicionadas:")
print(f"      event_degree:    grau do nó-evento (quantas msgs matcharam com ele)")
print(f"      message_degree:  grau do nó-mensagem (quantos eventos ela tocou)")
print(f"      message_breadth: quantos tipos diferentes de ataque a mensagem cobriu")

print(f"\n   Estatísticas das features:")
for col in ["event_degree", "message_degree", "message_breadth", "similarity"]:
    print(f"      {col:18}: min={matches[col].min()}, max={matches[col].max()}, média={matches[col].mean():.2f}")

# Salva features
matches_com_features = matches.copy()
matches_com_features.to_csv(os.path.join(OUTPUT_DIR, "features_grafo.csv"), index=False, encoding="utf-8-sig")
print(f"\n   Features salvas em: features_grafo.csv")


# 4. PREPARAR DADOS PARA O CLASSIFICADOR
print_secao("4. PREPARANDO DADOS PARA RANDOM FOREST")

# Marca os 136 sinais qualificados como classe positiva (1)
ids_qualificados = set(sinais["tg_msg_id"].astype(str) + "_" + sinais["event_id"].astype(str))
matches["match_key"] = matches["tg_msg_id"].astype(str) + "_" + matches["event_id"].astype(str)
matches["label"] = matches["match_key"].isin(ids_qualificados).astype(int)

# Só consideramos preditivos (faz sentido — não classifica repercussão como predição)
preditivos = matches[matches["tipo"] == "preditivo"].copy()

print(f"   Total de preditivos: {len(preditivos)}")
print(f"      Classe 1 (sinais reais):  {(preditivos['label'] == 1).sum()}")
print(f"      Classe 0 (sem evidência): {(preditivos['label'] == 0).sum()}")

# Balancear: amostra negativos = 3x os positivos (mantém alguma proporção sem desbalanceio extremo)
n_positivos = (preditivos["label"] == 1).sum()
positivos = preditivos[preditivos["label"] == 1]
negativos = preditivos[preditivos["label"] == 0].sample(n=min(n_positivos * 3, (preditivos["label"] == 0).sum()),
                                                       random_state=RANDOM_SEED)

dataset = pd.concat([positivos, negativos], ignore_index=True).sample(frac=1, random_state=RANDOM_SEED)
print(f"\n   Dataset final balanceado: {len(dataset)} matches")
print(f"      Classe 1: {(dataset['label'] == 1).sum()}")
print(f"      Classe 0: {(dataset['label'] == 0).sum()}")

# Features a usar
features_numericas = ["similarity", "lead_time", "event_degree", "message_degree", "message_breadth"]

# Codifica attack_norm (categórico) como número
le_attack = LabelEncoder()
dataset["attack_encoded"] = le_attack.fit_transform(dataset["attack_norm"].fillna("unknown").astype(str))
features_finais = features_numericas + ["attack_encoded"]

X = dataset[features_finais]
y = dataset["label"]

# Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
print(f"\n   Treino: {len(X_train)} matches")
print(f"   Teste:  {len(X_test)} matches")


# 5. TREINAR RANDOM FOREST
print_secao("5. TREINANDO RANDOM FOREST")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    class_weight="balanced",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

rf.fit(X_train, y_train)
print("   Modelo treinado.")

# Validação cruzada
cv_scores = cross_val_score(rf, X, y, cv=5, scoring="f1")
print(f"   Validação cruzada (5-fold) F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 6. AVALIAR
print_secao("6. AVALIANDO O MODELO")

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
auc = roc_auc_score(y_test, y_proba)

print(f"\n   Matriz de Confusão (no conjunto de teste):")
print(f"                     Predito 0    Predito 1")
print(f"      Real 0       {tn:>8}     {fp:>8}  ← falsos positivos")
print(f"      Real 1       {fn:>8}     {tp:>8}  ← verdadeiros positivos")

print(f"\n   Métricas:")
print(f"      Precisão:     {precision:.4f}  (dos que classificou como sinal real, quantos eram)")
print(f"      Recall:       {recall:.4f}  (dos sinais reais, quantos achou)")
print(f"      F1-Score:     {f1:.4f}  (média harmônica)")
print(f"      AUC-ROC:      {auc:.4f}  (área sob a curva ROC, 0.5=aleatório, 1.0=perfeito)")

# Salva métricas
metricas_df = pd.DataFrame([
    {"metrica": "Precisão", "valor": round(precision, 4)},
    {"metrica": "Recall", "valor": round(recall, 4)},
    {"metrica": "F1-Score", "valor": round(f1, 4)},
    {"metrica": "AUC-ROC", "valor": round(auc, 4)},
    {"metrica": "Verdadeiros Positivos", "valor": int(tp)},
    {"metrica": "Falsos Positivos", "valor": int(fp)},
    {"metrica": "Verdadeiros Negativos", "valor": int(tn)},
    {"metrica": "Falsos Negativos", "valor": int(fn)},
    {"metrica": "CV F1 (média)", "valor": round(cv_scores.mean(), 4)},
    {"metrica": "CV F1 (desvio)", "valor": round(cv_scores.std(), 4)},
])
metricas_df.to_csv(os.path.join(OUTPUT_DIR, "rf_avaliacao.csv"), index=False, encoding="utf-8-sig")

# 7. IMPORTÂNCIA DAS FEATURES
print_secao("7. IMPORTÂNCIA DAS FEATURES")

importancias = pd.DataFrame({
    "feature": features_finais,
    "importancia": rf.feature_importances_,
}).sort_values("importancia", ascending=False)

print(f"\n   Quanto cada feature ajuda a separar sinal real de ruído:")
for _, row in importancias.iterrows():
    pct = row["importancia"] * 100
    barra = "█" * int(pct / 2)
    print(f"      {row['feature']:20} {pct:5.1f}%  {barra}")

importancias.to_csv(os.path.join(OUTPUT_DIR, "rf_importancia_features.csv"), index=False, encoding="utf-8-sig")

# 8. CRITÉRIO DE SELEÇÃO (REGRAS DERIVADAS)
print_secao("8. CRITÉRIO DE SELEÇÃO DERIVADO")

# Para cada feature, calcula a média entre positivos e negativos
print(f"\n   Comparação: sinais reais vs ruído (no dataset balanceado)")
print(f"      {'Feature':20} {'Sinais reais':>14} {'Ruído':>14} {'Sinal/Ruído':>14}")
criterios = []
for f in features_numericas:
    media_pos = dataset[dataset["label"] == 1][f].mean()
    media_neg = dataset[dataset["label"] == 0][f].mean()
    ratio = media_pos / media_neg if media_neg != 0 else float("inf")
    print(f"      {f:20} {media_pos:>14.3f} {media_neg:>14.3f} {ratio:>14.2f}x")
    criterios.append({
        "feature": f,
        "media_sinais_reais": round(media_pos, 4),
        "media_ruido": round(media_neg, 4),
        "razao": round(ratio, 2),
    })

pd.DataFrame(criterios).to_csv(os.path.join(OUTPUT_DIR, "rf_criterio_selecao.csv"), index=False, encoding="utf-8-sig")

# 9. VISUALIZAÇÕES
print_secao("9. GERANDO VISUALIZAÇÕES")

# 9.1 Gráfico de importância das features
fig, ax = plt.subplots(figsize=(10, 5))
importancias_sorted = importancias.sort_values("importancia")
ax.barh(importancias_sorted["feature"], importancias_sorted["importancia"], color="steelblue")
ax.set_xlabel("Importância")
ax.set_title("Importância das Features (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_importancia.png"), dpi=120)
plt.close()
print("   plot_importancia.png salvo")

# 9.2 Matriz de confusão
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Ruído (0)", "Sinal real (1)"])
ax.set_yticklabels(["Ruído (0)", "Sinal real (1)"])
ax.set_xlabel("Predito pelo modelo")
ax.set_ylabel("Real")
ax.set_title("Matriz de Confusão")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=16)
plt.colorbar(im)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_matriz_confusao.png"), dpi=120)
plt.close()
print("   plot_matriz_confusao.png salvo")

# 9.3 Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatório")
ax.set_xlabel("Taxa de Falsos Positivos")
ax.set_ylabel("Taxa de Verdadeiros Positivos")
ax.set_title("Curva ROC")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_roc.png"), dpi=120)
plt.close()
print("   plot_roc.png salvo")

# 9.4 Visualização do grafo (subgrafo dos sinais qualificados)
print("   Gerando visualização do grafo (subgrafo dos sinais qualificados)...")

# Pega só eventos e mensagens dos sinais qualificados
eventos_sinais = sinais["event_id"].unique()
msgs_sinais = [f"M_{m}" for m in sinais["tg_msg_id"].unique()]
nos_sinais = list(eventos_sinais) + msgs_sinais
sub_G = G.subgraph(nos_sinais).copy()

fig, ax = plt.subplots(figsize=(14, 10))
pos = nx.spring_layout(sub_G, k=0.5, iterations=50, seed=RANDOM_SEED)


cores = ["steelblue" if sub_G.nodes[n]["tipo"] == "event" else "orange" for n in sub_G.nodes()]
tamanhos = [G.degree(n) * 15 + 50 for n in sub_G.nodes()]

nx.draw_networkx_edges(sub_G, pos, alpha=0.3, width=0.5)
nx.draw_networkx_nodes(sub_G, pos, node_color=cores, node_size=tamanhos, alpha=0.8)

ax.set_title(f"Grafo Bipartido — Sinais Qualificados ({len(eventos_sinais)} eventos, {len(msgs_sinais)} mensagens)")
ax.text(0.02, 0.98, "Azul: eventos do Hackmageddon\nLaranja: mensagens do Telegram\nTamanho do nó proporcional ao grau",
        transform=ax.transAxes, verticalalignment="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_grafo_bipartido.png"), dpi=120)
plt.close()
print("   plot_grafo_bipartido.png salvo")

# 10. RESUMO FINAL
print_secao("RESUMO FINAL")
print(f"\n   Modelo Random Forest treinado com {len(features_finais)} features")
print(f"   F1-Score no teste: {f1:.4f}   AUC: {auc:.4f}")
print(f"\n   Feature mais importante: {importancias.iloc[0]['feature']} ({importancias.iloc[0]['importancia']*100:.1f}%)")
print(f"\n   Arquivos gerados em: {OUTPUT_DIR}")
print(f"      Tabelas:")
print(f"        - features_grafo.csv")
print(f"        - rf_avaliacao.csv")
print(f"        - rf_importancia_features.csv")
print(f"        - rf_criterio_selecao.csv")
print(f"      Gráficos:")
print(f"        - plot_grafo_bipartido.png")
print(f"        - plot_importancia.png")
print(f"        - plot_matriz_confusao.png")
print(f"        - plot_roc.png")
print()