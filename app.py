# app.py — исследовательская утилита для анализа эмбеддингов

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sentence_transformers import util

from utils import (
    preprocess_text,
    read_uploaded_file_bytes,
    jaccard_tokens,
    style_suspicious_and_low,
    load_model,
    encode_texts_in_batches,
    precision_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    cluster_embeddings,
)

# ================= Streamlit config =================
st.set_page_config(page_title="Embedding Analyzer", layout="wide")
st.title("🔬 Embedding Research Utility")

# ================= Sidebar =================
st.sidebar.header("Модель")
model_id = st.sidebar.text_input("HF Model ID", value="sentence-transformers/all-MiniLM-L6-v2")
batch_size = st.sidebar.number_input("Batch size", 8, 1024, 64, 8)

st.sidebar.header("Пороги")
semantic_threshold = st.sidebar.slider("Semantic (>=)", 0.0, 1.0, 0.8, 0.01)
lexical_threshold = st.sidebar.slider("Lexical (<=)", 0.0, 1.0, 0.3, 0.01)
low_score_threshold = st.sidebar.slider("Low score (<)", 0.0, 1.0, 0.75, 0.01)

# ================= Load model =================
try:
    with st.spinner("Загружаю модель..."):
        model = load_model(model_id)
    st.sidebar.success("Модель загружена")
except Exception as e:
    st.sidebar.error(f"Ошибка загрузки: {e}")
    st.stop()

# ================= Upload =================
st.header("1. Данные")
uploaded_file = st.file_uploader("Файл (CSV/XLSX/JSON)", type=["csv", "xlsx", "xls", "json", "ndjson"])
if uploaded_file is None:
    st.info("Загрузите файл для начала.")
    st.stop()

try:
    df, file_hash = read_uploaded_file_bytes(uploaded_file)
except Exception as e:
    st.error(f"Ошибка чтения файла: {e}")
    st.stop()

if not {"phrase_1", "phrase_2"}.issubset(df.columns):
    st.error("Файл должен содержать колонки phrase_1, phrase_2")
    st.stop()

# ================= Preprocess =================
st.subheader("Редактирование датасета")
edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="dataset_editor")
df = edited_df.copy()
df["phrase_1"] = df["phrase_1"].map(preprocess_text)
df["phrase_2"] = df["phrase_2"].map(preprocess_text)

# ================= Embeddings =================
phrases_all = list(set(df["phrase_1"].tolist() + df["phrase_2"].tolist()))
phrase2idx = {p: i for i, p in enumerate(phrases_all)}

with st.spinner("Кодирую фразы..."):
    embeddings = encode_texts_in_batches(model, phrases_all, batch_size)

# compute pair scores
scores, lexical_scores = [], []
for _, row in df.iterrows():
    p1, p2 = row["phrase_1"], row["phrase_2"]
    emb1, emb2 = embeddings[phrase2idx[p1]], embeddings[phrase2idx[p2]]
    score = float(util.cos_sim(emb1, emb2).item())
    scores.append(score)
    lexical_scores.append(jaccard_tokens(p1, p2))

df["score"] = scores
df["lexical_score"] = lexical_scores

# ================= Tabs =================
st.header("2. Аналитика")
tabs = st.tabs(["Сводка", "Explore", "Retrieval Metrics", "Semantic Search", "Clustering"])

# --- Summary ---
with tabs[0]:
    total = len(df)
    low_cnt = int((df["score"] < low_score_threshold).sum())
    susp_cnt = int(((df["score"] >= semantic_threshold) & (df["lexical_score"] <= lexical_threshold)).sum())
    colA, colB, colC = st.columns(3)
    colA.metric("Pairs", total)
    colB.metric("Mean score", f"{df['score'].mean():.4f}")
    colC.metric("Suspicious", susp_cnt)

    st.markdown("### Подсветка")
    st.dataframe(style_suspicious_and_low(df, semantic_threshold, lexical_threshold, low_score_threshold),
                 use_container_width=True)

# --- Explore ---
with tabs[1]:
    left, right = st.columns(2)
    with left:
        chart = alt.Chart(pd.DataFrame({"score": df["score"]})).mark_bar().encode(
            alt.X("score:Q", bin=alt.Bin(maxbins=30)),
            y="count()", tooltip=["count()"]
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

    with right:
        chart_lex = alt.Chart(pd.DataFrame({"lexical_score": df["lexical_score"]})).mark_bar().encode(
            alt.X("lexical_score:Q", bin=alt.Bin(maxbins=30)),
            y="count()", tooltip=["count()"]
        ).interactive()
        st.altair_chart(chart_lex, use_container_width=True)

    st.markdown("#### Scatter score vs lexical")
    scatter_df = df[["score","lexical_score"]]
    sc = alt.Chart(scatter_df).mark_point(opacity=0.6).encode(
        x="lexical_score:Q",
        y=alt.Y("score:Q", scale=alt.Scale(domain=[0,1])),
        tooltip=["score","lexical_score"]
    ).interactive()
    st.altair_chart(sc, use_container_width=True)

# --- Retrieval metrics ---
with tabs[2]:
    st.subheader("Retrieval evaluation")
    k = st.slider("k", 1, 20, 5)
    # Простейший синтетический ground truth: phrase_1 ~ phrase_2
    y_true, y_pred = [], []
    for _, row in df.iterrows():
        idx1, idx2 = phrase2idx[row["phrase_1"]], phrase2idx[row["phrase_2"]]
        sims = util.cos_sim(embeddings[idx1], embeddings)[0].cpu().numpy()
        sorted_idx = np.argsort(-sims)
        y_true.append([idx2])
        y_pred.append(sorted_idx.tolist())

    prec = np.mean([precision_at_k(t, p, k) for t, p in zip(y_true, y_pred)])
    mrr = np.mean([mean_reciprocal_rank(t, p) for t, p in zip(y_true, y_pred)])
    ndcg = np.mean([ndcg_at_k(t, p, k) for t, p in zip(y_true, y_pred)])
    st.json({"precision@k": prec, "MRR": mrr, "nDCG": ndcg})

# --- Semantic search ---
with tabs[3]:
    st.subheader("Semantic Search")
    query = st.text_input("Введите запрос")
    topn = st.slider("Top N", 1, 20, 5)
    if query:
        q_emb = encode_texts_in_batches(model, [preprocess_text(query)], batch_size)
        sims = util.cos_sim(q_emb, embeddings)[0].cpu().numpy()
        top_idx = np.argsort(-sims)[:topn]
        results = [(phrases_all[i], float(sims[i])) for i in top_idx]
        res_df = pd.DataFrame(results, columns=["phrase", "score"])
        st.dataframe(res_df, use_container_width=True)

# --- Clustering ---
with tabs[4]:
    st.subheader("Clustering")
    n_clusters = st.slider("Clusters (KMeans)", 2, 20, 5)
    emb_2d, labels = cluster_embeddings(embeddings, n_clusters)
    cluster_df = pd.DataFrame({"x": emb_2d[:,0], "y": emb_2d[:,1], "cluster": labels, "phrase": phrases_all})
    st.dataframe(cluster_df.head(20), use_container_width=True)

    chart = alt.Chart(cluster_df).mark_circle(size=60).encode(
        x="x:Q", y="y:Q", color="cluster:N", tooltip=["phrase"]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

# ================= Export =================
st.header("3. Экспорт")
csv_bytes = df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Скачать результаты CSV", data=csv_bytes, file_name="results.csv", mime="text/csv")
