import streamlit as st
import pandas as pd
from utils import *

st.set_page_config(page_title="Synonym Checker v3", layout="wide")
st.title("🔎 Synonym Checker & Model Analyzer v3")

# --- Sidebar: загрузка моделей ---
st.sidebar.header("Модели")
model_source = st.sidebar.selectbox("Источник модели", ["huggingface","google_drive"])
model_id_a = st.sidebar.text_input("Model A ID", value="sentence-transformers/all-MiniLM-L6-v2")
model_a = load_model(model_source, model_id_a)

enable_ab = st.sidebar.checkbox("A/B тест моделей", value=False)
model_b = None
if enable_ab:
    model_id_b = st.sidebar.text_input("Model B ID", value="sentence-transformers/all-MiniLM-L6-v2")
    model_b = load_model(model_source, model_id_b)

# --- Загрузка датасета ---
st.header("📥 Загрузка датасета")
uploaded_file = st.file_uploader("Выберите CSV/XLSX/JSON файл", type=["csv","xlsx","json"])
df = None
if uploaded_file:
    df = read_uploaded_file_bytes(uploaded_file)
    st.success(f"Датасет загружен, {len(df)} строк")
    st.dataframe(df.head())

# --- Проверка пары фраз ---
st.header("🔎 Проверка пары фраз")
text1 = st.text_input("Фраза 1")
text2 = st.text_input("Фраза 2")
if st.button("Проверить пару"):
    if not text1 or not text2: st.warning("Введите обе фразы")
    else:
        t1, t2 = preprocess_text(text1), preprocess_text(text2)
        emb1 = encode_texts_in_batches(model_a,[t1])
        emb2 = encode_texts_in_batches(model_a,[t2])
        score_a = cosine_similarity(emb1[0], emb2[0])
        st.metric("Score A", f"{score_a:.4f}")
        if model_b:
            emb1b = encode_texts_in_batches(model_b,[t1])
            emb2b = encode_texts_in_batches(model_b,[t2])
            score_b = cosine_similarity(emb1b[0], emb2b[0])
            st.metric("Score B", f"{score_b:.4f}", delta=f"{score_b-score_a:+.4f}")

# --- Генерация тестов ---
st.header("⚡ Генерация тестов")
if st.button("Сгенерировать случайные пары"):
    if df is not None:
        df_pairs = generate_synonym_pairs(df, text_col=df.columns[0], n=200)
        st.dataframe(df_pairs.head())

# --- Визуализация embeddings ---
st.header("📊 Визуализация Embeddings")
if st.button("Построить 2D scatter"):
    if df is not None:
        texts = df[df.columns[0]].dropna().tolist()[:200]
        plot_embeddings_2d(model_a, texts)
