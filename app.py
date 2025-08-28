# app.py — Synonym Checker (+) с расширенными аналитическими функциями
import os
import json
import time
import tempfile
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from sentence_transformers import util

# Локальные утилиты
from utils import (
    preprocess_text, file_md5, read_uploaded_file_bytes, parse_topics_field,
    jaccard_tokens, simple_style_suspicious_and_low as style_suspicious_and_low,
    simple_flags, pos_first_token, bootstrap_diff_ci,
    _load_model_from_source, encode_texts_in_batches,
    cosine_sim, dot_product, euclidean_dist, manhattan_dist, pair_score,
    build_neighbors, project_embeddings,
    load_sts_dataset, evaluate_sts,
    robustness_probe, find_suspicious,
    _save_plot_hist, _save_scatter, export_pdf_report,
    # Метрики ранжирования
    compute_mrr, compute_recall_at_k, compute_ndcg_at_k, evaluate_ranking_metrics
)

# ============== Конфиг страницы ==============
st.set_page_config(page_title="Synonym Checker", layout="wide")
st.title("🔎 Synonym Checker")

# ======== Кэшируем загрузку модели ========
@st.cache_resource(show_spinner=False)
def load_model_from_source(source: str, identifier: str):
    return _load_model_from_source(source, identifier)

# ============== Сайдбар: настройки модели ==============
st.sidebar.header("Настройки модели")

model_source = st.sidebar.selectbox("Источник модели", ["huggingface", "google_drive"], index=0)
DEFAULT_HF = "sentence-transformers/all-MiniLM-L6-v2"
if model_source == "huggingface":
    model_id = st.sidebar.text_input("Hugging Face Model ID", value=DEFAULT_HF)
else:
    model_id = st.sidebar.text_input("Google Drive File ID", value="1RR15OMLj9vfSrVa1HN-dRU-4LbkdbRRf")

enable_ab_test = st.sidebar.checkbox("Включить A/B тест двух моделей", value=False)
if enable_ab_test:
    ab_model_source = st.sidebar.selectbox("Источник второй модели", ["huggingface", "google_drive"], index=0, key="ab_source")
    if ab_model_source == "huggingface":
        ab_model_id = st.sidebar.text_input("Hugging Face Model ID (B)", value="sentence-transformers/all-MiniLM-L12-v2", key="ab_id")
    else:
        ab_model_id = st.sidebar.text_input("Google Drive File ID (B)", value="", key="ab_id")
else:
    ab_model_id = ""

batch_size = st.sidebar.number_input("Batch size для энкодинга", min_value=8, max_value=1024, value=64, step=8)

# Доп. настройки
st.sidebar.header("Детектор неочевидных совпадений")
enable_detector = st.sidebar.checkbox("Включить детектор (high sem, low lex)", value=True)
semantic_threshold = st.sidebar.slider("Порог семантической схожести (>=)", 0.0, 1.0, 0.80, 0.01)
lexical_threshold = st.sidebar.slider("Порог лексической похожести (<=)", 0.0, 1.0, 0.30, 0.01)
low_score_threshold = st.sidebar.slider("Порог низкой семантической схожести", 0.0, 1.0, 0.75, 0.01)

# Новые: выбор метрики сходства
st.sidebar.header("Метрики сходства")
metric_choice = st.sidebar.selectbox("Метрика", ["cosine", "dot", "euclidean", "manhattan"], index=0)

# ======== Загрузка моделей ========
try:
    with st.spinner("Загружаю основную модель..."):
        model_a = load_model_from_source(model_source, model_id)
    st.sidebar.success("Основная модель загружена")
except Exception as e:
    st.sidebar.error(f"Не удалось загрузить основную модель: {e}")
    st.stop()

model_b = None
if enable_ab_test and ab_model_id.strip():
    try:
        with st.spinner("Загружаю модель B..."):
            model_b = load_model_from_source(ab_model_source, ab_model_id)
        st.sidebar.success("Модель B загружена")
    except Exception as e:
        st.sidebar.error(f"Не удалось загрузить модель B: {e}")
        st.stop()

# ======== История - Отключил ========
if "suggestions" not in st.session_state:
    st.session_state["suggestions"] = []
if "experiments" not in st.session_state:
    st.session_state["experiments"] = []  # reproducibility: сохранённые запуски

def add_suggestions(phrases: List[str]):
    s = [p for p in phrases if p and isinstance(p, str)]
    for p in reversed(s):
        if p not in st.session_state["suggestions"]:
            st.session_state["suggestions"].insert(0, p)
    st.session_state["suggestions"] = st.session_state["suggestions"][:200]

# ======== Режим: выбор ввода ========
if "mode" not in st.session_state:
    st.session_state.mode = "Файл (CSV/XLSX/JSON)"
if "pending_mode" not in st.session_state:
    st.session_state.pending_mode = None
if "pending_confirm" not in st.session_state:
    st.session_state.pending_confirm = False
if "mode_ui_v" not in st.session_state:
    st.session_state.mode_ui_v = 0

radio_key = f"mode_selector_{st.session_state.mode}_{st.session_state.mode_ui_v}"
mode_choice = st.radio(
    "Режим проверки",
    ["Файл (CSV/XLSX/JSON)", "Ручной ввод", "Бенчмаркинг (STS)"],
    index=0 if st.session_state.mode == "Файл (CSV/XLSX/JSON)" else (1 if st.session_state.mode=="Ручной ввод" else 2),
    horizontal=True,
    key=radio_key
)
if st.session_state.pending_mode is None and mode_choice != st.session_state.mode:
    st.session_state.pending_mode = mode_choice
    st.session_state.pending_confirm = False

if st.session_state.pending_mode:
    col_warn, col_yes, col_close = st.columns([4, 1, 0.6])
    with col_warn:
        st.warning(
            f"Перейти в режим **{st.session_state.pending_mode}**? "
            "Текущие данные будут удалены."
        )
    with col_yes:
        if st.button("✅ Да"):
            if not st.session_state.pending_confirm:
                st.session_state.pending_confirm = True
                st.info("Подтвердить✅")
            else:
                st.session_state.mode = st.session_state.pending_mode
                st.session_state.pending_mode = None
                st.session_state.pending_confirm = False
                for k in ["uploaded_file", "manual_input"]:
                    st.session_state.pop(k, None)
                st.rerun()
    with col_close:
        if st.button("❌", help="Отмена"):
            st.session_state.pending_mode = None
            st.session_state.pending_confirm = False
            st.session_state.mode_ui_v += 1

mode = st.session_state.mode

# ======== Общие хелперы для вычислений ========
def compute_pair_scores(model, pairs: List[Tuple[str, str]], metric: str, batch_size: int):
    if not pairs:
        return []
    phrases = list({p for pair in pairs for p in pair})
    p2i = {p: i for i, p in enumerate(phrases)}
    embs = encode_texts_in_batches(model, phrases, batch_size=batch_size)
    out = []
    for a, b in pairs:
        s = pair_score(embs[p2i[a]], embs[p2i[b]], metric=metric)
        out.append(s)
    return out

# ======= Блок: ручной ввод =======
def _set_manual_value(key: str, val: str):
    st.session_state[key] = val

if mode == "Ручной ввод":
    st.header("Ручной ввод пар фраз")

    with st.expander("Проверить одну пару фраз (быстро)"):
        if "manual_text1" not in st.session_state:
            st.session_state["manual_text1"] = ""
        if "manual_text2" not in st.session_state:
            st.session_state["manual_text2"] = ""
        text1 = st.text_input("Фраза 1", key="manual_text1")
        text2 = st.text_input("Фраза 2", key="manual_text2")

        if st.button("Проверить пару", key="manual_check"):
            if not text1 or not text2:
                st.warning("Введите обе фразы.")
            else:
                t1 = preprocess_text(text1); t2 = preprocess_text(text2)
                add_suggestions([t1, t2])
                emb1 = encode_texts_in_batches(model_a, [t1], batch_size)
                emb2 = encode_texts_in_batches(model_a, [t2], batch_size)
                score_a = float(pair_score(emb1[0], emb2[0], metric=metric_choice))
                lex = jaccard_tokens(t1, t2)

                st.subheader("Результат (модель A)")
                col1, col2, col3 = st.columns([1,1,1])
                col1.metric("Score A", f"{score_a:.4f}")
                col2.metric("Jaccard (lexical)", f"{lex:.4f}")

                is_suspicious_single = False
                if enable_detector and (score_a >= semantic_threshold) and (lex <= lexical_threshold):
                    is_suspicious_single = True
                    st.warning("Обнаружено НЕОЧЕВИДНОЕ совпадение: высокая семантическая схожесть, низкая лексическая похожесть.")

                if model_b is not None:
                    emb1b = encode_texts_in_batches(model_b, [t1], batch_size)
                    emb2b = encode_texts_in_batches(model_b, [t2], batch_size)
                    score_b = float(pair_score(emb1b[0], emb2b[0], metric=metric_choice))
                    delta = score_b - score_a
                    col3.metric("Score B", f"{score_b:.4f}", delta=f"{delta:+.4f}")
                    comp_df = pd.DataFrame({"model": ["A","B"], "score":[score_a, score_b]})
                    chart = alt.Chart(comp_df).mark_bar().encode(
                        x=alt.X('model:N', title=None),
                        y=alt.Y('score:Q', title=f"{metric_choice} score"),
                        tooltip=['model','score']
                    )
                    st.altair_chart(chart.properties(width=300), use_container_width=False)
                else:
                    col3.write("")

    with st.expander("Ввести несколько пар (каждая пара на новой строке). Формат: `фраза1 || фраза2` / TAB / `,`"):
        bulk_text = st.text_area("Вставьте пары (по одной в строке)", height=180, key="bulk_pairs")
        st.caption("Если разделитель встречается в тексте — используйте `||`.")
        if st.button("Проверить все пары (ручной ввод)", key="manual_bulk_check"):
            lines = [l.strip() for l in bulk_text.splitlines() if l.strip()]
            if not lines:
                st.warning("Ничего не введено.")
            else:
                parsed = []
                for ln in lines:
                    if "||" in ln:
                        p1, p2 = ln.split("||", 1)
                    elif "\t" in ln:
                        p1, p2 = ln.split("\t", 1)
                    elif "," in ln:
                        p1, p2 = ln.split(",", 1)
                    else:
                        p1, p2 = ln, ""
                    p1 = preprocess_text(p1); p2 = preprocess_text(p2)
                    if p1 and p2:
                        parsed.append((p1, p2))
                if not parsed:
                    st.warning("Нет корректных пар.")
                else:
                    add_suggestions([p for pair in parsed for p in pair])
                    phrases_all = list({p for pair in parsed for p in pair})
                    phrase2idx = {p:i for i,p in enumerate(phrases_all)}
                    with st.spinner("Кодирую фразы моделью A..."):
                        embeddings_a = encode_texts_in_batches(model_a, phrases_all, batch_size)
                    embeddings_b = None
                    if model_b is not None:
                        with st.spinner("Кодирую фразы моделью B..."):
                            embeddings_b = encode_texts_in_batches(model_b, phrases_all, batch_size)
                    rows = []
                    for p1, p2 in parsed:
                        emb1 = embeddings_a[phrase2idx[p1]]
                        emb2 = embeddings_a[phrase2idx[p2]]
                        score_a = float(pair_score(emb1, emb2, metric=metric_choice))
                        score_b = None
                        if embeddings_b is not None:
                            emb1b = embeddings_b[phrase2idx[p1]]
                            emb2b = embeddings_b[phrase2idx[p2]]
                            score_b = float(pair_score(emb1b, emb2b, metric=metric_choice))
                        lex = jaccard_tokens(p1, p2)
                        rows.append({"phrase_1": p1, "phrase_2": p2, "score": score_a, "score_b": score_b, "lexical_score": lex})
                    res_df = pd.DataFrame(rows)
                    st.subheader("Результаты (ручной массовый ввод)")
                    styled = style_suspicious_and_low(res_df, semantic_threshold, lexical_threshold, low_score_threshold)
                    st.dataframe(styled, use_container_width=True)
                    csv_bytes = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Скачать результаты CSV", data=csv_bytes, file_name="manual_results.csv", mime="text/csv")

                    if enable_detector:
                        susp = find_suspicious(res_df, score_col="score", lexical_col="lexical_score",
                                               label_col=None,
                                               semantic_threshold=semantic_threshold,
                                               lexical_threshold=lexical_threshold,
                                               low_score_threshold=low_score_threshold)
                        susp_df = susp.get("high_sem_low_lex", pd.DataFrame())
                        if not susp_df.empty:
                            st.markdown("### Неочевидные совпадения (high semantic, low lexical)")
                            st.write(f"Найдено {len(susp_df)} пар.")
                            st.dataframe(susp_df, use_container_width=True)
                            susp_csv = susp_df.to_csv(index=False).encode('utf-8')
                            st.download_button("Скачать suspicious CSV", data=susp_csv, file_name="suspicious_manual_bulk.csv", mime="text/csv")

# ======= Блок: Бенчмаркинг (STS) =======
if mode == "Бенчмаркинг (STS)":
    st.header("Бенчмаркинг модели на STS")
    colb1, colb2 = st.columns(2)
    with colb1:
        ds_name = st.selectbox("Датасет", ["stsb_multi_mt", "stsb"], index=0)
    with colb2:
        lang = st.selectbox("Язык (для stsb_multi_mt)", ["en", "ru", "de", "fr"], index=0)

    if st.button("Загрузить датасет и посчитать метрики"):
        with st.spinner("Загружаю датасет..."):
            df_sts = load_sts_dataset(ds_name, lang if ds_name!="stsb" else "en")
        st.write(f"Пример данных: {len(df_sts)} записей")
        st.dataframe(df_sts.head(10), use_container_width=True)

        with st.spinner("Считаю корреляции..."):
            res = evaluate_sts(model_a, df_sts, metric=metric_choice, batch_size=batch_size)
        st.success("Готово")
        st.write(f"**Spearman**: {res.get('spearman')}")
        st.write(f"**Pearson**: {res.get('pearson')}")
        st.write(f"**n**: {res.get('n')}  |  **metric**: {res.get('metric')}")

        preds = np.array(res.get("preds", []), dtype=float)
        labels = np.array(res.get("labels", []), dtype=float)

        # Графики (Altair)
        chart_df = pd.DataFrame({"pred": preds, "label": labels})
        sc = alt.Chart(chart_df).mark_point(opacity=0.5).encode(
            x=alt.X("label:Q", title="Human score"),
            y=alt.Y("pred:Q", title="Model score"),
            tooltip=["label","pred"]
        ).interactive()
        st.altair_chart(sc, use_container_width=True)

        # Сохранение «эксперимента»
        if st.button("Сохранить эксперимент (reproducibility)"):
            exp = {
                "type": "sts_benchmark",
                "dataset": f"{ds_name}:{lang}",
                "model_a": model_id,
                "metric": metric_choice,
                "spearman": res.get("spearman"),
                "pearson": res.get("pearson"),
                "n": res.get("n"),
                "timestamp": pd.Timestamp.now().isoformat(),
            }
            st.session_state["experiments"].append(exp)
            st.success("Эксперимент сохранён (в памяти сессии). Скачайте JSON ниже при необходимости.")

        # Экспорт PDF отчёта
        with tempfile.TemporaryDirectory() as td:
            hist_path = os.path.join(td, "pred_hist.png")
            _save_plot_hist(preds, "Predicted scores histogram", hist_path)
            scatter_path = os.path.join(td, "pred_vs_label.png")
            _save_scatter(labels, preds, "Human score", "Model score", "Pred vs Label", scatter_path)
            out_pdf = os.path.join(td, "sts_report.pdf")
            rep = {
                "dataset": f"{ds_name}:{lang}",
                "model": model_id,
                "metric": metric_choice,
                "spearman": res.get("spearman"),
                "pearson": res.get("pearson"),
                "n": res.get("n"),
            }
            pdf_path = export_pdf_report(rep, {"Pred histogram": hist_path, "Pred vs Label": scatter_path}, out_pdf)
            if pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button("⬇️ Скачать PDF отчёт", data=f.read(), file_name="sts_report.pdf", mime="application/pdf")
            else:
                st.info("PDF отчёт недоступен (нет reportlab). Установите зависимость, чтобы включить экспорт.")

# ======= Блок: файл (основной сценарий, твой код сохранён и расширен) =======
if mode == "Файл (CSV/XLSX/JSON)":
    st.header("1. Загрузите CSV, Excel или JSON с колонками: phrase_1, phrase_2, topics (опционально), label (опционально)")
    uploaded_file = st.file_uploader("Выберите файл", type=["csv", "xlsx", "xls", "json", "ndjson"])

    if uploaded_file is not None:
        try:
            df, file_hash = read_uploaded_file_bytes(uploaded_file)
        except Exception as e:
            st.error(f"Ошибка чтения файла: {e}")
            st.stop()

        required_cols = {"phrase_1", "phrase_2"}
        if not required_cols.issubset(set(df.columns)):
            st.error(f"Файл должен содержать колонки: {required_cols}")
            st.stop()

        st.subheader("✏️ Редактирование датасета перед проверкой")
        st.caption("Можно изменять, добавлять и удалять строки. Изменения временные (только в этой сессии).")
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="dataset_editor")
        edited_csv = edited_df.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Скачать обновлённый датасет (CSV)", data=edited_csv, file_name="edited_dataset.csv", mime="text/csv")
        df = edited_df.copy()

        # Препроцессинг
        df["phrase_1"] = df["phrase_1"].map(preprocess_text)
        df["phrase_2"] = df["phrase_2"].map(preprocess_text)
        if "topics" in df.columns:
            df["topics_list"] = df["topics"].map(parse_topics_field)
        else:
            df["topics_list"] = [[] for _ in range(len(df))]

        for col in ["phrase_1", "phrase_2"]:
            flags = df[col].map(simple_flags)
            df[f"{col}_len_tok"] = flags.map(lambda d: d["len_tok"])
            df[f"{col}_len_char"] = flags.map(lambda d: d["len_char"])
            df[f"{col}_has_neg"] = flags.map(lambda d: d["has_neg"])
            df[f"{col}_has_num"] = flags.map(lambda d: d["has_num"])
            df[f"{col}_has_date"] = flags.map(lambda d: d["has_date"])
            df[f"{col}_pos1"] = df[col].map(pos_first_token) if pos_first_token else "NA"

        add_suggestions(list(set(df["phrase_1"].tolist() + df["phrase_2"].tolist())))

        # Энкодинг
        phrases_all = list(set(df["phrase_1"].tolist() + df["phrase_2"].tolist()))
        phrase2idx = {p: i for i, p in enumerate(phrases_all)}
        with st.spinner("Кодирую фразы моделью A..."):
            embeddings_a = encode_texts_in_batches(model_a, phrases_all, batch_size)
        embeddings_b = None
        if enable_ab_test and model_b is not None:
            with st.spinner("Кодирую фразы моделью B..."):
                embeddings_b = encode_texts_in_batches(model_b, phrases_all, batch_size)

        # Метрики по парам
        scores, scores_b, lexical_scores = [], [], []
        for _, row in df.iterrows():
            p1, p2 = row["phrase_1"], row["phrase_2"]
            emb1_a, emb2_a = embeddings_a[phrase2idx[p1]], embeddings_a[phrase2idx[p2]]
            score_a = float(pair_score(emb1_a, emb2_a, metric=metric_choice))
            scores.append(score_a)
            if embeddings_b is not None:
                emb1_b, emb2_b = embeddings_b[phrase2idx[p1]], embeddings_b[phrase2idx[p2]]
                scores_b.append(float(pair_score(emb1_b, emb2_b, metric=metric_choice)))
            lex_score = jaccard_tokens(p1, p2)
            lexical_scores.append(lex_score)

        df["score"] = scores
        if embeddings_b is not None:
            df["score_b"] = scores_b
        df["lexical_score"] = lexical_scores

        # ===== Новые вкладки аналитики сверху =====
        st.subheader("2. Аналитика")
        tabs = st.tabs(["Сводка", "Разведка (Explore)", "Срезы (Slices)", "A/B тест", "Визуализация (PCA/UMAP)", "Top-N соседи", "Ранжирование", "Robustness", "Экспорт", "Reproducibility"])

        # = Svodka =
        with tabs[0]:
            total = len(df)
            low_cnt = int((df["score"] < low_score_threshold).sum())
            susp_cnt = int(((df["score"] >= semantic_threshold) & (df["lexical_score"] <= lexical_threshold)).sum())
            colA, colB, colC, colD = st.columns(4)
            colA.metric("Размер датасета", f"{total}")
            colB.metric("Средний score", f"{df['score'].mean():.4f}")
            colC.metric("Медиана score", f"{df['score'].median():.4f}")
            colD.metric(f"Низкие (<{low_score_threshold:.2f})", f"{low_cnt} ({(low_cnt / max(total,1)):.0%})")
            st.caption(f"Неочевидные совпадения (high-sem/low-lex): {susp_cnt} ({(susp_cnt / max(total,1)):.0%})")
            st.caption(f"Метрика: **{metric_choice}**")

        # = Explore =
        with tabs[1]:
            st.markdown("#### Распределения и взаимосвязи")
            left, right = st.columns(2)
            with left:
                chart = alt.Chart(pd.DataFrame({"score": df["score"]})).mark_bar().encode(
                    alt.X("score:Q", bin=alt.Bin(maxbins=30), title=f"{metric_choice} score"),
                    y='count()', tooltip=['count()']
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
            with right:
                chart_lex = alt.Chart(pd.DataFrame({"lexical_score": df["lexical_score"]})).mark_bar().encode(
                    alt.X("lexical_score:Q", bin=alt.Bin(maxbins=30), title="Jaccard (лексика)"),
                    y='count()', tooltip=['count()']
                ).interactive()
                st.altair_chart(chart_lex, use_container_width=True)

            st.markdown("##### Точечный график: семантика vs лексика")
            scatter_df = df[["score","lexical_score"]].copy()
            sc = alt.Chart(scatter_df).mark_point(opacity=0.6).encode(
                x=alt.X("lexical_score:Q", title="Jaccard (лексика)"),
                y=alt.Y("score:Q", title=f"{metric_choice} (семантика)"),
                tooltip=["score","lexical_score"]
            ).interactive()
            st.altair_chart(sc, use_container_width=True)

            if enable_detector:
                st.markdown("##### Неочевидные совпадения")
                susp = find_suspicious(df, "score", "lexical_score", label_col=("label" if "label" in df.columns else None),
                                       semantic_threshold=semantic_threshold,
                                       lexical_threshold=lexical_threshold,
                                       low_score_threshold=low_score_threshold)
                susp_df = susp.get("high_sem_low_lex", pd.DataFrame())
                if susp_df.empty:
                    st.info("Не найдено пар под текущие пороги.")
                else:
                    st.write(f"Пар: {len(susp_df)}")
                    st.dataframe(susp_df[["phrase_1","phrase_2","score","lexical_score"]], use_container_width=True)

        # = Slices =
        with tabs[2]:
            st.markdown("#### Срезы качества")
            len_bins = st.selectbox("Биннинг по длине (сумма токенов)", ["[0,4]", "[5,9]", "[10,19]", "[20,+)"], index=1)
            def _len_bucket(r):
                n = int(r["phrase_1_len_tok"] + r["phrase_2_len_tok"])
                if n <= 4: return "[0,4]"
                if n <= 9: return "[5,9]"
                if n <= 19: return "[10,19]"
                return "[20,+)"
            df["_len_bucket"] = df.apply(_len_bucket, axis=1)

            topic_mode = st.checkbox("Агрегация по topics", value=("topics_list" in df.columns))
            df["_any_neg"] = df["phrase_1_has_neg"] | df["phrase_2_has_neg"]
            df["_any_num"] = df["phrase_1_has_num"] | df["phrase_2_has_num"]
            df["_any_date"] = df["phrase_1_has_date"] | df["phrase_2_has_date"]

            cols1 = st.columns(3)
            with cols1[0]:
                st.markdown("**По длине**")
                agg_len = df.groupby("_len_bucket")["score"].agg(["count","mean","median"]).reset_index().sort_values("_len_bucket")
                st.dataframe(agg_len, use_container_width=True)
            with cols1[1]:
                st.markdown("**Отрицания/Числа/Даты**")
                flags_view = []
                for flag in ["_any_neg","_any_num","_any_date"]:
                    sub = df[df[flag]]
                    flags_view.append({"флаг":flag, "count":len(sub), "mean":float(sub["score"].mean()) if len(sub)>0 else np.nan})
                st.dataframe(pd.DataFrame(flags_view), use_container_width=True)
            with cols1[2]:
                st.markdown("**POS первого токена**")
                if "phrase_1_pos1" in df.columns:
                    pos_agg = df.groupby("phrase_1_pos1")["score"].agg(["count","mean"]).reset_index().rename(columns={"phrase_1_pos1":"POS"})
                    st.dataframe(pos_agg.sort_values("count", ascending=False), use_container_width=True)
                else:
                    st.info("Морфология (POS) недоступна: не установлен pymorphy2")

            if topic_mode:
                st.markdown("**По темам (topics)**")
                exploded = df.explode("topics_list")
                exploded["topics_list"] = exploded["topics_list"].fillna("")
                exploded = exploded[exploded["topics_list"].astype(str)!=""]
                if exploded.empty:
                    st.info("В датасете нет непустых topics.")
                else:
                    top_agg = exploded.groupby("topics_list")["score"].agg(["count","mean","median"]).reset_index().sort_values("count", ascending=False)
                    st.dataframe(top_agg, use_container_width=True)

        # = AB test =
        with tabs[3]:
            if (not enable_ab_test) or ("score_b" not in df.columns):
                st.info("A/B тест отключён или нет столбца score_b.")
            else:
                st.markdown("#### Сравнение моделей A vs B")
                colx, coly, colz = st.columns(3)
                colx.metric("Средний A", f"{df['score'].mean():.4f}")
                coly.metric("Средний B", f"{df['score_b'].mean():.4f}")
                colz.metric("Δ (B - A)", f"{(df['score_b'].mean()-df['score'].mean()):+.4f}")

                n_boot = st.slider("Бутстрэп итераций", 200, 2000, 500, 100)
                mean_diff, low, high = bootstrap_diff_ci(df["score_b"].to_numpy(), df["score"].to_numpy(), n_boot=n_boot)
                st.write(f"ДИ (95%) для Δ (B−A): **[{low:+.4f}, {high:+.4f}]**, средняя разница: **{mean_diff:+.4f}**")
                ab_df = pd.DataFrame({"A": df["score"], "B": df["score_b"]})
                ab_chart = alt.Chart(ab_df.reset_index()).mark_point(opacity=0.5).encode(
                    x=alt.X("A:Q"),
                    y=alt.Y("B:Q"),
                    tooltip=["A","B"]
                ).interactive()
                st.altair_chart(ab_chart, use_container_width=True)

                delta_df = df.copy()
                delta_df["delta"] = delta_df["score_b"] - delta_df["score"]
                st.markdown("**Топ, где B ≫ A**")
                st.dataframe(
                    delta_df.sort_values("delta", ascending=False).head(10)[["phrase_1","phrase_2","score","score_b","delta"]],
                    use_container_width=True
                )
                st.markdown("**Топ, где A ≫ B**")
                st.dataframe(
                    delta_df.sort_values("delta", ascending=True).head(10)[["phrase_1","phrase_2","score","score_b","delta"]],
                    use_container_width=True
                )

        # = Визуализация эмбеддингов =
        with tabs[4]:
            st.markdown("#### Визуализация эмбеддингов (PCA / UMAP)")
            method = st.selectbox("Метод проекции", ["PCA", "UMAP"], index=0)
            dim = st.selectbox("Размерность проекции", ["2D", "3D"], index=0)
            target = st.selectbox("Какие фразы визуализировать", ["phrase_1", "phrase_2"], index=0)
            embs = embeddings_a if target == "phrase_1" else embeddings_a  # у нас один набор на все фразы
            proj = project_embeddings(embs, method=method.lower(), n_components=(2 if dim=="2D" else 3))
            if proj is None:
                st.info("Проекция недоступна (нет зависимостей sklearn/umap).")
            else:
                proj_df = pd.DataFrame(proj, columns=["x","y"] if dim=="2D" else ["x","y","z"])
                proj_df["text"] = phrases_all
                if dim == "2D":
                    c = alt.Chart(proj_df).mark_point(opacity=0.6).encode(
                        x="x:Q", y="y:Q", tooltip=["text"]
                    ).interactive()
                    st.altair_chart(c, use_container_width=True)
                else:
                    st.write("3D scatter недоступен в Altair. Показываю 2D как fallback.")
                    c = alt.Chart(proj_df).mark_point(opacity=0.6).encode(
                        x="x:Q", y="y:Q", tooltip=["text"]
                    ).interactive()
                    st.altair_chart(c, use_container_width=True)

        # = Top-N соседи =
        with tabs[5]:
            st.markdown("#### Top-N ближайших соседей (по выбранной метрике)")
            topn = st.slider("N соседей", 2, 20, 5)
            nn, dists, idxs = build_neighbors(embeddings_a, metric=metric_choice, n_neighbors=topn)
            if nn is None or dists is None:
                st.info("Соседи недоступны (нет sklearn).")
            else:
                # Таблица: для каждой фразы — соседи
                rows = []
                for i, p in enumerate(phrases_all):
                    for rank, j in enumerate(idxs[i]):
                        if int(j) == i:
                            continue
                        rows.append({
                            "query": p,
                            "neighbor": phrases_all[int(j)],
                            "rank": int(rank),
                            "distance": float(dists[i][rank])
                        })
                nb_df = pd.DataFrame(rows)
                st.dataframe(nb_df.head(300), use_container_width=True)
                nb_csv = nb_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Скачать соседей (CSV)", data=nb_csv, file_name="neighbors.csv", mime="text/csv")

                # ===== Метрики ранжирования =====
                if "label" in df.columns:
                    st.markdown("### Метрики ранжирования (MRR, Recall@k, nDCG@k)")
                    if st.button("Вычислить метрики по топ-N"):
                        # Строим словарь neighbors: {индекс -> список индексов соседей}
                        phrase_to_idx = {p: idx for idx, p in enumerate(phrases_all)}
                        neighbors_dict = {}
                        for i, p in enumerate(phrases_all):
                            ranked = [int(j) for j in idxs[i] if int(j) != i]
                            neighbors_dict[i] = ranked

                        # Строим словарь релевантных: {индекс -> set индексов релевантов}
                        relevance_dict = {}
                        for _, row in df.iterrows():
                            if "label" in row and row["label"] == 1:
                                qid = phrase_to_idx.get(row["phrase_1"])
                                rel = phrase_to_idx.get(row["phrase_2"])
                                if qid is not None and rel is not None:
                                    relevance_dict.setdefault(qid, set()).add(rel)

                        k_values = [1, 3, 5, 10]
                        metrics_df = evaluate_ranking_metrics(neighbors_dict, relevance_dict, k_values)

                        st.subheader("Сводка по метрикам")
                        avg_row = metrics_df.drop(columns=["query_id"]).mean().to_dict()
                        st.json(avg_row)

                        st.subheader("Детали по каждому запросу")
                        st.dataframe(metrics_df, use_container_width=True)

                        csv_metrics = metrics_df.to_csv(index=False).encode("utf-8")
                        st.download_button("⬇️ Скачать метрики (CSV)", data=csv_metrics, file_name="ranking_metrics.csv", mime="text/csv")
                else:
                    st.info("Для вычисления метрик ранжирования нужен столбец 'label' с релевантностью (0/1).")
        # = Ранжирование =
        with tabs[6]:
            st.markdown("#### Ранжирование")
        
            ds_mode = st.radio("Источник данных", ["Custom dataset", "Stub (MS MARCO)"], horizontal=True)

            if ds_mode == "Custom dataset":
                rank_file = st.file_uploader(
                    "Загрузите файл с колонками: query, candidate, label ИЛИ phrase_1, phrase_2, label",
                    type=["csv", "xlsx", "json"]
                )
                if rank_file is not None:
                    df_rank, _ = read_uploaded_file_bytes(rank_file)
                else:
                    df_rank = None

            else:
                # встроенный пример (демо)
                df_rank = pd.DataFrame([
                    {"query": "what is AI", "candidate": "artificial intelligence", "label": 1},
                    {"query": "what is AI", "candidate": "machine learning", "label": 1},
                    {"query": "what is AI", "candidate": "banana", "label": 0},
                    {"query": "who is the president of USA", "candidate": "Joe Biden", "label": 1},
                    {"query": "who is the president of USA", "candidate": "Barack Obama", "label": 0},
                ])
            if df_rank is not None:
                st.write("Размер датасета:", len(df_rank))
                st.dataframe(df_rank.head(10), use_container_width=True)

                # --- Приводим к единому формату ---
                if "query" in df_rank.columns and "candidate" in df_rank.columns:
                    queries = df_rank["query"].map(preprocess_text).tolist()
                    candidates = df_rank["candidate"].map(preprocess_text).tolist()
                elif "phrase_1" in df_rank.columns and "phrase_2" in df_rank.columns:
                    queries = df_rank["phrase_1"].map(preprocess_text).tolist()
                    candidates = df_rank["phrase_2"].map(preprocess_text).tolist()
                else:
                    st.error("В датасете должны быть колонки (query, candidate, label) или (phrase_1, phrase_2, label).")
                    st.stop()

                if "label" not in df_rank.columns:
                    st.error("В датасете отсутствует колонка 'label'.")
                    st.stop()

                labels = df_rank["label"].astype(int).tolist()

                # --- Энкодинг Model A ---
                with st.spinner("Энкодинг Model A..."):
                    q_emb_a = encode_texts_in_batches(model_a, queries, batch_size)
                    c_emb_a = encode_texts_in_batches(model_a, candidates, batch_size)

                q_emb_b, c_emb_b = None, None
                if model_b is not None:
                    with st.spinner("Энкодинг Model B..."):
                        q_emb_b = encode_texts_in_batches(model_b, queries, batch_size)
                        c_emb_b = encode_texts_in_batches(model_b, candidates, batch_size)

                # --- Вычисление метрик ---
                k_values = [1, 3, 5, 10]

                def build_eval(emb_q, emb_c):
                    neighbors = {}
                    relevance = {}
                    for qid, qe in enumerate(emb_q):
                        sims = [pair_score(qe, ce, metric=metric_choice) for ce in emb_c]
                        ranked = np.argsort(sims)[::-1]  # сортируем от большего к меньшему
                        neighbors[qid] = ranked.tolist()
                        # релевантные кандидаты для этого запроса
                        rels = {i for i, l in enumerate(labels) if l == 1 and queries[i] == queries[qid]}
                        relevance[qid] = rels
                    return evaluate_ranking_metrics(neighbors, relevance, k_values)

                st.markdown("#### Метрики Model A")
                metrics_a = build_eval(q_emb_a, c_emb_a)
                st.dataframe(metrics_a, use_container_width=True)
                avg_a = metrics_a.drop(columns=["query_id"]).mean().rename("Model A")

                metrics_b, avg_b = None, None
                if q_emb_b is not None and c_emb_b is not None:
                    st.markdown("#### Метрики Model B")
                    metrics_b = build_eval(q_emb_b, c_emb_b)
                    st.dataframe(metrics_b, use_container_width=True)
                    avg_b = metrics_b.drop(columns=["query_id"]).mean().rename("Model B")

                # --- Сравнение ---
                st.markdown("### Сравнение моделей")
                if avg_b is not None:
                    comp_df = pd.concat([avg_a, avg_b], axis=1)
                    st.dataframe(comp_df)

                    chart_df = comp_df.reset_index().melt(id_vars="index", var_name="model", value_name="score")
                    chart = alt.Chart(chart_df).mark_bar().encode(
                        x="index:N", y="score:Q", color="model:N"
                    )
                    st.altair_chart(chart, use_container_width=True)

                    # Экспорт усреднённых метрик
                    avg_csv = comp_df.to_csv().encode("utf-8")
                    st.download_button("⬇️ Скачать усреднённые метрики (CSV)", data=avg_csv, file_name="ranking_summary.csv", mime="text/csv")
                else:
                    st.dataframe(avg_a)
                    avg_csv = avg_a.to_frame().to_csv().encode("utf-8")
                    st.download_button("⬇️ Скачать усреднённые метрики (CSV)", data=avg_csv, file_name="ranking_summary.csv", mime="text/csv")

                    # Экспорт детальных метрик
                    metrics_a_csv = metrics_a.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Скачать детальные метрики Model A (CSV)", data=metrics_a_csv, file_name="ranking_metrics_A.csv", mime="text/csv")
                    if metrics_b is not None:
                        metrics_b_csv = metrics_b.to_csv(index=False).encode("utf-8")
                        st.download_button("⬇️ Скачать детальные метрики Model B (CSV)", data=metrics_b_csv, file_name="ranking_metrics_B.csv", mime="text/csv")
                
        # = Robustness =
        with tabs[7]:
            st.markdown("#### Robustness / устойчивость")
            sample_n = st.slider("Сколько пар проверять", 1, min(20, len(df)), min(5, len(df)))
            pairs = list(zip(df["phrase_1"].tolist()[:sample_n], df["phrase_2"].tolist()[:sample_n]))
            if st.button("Запустить robustness-проверку"):
                with st.spinner("Генерирую варианты и считаю дельты..."):
                    rob_df = robustness_probe(model_a, pairs, metric=metric_choice, batch_size=batch_size)
                st.dataframe(rob_df, use_container_width=True)
                worst = rob_df.sort_values("delta").head(10)
                st.markdown("**Где модель падает (самые негативные дельты):**")
                st.dataframe(worst, use_container_width=True)
                csv_bytes = rob_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Скачать robustness CSV", data=csv_bytes, file_name="robustness.csv", mime="text/csv")

        # = Export =
        with tabs[8]:
            st.markdown("#### Экспорт отчёта (JSON/PDF)")
            report = {
                "file_name": uploaded_file.name,
                "file_hash": file_hash,
                "n_pairs": int(len(df)),
                "model_a": model_id,
                "model_b": ab_model_id if enable_ab_test else None,
                "metric": metric_choice,
                "thresholds": {
                    "semantic_threshold": float(semantic_threshold),
                    "lexical_threshold": float(lexical_threshold),
                    "low_score_threshold": float(low_score_threshold)
                },
                "summary": {
                    "mean_score": float(df["score"].mean()),
                    "median_score": float(df["score"].median()),
                    "low_count": int((df["score"] < low_score_threshold).sum()),
                    "suspicious_count": int(((df["score"] >= semantic_threshold) & (df["lexical_score"] <= lexical_threshold)).sum())
                }
            }
            rep_bytes = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button("💾 Скачать отчёт JSON", data=rep_bytes, file_name="synonym_checker_report.json", mime="application/json")

            # PDF с графиками
            with tempfile.TemporaryDirectory() as td:
                hist_path = os.path.join(td, "score_hist.png")
                _save_plot_hist(df["score"].to_numpy(), f"{metric_choice} score histogram", hist_path)
                sc_path = os.path.join(td, "sem_vs_lex.png")
                _save_scatter(df["lexical_score"].to_numpy(), df["score"].to_numpy(),
                              "Jaccard (lexical)", f"{metric_choice} (semantic)",
                              "Semantic vs Lex", sc_path)
                out_pdf = os.path.join(td, "report.pdf")
                pdf_path = export_pdf_report(report["summary"] | {"file_name": report["file_name"], "model": model_id, "metric": metric_choice},
                                             {"Score histogram": hist_path, "Sem vs Lex": sc_path}, out_pdf)
                if pdf_path and os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        st.download_button("⬇️ Скачать PDF отчёт", data=f.read(), file_name="report.pdf", mime="application/pdf")
                else:
                    st.info("PDF отчёт недоступен (нет reportlab). Установите зависимость, чтобы включить экспорт.")

        # = Reproducibility =
        with tabs[9]:
            st.markdown("#### Сохранение/сравнение экспериментов")
            if st.button("Сохранить текущий запуск как эксперимент"):
                exp = {
                    "type": "file_run",
                    "file": uploaded_file.name,
                    "file_hash": file_hash,
                    "model_a": model_id,
                    "model_b": ab_model_id if enable_ab_test else None,
                    "metric": metric_choice,
                    "mean_score": float(df["score"].mean()),
                    "timestamp": pd.Timestamp.now().isoformat()
                }
                st.session_state["experiments"].append(exp)
                st.success("Эксперимент сохранён в сессии.")

            if st.session_state["experiments"]:
                st.markdown("**Сохранённые эксперименты**")
                st.dataframe(pd.DataFrame(st.session_state["experiments"]), use_container_width=True)
                exp_bytes = json.dumps(st.session_state["experiments"], ensure_ascii=False, indent=2).encode("utf-8")
                st.download_button("⬇️ Скачать эксперименты (JSON)", data=exp_bytes, file_name="experiments.json", mime="application/json")
            else:
                st.info("Экспериментов пока нет.")

        # Итоги и таблицы (твоя логика сохранена)
    with st.expander("📊 3. Результаты и выгрузка", expanded=False):
        result_csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Скачать результаты CSV", data=result_csv, file_name="results.csv", mime="text/csv")
        styled_df = style_suspicious_and_low(df, semantic_threshold, lexical_threshold, low_score_threshold)
        st.dataframe(styled_df, use_container_width=True)

        # Suspicious блок (расширено с учётом label)
        if enable_detector:
            susp = find_suspicious(df,
                                   score_col="score",
                                   lexical_col="lexical_score",
                                   label_col=("label" if "label" in df.columns else None),
                                   semantic_threshold=semantic_threshold,
                                   lexical_threshold=lexical_threshold,
                                   low_score_threshold=low_score_threshold)
            st.markdown("### Подозрительные / аномальные случаи")
            for k, sdf in susp.items():
                st.markdown(f"**{k}** — {len(sdf)}")
                if not sdf.empty:
                    st.dataframe(sdf, use_container_width=True)
                    s_csv = sdf.to_csv(index=False).encode("utf-8")
                    st.download_button(f"⬇️ Скачать {k}.csv", data=s_csv, file_name=f"{k}.csv", mime="text/csv")
