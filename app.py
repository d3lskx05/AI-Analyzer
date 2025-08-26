# app.py
import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# local utils (import whole module to avoid import errors)
import utils

st.set_page_config(page_title="Synonym Checker — Product Edition", layout="wide")
st.title("🔎 Synonym Checker — Product Edition")

# -----------------------
# Sidebar: Models config
# -----------------------
st.sidebar.header("Model configuration")

model_source = st.sidebar.selectbox("Model source", ["huggingface", "google_drive"])
model_id = st.sidebar.text_input("Model A (ID or HF path)", value="sentence-transformers/all-MiniLM-L6-v2")
use_cache = st.sidebar.checkbox("Cache model load (recommended)", value=True)

@st.cache_resource
def _load_model_cached(src, mid):
    return utils.load_sentence_transformer(src, mid)

def load_model_from_cfg(src, mid, use_cache_flag=True):
    if use_cache_flag:
        return _load_model_cached(src, mid)
    return utils.load_sentence_transformer(src, mid)

with st.sidebar.expander("A/B testing (optional)"):
    enable_ab = st.sidebar.checkbox("Enable A/B test", value=False)
    model_b_source = st.sidebar.selectbox("Model B source", ["huggingface", "google_drive"]) if enable_ab else None
    model_b_id = st.sidebar.text_input("Model B ID", value="", key="modelb_id") if enable_ab else ""

# Load model A (eager)
model_a = None
try:
    with st.sidebar.spinner("Loading model A..."):
        model_a = load_model_from_cfg(model_source, model_id, use_cache)
    st.sidebar.success("Model A ready")
except Exception as e:
    st.sidebar.error(f"Failed to load Model A: {e}")

model_b = None
if enable_ab and model_b_id.strip():
    try:
        with st.sidebar.spinner("Loading model B..."):
            model_b = load_model_from_cfg(model_b_source, model_b_id, use_cache)
        st.sidebar.success("Model B ready")
    except Exception as e:
        st.sidebar.error(f"Failed to load Model B: {e}")

# -----------------------
# Session state init
# -----------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

if "suggestions" not in st.session_state:
    st.session_state["suggestions"] = []

# -----------------------
# Main UI: Tabs
# -----------------------
tabs = st.tabs(["Upload & Clean", "Pair Check", "Dataset Analysis", "A/B Compare", "Robustness", "Export / History"])

# -----------------------
# Tab: Upload & Clean
# -----------------------
with tabs[0]:
    st.header("1. Upload dataset & quick clean")
    uploaded = st.file_uploader("Upload CSV / XLSX / JSON (columns with text)", type=["csv", "xlsx", "json", "ndjson"])
    df = None
    file_hash = None
    if uploaded is not None:
        try:
            df, file_hash = utils.read_uploaded_file_bytes(uploaded)
            st.success(f"Loaded: {len(df)} rows, hash {file_hash}")
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            st.stop()

        st.subheader("Raw preview")
        st.dataframe(utils.safe_head(df, 10), use_container_width=True)

        st.subheader("Choose text column")
        text_col = st.selectbox("Text column for single-text workflows (or left for pair-based)", ["--none--"] + list(df.columns), index=0)
        pair_mode = st.checkbox("File already contains pairs (columns text1,text2,label?)", value=False)
        if pair_mode:
            st.info("Expecting columns: text1, text2 [, label]")
        else:
            st.info("We'll use one chosen text column for pair generation/visualizations")

        st.subheader("Edit dataset (quick)")
        edited = st.experimental_data_editor(df, num_rows="dynamic")
        if st.button("Apply edits"):
            df = edited.copy()
            st.success("Edits applied")
        st.session_state["uploaded_df"] = df
        st.session_state["uploaded_text_col"] = text_col
        st.session_state["file_hash"] = file_hash

# -----------------------
# Tab: Pair Check (manual & bulk)
# -----------------------
with tabs[1]:
    st.header("2. Pair check (manual and bulk)")
    st.subheader("Manual single pair")
    t1 = st.text_input("Phrase 1", key="manual_t1")
    t2 = st.text_input("Phrase 2", key="manual_t2")
    batch_size = st.sidebar.number_input("Batch size", min_value=8, max_value=2048, value=64, step=8)

    if st.button("Check pair", key="check_single"):
        if not model_a:
            st.error("Model A not loaded")
        elif not t1 or not t2:
            st.warning("Enter both phrases")
        else:
            pt1 = utils.preprocess_text(t1); pt2 = utils.preprocess_text(t2)
            emb1 = utils.encode_texts_in_batches(model_a, [pt1], batch_size=batch_size)
            emb2 = utils.encode_texts_in_batches(model_a, [pt2], batch_size=batch_size)
            score_a = utils.cos_sim_from_emb(emb1[0], emb2[0])
            lex = utils.jaccard_tokens(pt1, pt2)
            st.metric("Score A", f"{score_a:.4f}")
            st.metric("Jaccard (lex)", f"{lex:.4f}")
            if model_b:
                emb1b = utils.encode_texts_in_batches(model_b, [pt1], batch_size=batch_size)
                emb2b = utils.encode_texts_in_batches(model_b, [pt2], batch_size=batch_size)
                score_b = utils.cos_sim_from_emb(emb1b[0], emb2b[0])
                st.metric("Score B", f"{score_b:.4f}", delta=f"{(score_b - score_a):+.4f}")
            # edge-case checks
            fa = utils.simple_flags(pt1); fb = utils.simple_flags(pt2)
            if fa["has_neg"] != fb["has_neg"]:
                st.warning("NEGATION mismatch between phrases — model can fail here.")
            if abs(fa["len_tok"] - fb["len_tok"]) > 10:
                st.info("Length difference is large — check for truncation/punctuation issues.")
            # save to history
            rec = {"type": "manual_single", "t1": pt1, "t2": pt2, "score_a": score_a, "lex": lex, "timestamp": pd.Timestamp.now().isoformat()}
            if model_b:
                rec["score_b"] = score_b
            st.session_state["history"].append(rec)
            st.success("Saved result to history")

    st.subheader("Bulk / File-based pairs")
    if "uploaded_df" in st.session_state:
        df = st.session_state["uploaded_df"]
        # detect columns
        text1_col = None; text2_col = None; label_col = None
        if set(["text1", "text2"]).issubset(set(df.columns)):
            text1_col = "text1"; text2_col = "text2"
        else:
            # try common column names
            candidates = [c for c in df.columns if "text" in c.lower() or "phrase" in c.lower()]
            if len(candidates) >= 2:
                text1_col = candidates[0]; text2_col = candidates[1]
        # label col guess
        for name in ["label", "is_duplicate", "is_similar", "target"]:
            if name in df.columns:
                label_col = name; break

        st.write(f"Detected pair columns: text1='{text1_col}', text2='{text2_col}', label='{label_col}'")
        if text1_col and text2_col:
            if st.button("Run encoding & analysis on uploaded pairs"):
                texts_all = list(pd.unique(df[text1_col].fillna("").astype(str).tolist() + df[text2_col].fillna("").astype(str).tolist()))
                with st.spinner("Encoding texts..."):
                    embs_all = utils.encode_texts_in_batches(model_a, texts_all, batch_size=batch_size)
                idx = {t: i for i, t in enumerate(texts_all)}
                scores = []
                for _, r in df.iterrows():
                    a = str(r[text1_col] or "")
                    b = str(r[text2_col] or "")
                    s = utils.cos_sim_from_emb(embs_all[idx[a]], embs_all[idx[b]])
                    scores.append(s)
                df["score"] = scores
                df["lexical_score"] = df.apply(lambda r: utils.jaccard_tokens(str(r[text1_col] or ""), str(r[text2_col] or "")), axis=1)
                st.subheader("Preview results")
                st.dataframe(df[[text1_col, text2_col, "score", "lexical_score"]].head(50), use_container_width=True)
                # detector for suspicious
                semantic_threshold = st.sidebar.slider("Semantic threshold (suspicious >=)", 0.0, 1.0, 0.80, 0.01)
                lexical_threshold = st.sidebar.slider("Lexical threshold (suspicious <=)", 0.0, 1.0, 0.30, 0.01)
                susp = df[(df["score"] >= semantic_threshold) & (df["lexical_score"] <= lexical_threshold)]
                st.write(f"Suspicious pairs found: {len(susp)}")
                if not susp.empty:
                    st.dataframe(susp[[text1_col, text2_col, "score", "lexical_score"]].head(200))
                # save results + metadata to history
                rec = {"type": "file_pairs", "n_pairs": len(df), "model_a": model_id, "timestamp": pd.Timestamp.now().isoformat(),
                       "semantic_threshold": float(semantic_threshold), "lexical_threshold": float(lexical_threshold)}
                if label_col:
                    # compute metrics if labels present
                    y_true = df[label_col].astype(int).to_numpy()
                    y_scores = df["score"].to_numpy()
                    metrics = utils.compute_binary_metrics_and_plots(y_true, y_scores)
                    rec["metrics"] = {"f1": metrics["f1"], "precision": metrics["precision"], "recall": metrics["recall"],
                                      "roc_auc": metrics["roc_auc"], "pr_auc": metrics["pr_auc"]}
                st.session_state["history"].append(rec)
                st.success("Analysis complete and saved to history")
        else:
            st.info("Dataset does not look like pairwise table. Use Generate Tests or reformat your file.")

# -----------------------
# Tab: Dataset Analysis
# -----------------------
with tabs[2]:
    st.header("3. Dataset analysis & visualizations")
    if "uploaded_df" not in st.session_state:
        st.info("Upload dataset in 'Upload & Clean' tab first")
    else:
        df = st.session_state["uploaded_df"]
        st.subheader("Quick stats")
        st.write(df.describe(include="all").T)
        # choose a text column
        cols_text = [c for c in df.columns if df[c].dtype == object or df[c].dtype == "string"]
        if not cols_text:
            st.info("No text-like columns detected")
        else:
            col = st.selectbox("Visualization text column", cols_text)
            sample_n = st.slider("Number of samples to visualize (max)", 50, 2000, 500, 50)
            texts = df[col].dropna().astype(str).tolist()[:sample_n]
            if st.button("Plot 2D embeddings (sample)"):
                embs = utils.encode_texts_in_batches(model_a, texts, batch_size=batch_size)
                coords = utils.reduce_embeddings_for_plot(embs, method="tsne", n_components=2)
                fig = utils.plot_2d_scatter(coords, labels=None, title=f"2D ({len(texts)} samples)")
                st.pyplot(fig)
            if st.button("Plot 3D embeddings (sample)"):
                embs = utils.encode_texts_in_batches(model_a, texts, batch_size=batch_size)
                coords = utils.reduce_embeddings_for_plot(embs, method="tsne", n_components=3)
                fig = utils.plot_3d_scatter(coords, labels=None, title=f"3D ({len(texts)} samples)")
                st.pyplot(fig)

# -----------------------
# Tab: A/B Compare
# -----------------------
with tabs[3]:
    st.header("4. A/B model comparison")
    if not enable_ab or not model_b:
        st.info("Enable model B in sidebar to use this tab")
    else:
        st.subheader("Compare on generated or uploaded pairs")
        mode = st.radio("Source for comparison", ["Uploaded pairs", "Generate random test pairs"], index=0)
        if mode == "Uploaded pairs" and "uploaded_df" in st.session_state:
            dfu = st.session_state["uploaded_df"]
            # detect pair columns
            if set(["text1", "text2"]).issubset(set(dfu.columns)):
                df_pairs = dfu[["text1", "text2"]].dropna().head(200)
            else:
                # try first two text-like columns
                candidates = [c for c in dfu.columns if "text" in c.lower() or "phrase" in c.lower()]
                if len(candidates) >= 2:
                    df_pairs = dfu[[candidates[0], candidates[1]]].dropna().head(200)
                    df_pairs.columns = ["text1", "text2"]
                else:
                    st.info("No suitable pair columns found in uploaded file")
                    df_pairs = pd.DataFrame(columns=["text1", "text2"])
        else:
            # generate
            base_df = st.session_state.get("uploaded_df", None)
            if base_df is None:
                st.info("Upload dataset to generate test pairs, using small internal sampling")
                df_pairs = pd.DataFrame(columns=["text1", "text2"])
            else:
                ngen = st.number_input("Number of generated pairs", min_value=50, max_value=5000, value=500, step=50)
                df_pairs = utils.generate_test_pairs_from_df(base_df, text_col=st.session_state.get("uploaded_text_col") or base_df.columns[0],
                                                       n_samples=int(ngen))
        if not df_pairs.empty:
            st.write(f"Pairs for comparison: {len(df_pairs)}")
            # encode unique texts
            texts = list(pd.unique(df_pairs["text1"].astype(str).tolist() + df_pairs["text2"].astype(str).tolist()))
            with st.spinner("Encoding texts for both models..."):
                embs_a = utils.encode_texts_in_batches(model_a, texts, batch_size=batch_size)
                embs_b = utils.encode_texts_in_batches(model_b, texts, batch_size=batch_size)
            idx = {t: i for i, t in enumerate(texts)}
            scores_a = []; scores_b = []
            for _, r in df_pairs.iterrows():
                a = str(r["text1"]); b = str(r["text2"])
                sa = utils.cos_sim_from_emb(embs_a[idx[a]], embs_a[idx[b]])
                sb = utils.cos_sim_from_emb(embs_b[idx[a]], embs_b[idx[b]])
                scores_a.append(sa); scores_b.append(sb)
            df_pairs["score_a"] = scores_a
            df_pairs["score_b"] = scores_b
            st.dataframe(df_pairs.head(50), use_container_width=True)
            # heatmap of average differences in buckets
            labels = ["A", "B"]
            # small matrix: mean scores
            matrix = np.array([[np.mean(scores_a), np.mean(scores_b)], [np.mean(scores_b), np.mean(scores_a)]])
            fig_heat = utils.plot_model_comparison_heatmap(matrix, labels=["A", "B"], title="Mean score comparison")
            st.pyplot(fig_heat)
            # save history
            rec = {"type": "ab_compare", "n_pairs": len(df_pairs), "mean_a": float(np.mean(scores_a)), "mean_b": float(np.mean(scores_b)),
                   "timestamp": pd.Timestamp.now().isoformat(), "model_a": model_id, "model_b": model_b_id}
            st.session_state["history"].append(rec)
            st.success("A/B comparison saved to history")

# -----------------------
# Tab: Robustness
# -----------------------
with tabs[4]:
    st.header("5. Robustness & perturbation tests")
    st.write("Replace tokens with synonyms map and measure delta of cosine similarity.")
    default_map = {"привет": "здравствуй", "работа": "должность"}
    synonyms_input = st.text_area("Synonyms map (json)", value=json.dumps(default_map, ensure_ascii=False), height=120)
    try:
        synonyms_map = json.loads(synonyms_input)
        if not isinstance(synonyms_map, dict):
            synonyms_map = {}
    except Exception:
        synonyms_map = {}
    n_sample = st.number_input("Sample pairs to test", min_value=10, max_value=2000, value=200, step=10)
    if st.button("Run robustness on generated pairs"):
        base_df = st.session_state.get("uploaded_df", None)
        if base_df is None:
            st.warning("Upload dataset first")
        else:
            df_pairs = utils.generate_test_pairs_from_df(base_df, text_col=st.session_state.get("uploaded_text_col") or base_df.columns[0],
                                                   n_samples=n_sample, negative_ratio=0.5)
            with st.spinner("Running robustness test..."):
                res_df = utils.robustness_scores_for_pairs(model_a, df_pairs, synonyms_map=synonyms_map, batch_size=batch_size)
            st.dataframe(res_df.head(200), use_container_width=True)
            # show distribution of delta
            figd, axd = plt.subplots(figsize=(6, 3))
            axd.hist(res_df["delta"].fillna(0.0).values, bins=40)
            axd.set_title("Distribution of (perturbed - orig) cosine similarity")
            st.pyplot(figd)
            # save history
            rec = {"type": "robustness", "n_pairs": len(res_df), "synonyms": synonyms_map, "mean_delta": float(res_df["delta"].mean()), "timestamp": pd.Timestamp.now().isoformat()}
            st.session_state["history"].append(rec)
            st.success("Robustness test completed and saved")

# -----------------------
# Tab: Export / History
# -----------------------
with tabs[5]:
    st.header("6. Export / History")
    st.subheader("History of runs (in-memory this session)")
    if st.session_state["history"]:
        st.write(f"Entries: {len(st.session_state['history'])}")
        for i, rec in enumerate(reversed(st.session_state["history"])):
            st.markdown(f"**Run #{len(st.session_state['history']) - i}** — {rec.get('type','-')} — {rec.get('timestamp','-')}")
            st.json({k: v for k, v in rec.items() if k not in ("results",)})
            st.markdown("---")
        if st.button("Download history JSON"):
            out = json.dumps(st.session_state["history"], ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button("Download history.json", data=out, file_name="synchecker_history.json", mime="application/json")
    else:
        st.info("No history yet. Run some analyses.")

    st.subheader("Export current analysis to PDF")
    export_fname = st.text_input("PDF filename", value="synchecker_report.pdf")
    # Minimal example export: take last history item metrics if available
    if st.button("Export PDF report (last run)"):
        if not st.session_state["history"]:
            st.warning("No history to export")
        else:
            last = st.session_state["history"][-1]
            # create a minimal summary table and placeholder figs
            summary_df = pd.DataFrame([last])
            figs = []
            # if last run contained metric figures objects (not in our simple recs), we'd include them.
            # Here we create a placeholder figure
            figp, axp = plt.subplots(figsize=(6, 3))
            axp.text(0.1, 0.5, f"Run type: {last.get('type')}\nTime: {last.get('timestamp')}", fontsize=12)
            axp.axis("off")
            figs.append(figp)
            # export
            utils.export_report_pdf(export_fname, summary_df, figs, metadata={"model_a": model_id, "generated": str(pd.Timestamp.now())})
            st.success(f"Report saved to {export_fname}")
            # offer download (if file exists)
            if os.path.exists(export_fname):
                with open(export_fname, "rb") as f:
                    st.download_button("Download PDF", data=f, file_name=export_fname, mime="application/pdf")

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.caption("Synonym Checker — extended edition. Keep in mind this is an in-memory demo. For production: add persistent storage, async encoding queues, authentication, and rate limits.")
