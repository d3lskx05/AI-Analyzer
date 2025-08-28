# utils.py
# Вспомогательные функции и утилиты для Synonym Checker
from __future__ import annotations
import io
import os
import re
import json
import tarfile
import zipfile
import shutil
import hashlib
import tempfile
from typing import List, Tuple, Dict, Any, Optional, Set

import numpy as np
import pandas as pd

# ====== ML / NLP ======
from sentence_transformers import SentenceTransformer, util

# Метрики/статистика
try:
    from scipy.stats import spearmanr, pearsonr
except Exception:
    spearmanr = None
    pearsonr = None

# Визуализация (для PDF)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# sklearn (опционально)
try:
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
except Exception:
    PCA = None
    NearestNeighbors = None

# UMAP (опционально)
try:
    import umap
except Exception:
    umap = None

# PDF (опционально)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas as pdf_canvas
except Exception:
    A4 = None
    ImageReader = None
    pdf_canvas = None

# datasets (опционально)
try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

# pymorphy2 (опционально)
try:
    import pymorphy2   # type: ignore
    _MORPH = pymorphy2.MorphAnalyzer()
except Exception:
    _MORPH = None

# nltk.wordnet (опционально)
try:
    import nltk
    from nltk.corpus import wordnet as wn
    _NLTK_OK = True
except Exception:
    _NLTK_OK = False
    wn = None

# ============== Утилиты ==============

def preprocess_text(t: Any) -> str:
    if pd.isna(t):
        return ""
    return " ".join(str(t).lower().strip().split())

def file_md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def _try_read_json(raw: bytes) -> pd.DataFrame:
    try:
        obj = json.loads(raw.decode("utf-8"))
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            return pd.DataFrame(obj)
    except Exception:
        pass
    try:
        return pd.read_json(io.BytesIO(raw), lines=True)
    except Exception:
        pass
    raise ValueError("Не удалось распознать JSON/NDJSON")

def read_uploaded_file_bytes(uploaded) -> Tuple[pd.DataFrame, str]:
    raw = uploaded.read()
    h = file_md5(raw)
    name = (uploaded.name or "").lower()
    if name.endswith(".json") or name.endswith(".ndjson"):
        df = _try_read_json(raw)
        return df, h
    try:
        df = pd.read_csv(io.BytesIO(raw))
        return df, h
    except Exception:
        pass
    try:
        df = pd.read_excel(io.BytesIO(raw))
        return df, h
    except Exception as e:
        raise ValueError("Файл должен быть CSV, Excel или JSON. Ошибка: " + str(e))

def parse_topics_field(val) -> List[str]:
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    s = str(val).strip()
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    for sep in [";", "|", ","]:
        if sep in s:
            return [p.strip() for p in s.split(sep) if p.strip()]
    return [s] if s else []

def jaccard_tokens(a: str, b: str) -> float:
    sa = set([t for t in a.split() if t])
    sb = set([t for t in b.split() if t])
    if not sa and not sb:
        return 0.0
    inter = sa & sb
    union = sa | sb
    return len(inter) / len(union) if union else 0.0

def simple_style_suspicious_and_low(df, sem_thresh: float, lex_thresh: float, low_score_thresh: float):
    def highlight(row):
        out = []
        try:
            score = float(row.get('score', 0))
        except Exception:
            score = 0.0
        try:
            lex = float(row.get('lexical_score', 0))
        except Exception:
            lex = 0.0
        is_low_score = (score < low_score_thresh)
        is_suspicious = (score >= sem_thresh and lex <= lex_thresh)
        for _ in row:
            if is_suspicious:
                out.append('background-color: #fff2b8')
            elif is_low_score:
                out.append('background-color: #ffcccc')
            else:
                out.append('')
        return out
    return df.style.apply(highlight, axis=1)

NEG_PAT = re.compile(r"\bне\b|\bни\b|\bнет\b", flags=re.IGNORECASE)
NUM_PAT = re.compile(r"\b\d+\b")
DATE_PAT = re.compile(r"\b\d{1,2}[./-]\d{1,2}([./-]\d{2,4})?\b")

def simple_flags(text: str) -> Dict[str, Any]:
    t = text or ""
    return {
        "has_neg": bool(NEG_PAT.search(t)),
        "has_num": bool(NUM_PAT.search(t)),
        "has_date": bool(DATE_PAT.search(t)),
        "len_char": len(t),
        "len_tok": len([x for x in t.split() if x]),
    }

def pos_first_token(text: str) -> str:
    if _MORPH is None:
        return "NA"
    toks = [t for t in text.split() if t]
    if not toks:
        return "NA"
    p = _MORPH.parse(toks[0])[0]
    return str(p.tag.POS) if p and p.tag and p.tag.POS else "NA"

def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 500, seed: int = 42, ci: float = 0.95):
    rng = np.random.default_rng(seed)
    diffs = []
    n = min(len(a), len(b))
    if n == 0:
        return 0.0, 0.0, 0.0
    a = np.asarray(a); b = np.asarray(b)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(np.mean(a[idx] - b[idx]))
    diffs = np.array(diffs)
    mean_diff = float(np.mean(diffs))
    low = float(np.quantile(diffs, (1-ci)/2))
    high = float(np.quantile(diffs, 1-(1-ci)/2))
    return mean_diff, low, high

# ============== Model IO ==============

def download_file_from_gdrive(file_id: str) -> str:
    import gdown
    tmp_dir = tempfile.gettempdir()
    archive_path = os.path.join(tmp_dir, f"model_gdrive_{file_id}")
    model_dir = os.path.join(tmp_dir, f"model_gdrive_extracted_{file_id}")
    if not os.path.exists(archive_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, archive_path, quiet=True)
    if os.path.exists(model_dir) and os.path.isdir(model_dir):
        return model_dir
    os.makedirs(model_dir, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(model_dir)
    else:
        try:
            shutil.copy(archive_path, model_dir)
        except Exception:
            pass
    return model_dir

def _load_model_from_source(source: str, identifier: str) -> SentenceTransformer:
    if source == "huggingface":
        model_path = identifier
    elif source == "google_drive":
        model_path = download_file_from_gdrive(identifier)
    else:
        raise ValueError("Unknown model source")
    return SentenceTransformer(model_path)

def encode_texts_in_batches(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    if not texts:
        return np.array([])
    embs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(embs)

# ============== Similarity helpers ==============

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(util.cos_sim(a, b).item())

def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def euclidean_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def manhattan_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))

def pair_score(a: np.ndarray, b: np.ndarray, metric: str = "cosine") -> float:
    m = metric.lower()
    if m == "cosine":
        return cosine_sim(a, b)
    if m == "dot":
        return dot_product(a, b)
    if m == "euclidean":
        d = euclidean_dist(a, b)
        return -d
    if m == "manhattan":
        d = manhattan_dist(a, b)
        return -d
    return cosine_sim(a, b)

def build_neighbors(embeddings: np.ndarray, metric: str = "cosine", n_neighbors: int = 5):
    """
    Возвращает (nn_index, distances, indices) похожих соседей; или (None, None, None) если sklearn недоступен.
    """
    if NearestNeighbors is None:
        return None, None, None
    m = metric.lower()
    if m == "cosine":
        nn_metric = "cosine"
    elif m == "dot":
        nn_metric = "cosine"
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    elif m == "euclidean":
        nn_metric = "euclidean"
    elif m == "manhattan":
        nn_metric = "manhattan"
    else:
        nn_metric = "cosine"
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(embeddings)), metric=nn_metric)
    nn.fit(embeddings)
    dists, idxs = nn.kneighbors(embeddings, return_distance=True)
    return nn, dists, idxs

# ============== Ranking metrics ==============

def compute_mrr(ranked: List[int], relevant: Set[int]) -> float:
    """Mean Reciprocal Rank для одного запроса"""
    for i, idx in enumerate(ranked, start=1):
        if idx in relevant:
            return 1.0 / i
    return 0.0

def compute_recall_at_k(ranked: List[int], relevant: Set[int], k: int) -> float:
    """Recall@k для одного запроса"""
    if not relevant:
        return 0.0
    return len([i for i in ranked[:k] if i in relevant]) / len(relevant)

def compute_ndcg_at_k(ranked: List[int], relevant: Set[int], k: int) -> float:
    """nDCG@k для бинарной релевантности"""
    dcg = 0.0
    for i, idx in enumerate(ranked[:k], start=1):
        if idx in relevant:
            dcg += 1.0 / np.log2(i + 1)
    ideal_dcg = sum([1.0 / np.log2(i + 1) for i in range(1, min(len(relevant), k) + 1)])
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def evaluate_ranking_metrics(neighbors: Dict[int, List[int]], relevance: Dict[int, Set[int]], k_values: List[int]) -> pd.DataFrame:
    """
    Вычисляет метрики ранжирования для множества запросов.
    neighbors: {qid -> [candidate_idx,...]}
    relevance: {qid -> set(candidate_idx,...)}
    """
    rows = []
    for qid, ranked in neighbors.items():
        rel = relevance.get(qid, set())
        row: Dict[str, Any] = {"query_id": qid, "MRR": compute_mrr(ranked, rel)}
        for k in k_values:
            row[f"Recall@{k}"] = compute_recall_at_k(ranked, rel, k)
            row[f"nDCG@{k}"] = compute_ndcg_at_k(ranked, rel, k)
        rows.append(row)
    return pd.DataFrame(rows)

# ============== Projection / visualization helpers ==============

def project_embeddings(embeddings: np.ndarray, method: str = "pca", n_components: int = 2, random_state: int = 42) -> Optional[np.ndarray]:
    if embeddings is None or len(embeddings) == 0:
        return None
    method = method.lower()
    if method == "pca":
        if PCA is None:
            return None
        pca = PCA(n_components=n_components, random_state=random_state)
        return pca.fit_transform(embeddings)
    elif method == "umap":
        if umap is None:
            return None
        reducer = umap.UMAP(n_components=n_components, random_state=random_state, n_neighbors=15, min_dist=0.1)
        return reducer.fit_transform(embeddings)
    else:
        return None

# ============== STS / robustness / pdf ==============

def _load_builtin_sts_sample(lang: str = "en") -> pd.DataFrame:
    data = [
        ("A man is playing guitar", "A person plays the guitar", 4.2),
        ("A man is playing guitar", "A child is playing soccer", 0.8),
        ("Two dogs are running", "A couple of dogs run", 4.0),
        ("He is not happy", "He is happy", 1.0),
        ("The price is 100", "The price is 101", 3.0),
    ]
    return pd.DataFrame(data, columns=["sentence1", "sentence2", "score"])

def load_sts_dataset(name: str = "stsb_multi_mt", lang: str = "en") -> pd.DataFrame:
    if load_dataset is None:
        return _load_builtin_sts_sample(lang)
    try:
        if name == "stsb":
            ds = load_dataset("glue", "stsb")
            df = pd.DataFrame(ds["validation"])
            df = df.rename(columns={"sentence1": "sentence1", "sentence2": "sentence2", "label": "score"})
            return df[["sentence1", "sentence2", "score"]]
        else:
            ds = load_dataset("stsb_multi_mt", name=lang)
            df = pd.DataFrame(ds["test"])
            return df[["sentence1", "sentence2", "similarity_score"]].rename(columns={"similarity_score": "score"})
    except Exception:
        return _load_builtin_sts_sample(lang)

def evaluate_sts(model: SentenceTransformer, df: pd.DataFrame, metric: str = "cosine", batch_size: int = 64) -> Dict[str, Any]:
    s1 = df["sentence1"].map(preprocess_text).tolist()
    s2 = df["sentence2"].map(preprocess_text).tolist()
    y = df["score"].astype(float).to_numpy()

    emb1 = encode_texts_in_batches(model, s1, batch_size)
    emb2 = encode_texts_in_batches(model, s2, batch_size)

    preds = []
    for i in range(len(s1)):
        preds.append(pair_score(emb1[i], emb2[i], metric=metric))
    preds = np.array(preds, dtype=float)

    if metric.lower() in {"euclidean", "manhattan"}:
        preds = -preds

    res = {"metric": metric, "n": len(df), "preds": preds, "labels": y}
    if spearmanr is not None:
        sp = spearmanr(preds, y).correlation
        res["spearman"] = float(sp) if sp is not None else None
    else:
        res["spearman"] = None
    if pearsonr is not None:
        try:
            pr = pearsonr(preds, y)[0]
        except Exception:
            pr = None
        res["pearson"] = float(pr) if pr is not None else None
    else:
        res["pearson"] = None
    return res

def _wordnet_synonyms(word: str) -> List[str]:
    out = set()
    if not _NLTK_OK or wn is None:
        return []
    try:
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                w = lemma.name().replace("_", " ").lower()
                if w != word.lower():
                    out.add(w)
    except Exception:
        pass
    return list(out)

def generate_robust_variants(text: str, max_per_word: int = 2) -> List[str]:
    t = preprocess_text(text)
    toks = t.split()
    variants = set()

    for i in range(min(len(toks) - 1, 2)):
        tmp = toks.copy()
        tmp[i], tmp[i+1] = tmp[i+1], tmp[i]
        variants.add(" ".join(tmp))

    if any(tok == "не" for tok in toks):
        variants.add(" ".join([tok for tok in toks if tok != "не"]))
    else:
        if toks:
            variants.add("не " + " ".join(toks))

    def bump_numbers(s: str) -> str:
        return re.sub(r"\d+", lambda m: str(int(m.group(0)) + 1), s)
    if re.search(r"\d+", t):
        variants.add(bump_numbers(t))

    for i, w in enumerate(toks[:5]):
        syns = _wordnet_synonyms(w)[:max_per_word]
        for s in syns:
            tmp = toks.copy()
            tmp[i] = s
            variants.add(" ".join(tmp))

    return [v for v in variants if v and v != t]

def robustness_probe(model: SentenceTransformer, pairs: List[Tuple[str, str]], metric: str = "cosine", batch_size: int = 64) -> pd.DataFrame:
    rows = []
    for (a, b) in pairs:
        base_a = preprocess_text(a); base_b = preprocess_text(b)
        base_emb_a = encode_texts_in_batches(model, [base_a], batch_size)[0]
        base_emb_b = encode_texts_in_batches(model, [base_b], batch_size)[0]
        base_score = pair_score(base_emb_a, base_emb_b, metric)

        variants = generate_robust_variants(base_a)
        if not variants:
            rows.append({"phrase_1": base_a, "phrase_2": base_b, "variant": base_a, "score": base_score, "delta": 0.0, "type": "base"})
            continue

        v_embs = encode_texts_in_batches(model, variants, batch_size)
        for v_text, v_emb in zip(variants, v_embs):
            s = pair_score(v_emb, base_emb_b, metric)
            rows.append({"phrase_1": base_a, "phrase_2": base_b, "variant": v_text, "score": s, "delta": s - base_score, "type": "variant"})
        rows.append({"phrase_1": base_a, "phrase_2": base_b, "variant": base_a, "score": base_score, "delta": 0.0, "type": "base"})
    return pd.DataFrame(rows)

def find_suspicious(df: pd.DataFrame,
                    score_col: str = "score",
                    lexical_col: Optional[str] = "lexical_score",
                    label_col: Optional[str] = None,
                    semantic_threshold: float = 0.80,
                    lexical_threshold: float = 0.30,
                    low_score_threshold: float = 0.75) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if score_col not in df.columns:
        return out
    if lexical_col and lexical_col in df.columns:
        out["high_sem_low_lex"] = df[(df[score_col] >= semantic_threshold) & (df[lexical_col] <= lexical_threshold)].copy()
    if label_col and label_col in df.columns:
        out["label_mismatch_positives"] = df[(df[label_col] == 1) & (df[score_col] < low_score_threshold)].copy()
        out["label_mismatch_negatives"] = df[(df[label_col] == 0) & (df[score_col] >= semantic_threshold)].copy()
    return out

def _save_plot_hist(values: np.ndarray, title: str, path: str):
    plt.figure()
    plt.hist(values, bins=30)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def _save_scatter(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, path: str):
    plt.figure()
    plt.scatter(x, y, alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def export_pdf_report(report_json: Dict[str, Any], charts: Dict[str, str], output_path: str) -> Optional[str]:
    if pdf_canvas is None or A4 is None or ImageReader is None:
        return None
    c = pdf_canvas.Canvas(output_path, pagesize=A4)
    w, h = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, h - 50, "Synonym Checker — Report")

    c.setFont("Helvetica", 9)
    y = h - 80
    c.drawString(40, y, "Summary:")
    y -= 14
    for k, v in list(report_json.items())[:20]:
        line = f"- {k}: {v}"
        c.drawString(50, y, line[:110])
        y -= 12
        if y < 120:
            c.showPage()
            y = h - 40

    c.setFont("Helvetica-Bold", 12)
    for name, path in charts.items():
        try:
            img = ImageReader(path)
            c.showPage()
            c.drawString(40, h - 40, name)
            c.drawImage(img, 40, 80, width=w-80, height=h-160, preserveAspectRatio=True, anchor='sw')
        except Exception:
            continue

    c.showPage()
    c.save()
    return output_path
