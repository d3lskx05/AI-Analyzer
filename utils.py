# utils.py
# Универсальные утилиты и аналитика для Synonym Checker
from __future__ import annotations

import io
import os
import re
import json
import math
import time
import hashlib
import zipfile
import tarfile
import tempfile
import shutil
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer, util

# Метрики/аналитика
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr, pearsonr

# UMAP опционален
try:
    import umap  # umap-learn
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

# PDF отчёт
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Морфология (опционально)
try:
    import pymorphy2   # type: ignore
    _MORPH = pymorphy2.MorphAnalyzer()
except Exception:
    _MORPH = None

# ==========================
# БАЗОВЫЕ УТИЛИТЫ
# ==========================

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
                return [str(x).strip() for x in parsed if x is not None and str(x).strip()]
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

# Подсветка (для st.dataframe.style)
def style_suspicious_and_low(df, sem_thresh: float, lex_thresh: float, low_score_thresh: float):
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
        is_suspicious = (score >= sem_thresh) and (lex <= lex_thresh)
        for _ in row:
            if is_suspicious:
                out.append('background-color: #fff2b8')  # жёлтый
            elif is_low_score:
                out.append('background-color: #ffcccc')  # розовый
            else:
                out.append('')
        return out
    return df.style.apply(highlight, axis=1)

# Простые признаки
NEG_PAT = re.compile(r"\bне\b|\bни\b|\bнет\b|\bnot\b|\bno\b|\bnever\b", flags=re.IGNORECASE)
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
    a = np.asarray(a)
    b = np.asarray(b)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(np.mean(a[idx] - b[idx]))
    diffs = np.array(diffs)
    mean_diff = float(np.mean(diffs))
    low = float(np.quantile(diffs, (1-ci)/2))
    high = float(np.quantile(diffs, 1-(1-ci)/2))
    return mean_diff, low, high

# ==========================
# МОДЕЛИ
# ==========================

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

def load_model_from_source(source: str, identifier: str) -> SentenceTransformer:
    if source == "huggingface":
        model_path = identifier
    elif source == "google_drive":
        model_path = download_file_from_gdrive(identifier)
    else:
        raise ValueError("Unknown model source")
    model = SentenceTransformer(model_path)
    return model

def encode_texts_in_batches(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    if not texts:
        return np.array([])
    embs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(embs)

# ==========================
# ДОП. МЕТРИКИ
# ==========================

def sim_cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(util.cos_sim(a, b).item())

def sim_dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def sim_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(-np.linalg.norm(a - b))  # чем больше, тем ближе (отрицательная дистанция)

def compute_similarity(a: np.ndarray, b: np.ndarray, metric: str = "cosine") -> float:
    if metric == "cosine":
        return sim_cos(a, b)
    if metric == "dot":
        return sim_dot(a, b)
    if metric in ("euclidean", "l2"):
        return sim_l2(a, b)
    raise ValueError("Unknown metric")

# ==========================
# kNN / Top-N
# ==========================

def build_knn(embeddings: np.ndarray, metric: str = "cosine", n_neighbors: int = 10) -> NearestNeighbors:
    # Для cosine используем метрику 'cosine', для l2 — 'euclidean'
    met = "cosine" if metric in ("cosine", "dot") else "euclidean"
    # Для dot product — нормализуем эмбеддинги и используем cosine
    X = embeddings.copy()
    if metric == "dot":
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        X = X / norms
        met = "cosine"
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=met)
    nn.fit(X)
    return nn

def query_topn(nn: NearestNeighbors, embeddings: np.ndarray, idx: int, topn: int = 5, metric: str = "cosine") -> List[Tuple[int, float]]:
    X = embeddings.copy()
    if metric == "dot":
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        X = X / norms
    dists, inds = nn.kneighbors(X[idx:idx+1], n_neighbors=min(topn+1, len(X)))
    inds = inds[0].tolist()
    dists = dists[0].tolist()
    out = []
    for i, d in zip(inds, dists):
        if i == idx:
            continue
        # конвертация расстояния в "похожесть"
        if nn.metric == "cosine":
            sim = 1.0 - float(d)
        else:
            sim = -float(d)
        out.append((i, sim))
    return out[:topn]

# ==========================
# DIMRED (PCA / UMAP)
# ==========================

def reduce_embeddings(embeddings: np.ndarray, method: str = "umap", n_components: int = 2, random_state: int = 42) -> np.ndarray:
    if method == "umap" and _HAS_UMAP:
        reducer = umap.UMAP(n_components=n_components, random_state=random_state, n_neighbors=15, min_dist=0.1)
        return reducer.fit_transform(embeddings)
    # Fallback на PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(embeddings)

# ==========================
# ROBUSTNESS / BIAS
# ==========================

_REPLACEMENTS_SYNONYM = {
    # мини-словарик (язык-агностичные базовые случаи)
    "и": "а также",
    "или": "либо",
    "быстро": "скоростно",
    "медленно": "неторопливо",
    "не": "не",
    "not": "not",
    "good": "nice",
    "bad": "awful",
}

def toggle_negation(text: str) -> str:
    # простая эвристика: добавляем/убираем "не"/"not" около первого глагола/прилагательного
    # если уже есть отрицание — попробуем убрать
    if re.search(r"\bне\b", text, flags=re.I):
        return re.sub(r"\bне\b\s*", "", text, count=1, flags=re.I).strip()
    if re.search(r"\bnot\b", text, flags=re.I):
        return re.sub(r"\bnot\b\s*", "", text, count=1, flags=re.I).strip()
    # иначе — добавим "не" перед первым словом, которое не артикль/союз
    toks = text.split()
    for i, t in enumerate(toks):
        if len(t) > 2:
            toks.insert(i, "не")
            break
    return " ".join(toks)

def tweak_numbers(text: str, delta: int = 1) -> str:
    def repl(m):
        try:
            v = int(m.group(0))
            return str(max(v + delta, 0))
        except Exception:
            return m.group(0)
    return re.sub(r"\b\d+\b", repl, text)

def replace_synonyms(text: str) -> str:
    toks = text.split()
    out = []
    for t in toks:
        rep = _REPLACEMENTS_SYNONYM.get(t.lower(), None)
        out.append(rep if rep else t)
    return " ".join(out)

def drop_stop_like(text: str) -> str:
    # очень простой "стоп-ворд" дроппер для шума
    return " ".join([t for t in text.split() if t.lower() not in {"и", "а", "но", "или", "the", "a", "an", "to"}])

def generate_perturbations(text: str) -> Dict[str, str]:
    """Возвращает набор вариаций текста для robustness-проверок."""
    t = preprocess_text(text)
    return {
        "orig": t,
        "neg_flip": toggle_negation(t),
        "num_plus1": tweak_numbers(t, +1),
        "num_minus1": tweak_numbers(t, -1),
        "synonymish": replace_synonyms(t),
        "drop_stop": drop_stop_like(t),
        "swap_words": " ".join(reversed(t.split())) if len(t.split()) > 1 else t,
    }

def bias_edge_flags(p1: str, p2: str) -> Dict[str, bool]:
    f1 = simple_flags(p1); f2 = simple_flags(p2)
    return {
        "neg_mismatch": f1["has_neg"] ^ f2["has_neg"],
        "num_mismatch": f1["has_num"] ^ f2["has_num"],
        "date_mismatch": f1["has_date"] ^ f2["has_date"],
    }

# ==========================
# БЕНЧМАРКИ / КОРРЕЛЯЦИИ
# ==========================

def compute_correlations(df: pd.DataFrame, score_col: str, gold_col: str) -> Dict[str, float]:
    """
    Если gold_col — вещественная «истинная» похожесть (0..1/0..5), считаем Spearman/Pearson.
    Если gold_col бинарный {0,1}, считаем accuracy (по оптимальному порогу).
    """
    scores = df[score_col].astype(float).to_numpy()
    gold = df[gold_col].to_numpy()

    # бинарный?
    is_binary = set(pd.Series(gold).dropna().unique()).issubset({0, 1, 0.0, 1.0})

    out = {}
    if is_binary:
        # подберём порог по максимуму accuracy
        best_acc, best_thr = -1.0, 0.5
        for thr in np.linspace(0.0, 1.0, 101):
            pred = (scores >= thr).astype(int)
            acc = accuracy_score(gold, pred)
            if acc > best_acc:
                best_acc, best_thr = acc, thr
        out.update({"accuracy": float(best_acc), "best_threshold": float(best_thr)})
    else:
        # нормализуем gold в [0,1] если похоже на 0..5
        g = gold.astype(float)
        g_min, g_max = np.nanmin(g), np.nanmax(g)
        if g_max - g_min > 1.5:  # вероятно шкала 0..5
            g = (g - g_min) / (g_max - g_min + 1e-12)
        sp = spearmanr(scores, g, nan_policy="omit")
        pr = pearsonr(scores, g)
        out.update({
            "spearman": float(sp.correlation) if sp.correlation is not None else np.nan,
            "pearson": float(pr.statistic) if hasattr(pr, "statistic") else float(pr[0]),
        })
    return out

# ==========================
# ОТЧЁТ (PDF)
# ==========================

def _save_temp_plot(fig) -> str:
    import matplotlib.pyplot as plt
    tmp = os.path.join(tempfile.gettempdir(), f"sc_plot_{int(time.time()*1000)}.png")
    fig.savefig(tmp, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return tmp

def build_pdf_report(
    pdf_path: str,
    title: str,
    summary: Dict[str, Any],
    charts: List[str]  # пути к png графикам
) -> str:
    c = canvas.Canvas(pdf_path, pagesize=A4)
    W, H = A4
    y = H - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, title)
    y -= 30
    c.setFont("Helvetica", 10)
    for k, v in summary.items():
        line = f"{k}: {v}"
        c.drawString(40, y, line)
        y -= 14
        if y < 100:
            c.showPage()
            y = H - 50
    for img_path in charts:
        if y < 300:
            c.showPage()
            y = H - 50
        try:
            img = ImageReader(img_path)
            iw, ih = img.getSize()
            scale = min((W-80)/iw, 400/ih)
            c.drawImage(img, 40, y-ih*scale, width=iw*scale, height=ih*scale)
            y -= ih*scale + 20
        except Exception:
            continue
    c.showPage()
    c.save()
    return pdf_path
