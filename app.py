# utils.py — утилиты для исследовательской аналитики эмбеддингов

import pandas as pd
import numpy as np
import hashlib
import json
import io
import zipfile
import tarfile
import shutil
import tempfile
import os
from typing import List, Tuple, Dict, Any

from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
import umap
import re


# ============== Препроцессинг ==============

def preprocess_text(t: Any) -> str:
    """Простой препроцессинг: lower + trim + нормализация пробелов"""
    if pd.isna(t):
        return ""
    return " ".join(str(t).lower().strip().split())


def file_md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


# ============== Загрузка датасетов ==============

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


# ============== Метрики ==============

def jaccard_tokens(a: str, b: str) -> float:
    sa = set([t for t in a.split() if t])
    sb = set([t for t in b.split() if t])
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def precision_at_k(y_true: List[int], y_pred: List[int], k: int) -> float:
    if not y_true or not y_pred:
        return 0.0
    y_true_set = set(y_true)
    return len(set(y_pred[:k]) & y_true_set) / min(k, len(y_pred))


def mean_reciprocal_rank(y_true: List[int], y_pred: List[int]) -> float:
    y_true_set = set(y_true)
    for i, p in enumerate(y_pred):
        if p in y_true_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(y_true: List[int], y_pred: List[int], k: int) -> float:
    y_true_set = set(y_true)
    dcg = 0.0
    for i, p in enumerate(y_pred[:k]):
        if p in y_true_set:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(y_true), k)))
    return dcg / idcg if idcg > 0 else 0.0


# ============== Модели и эмбеддинги ==============

def load_model(model_id: str) -> SentenceTransformer:
    return SentenceTransformer(model_id)


def encode_texts_in_batches(model, texts: List[str], batch_size: int = 64) -> np.ndarray:
    if not texts:
        return np.array([])
    return np.asarray(model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False))


# ============== Аналитика ==============

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


# ============== Кластеризация ==============

def cluster_embeddings(embeddings: np.ndarray, n_clusters: int = 10, random_state: int = 42):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=random_state)
    emb_2d = reducer.fit_transform(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(emb_2d)
    return emb_2d, labels
