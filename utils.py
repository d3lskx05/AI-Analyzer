# utils.py — ядро: загрузка/кодирование моделей, чтение данных, метрики, визуализация, экспорт

from __future__ import annotations
import io, os, json, re, tempfile, zipfile, tarfile, shutil, hashlib, random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# ====== Модели / фреймворки ======
from sentence_transformers import SentenceTransformer, util as st_util

# Gensim (опционально)
try:
    import gensim
    from gensim.models import KeyedVectors
    _HAS_GENSIM = True
except Exception:
    _HAS_GENSIM = False

# ====== ML / визуализация ======
from sklearn.metrics import (
    roc_curve, precision_recall_curve, f1_score, precision_score, recall_score, auc
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import matplotlib
matplotlib.use("Agg")  # важно для рендеринга без X-сервера (Streamlit Cloud)
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages

# ============== Общие утилиты ==============

def file_md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def preprocess_text(t: Any) -> str:
    if pd.isna(t):
        return ""
    return " ".join(str(t).lower().strip().split())

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

def _try_read_json(raw: bytes) -> pd.DataFrame:
    # list[dict] / dict(columns) / NDJSON
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
    if name.endswith((".json", ".ndjson")):
        df = _try_read_json(raw)
        return df, h
    # CSV
    try:
        df = pd.read_csv(io.BytesIO(raw))
        return df, h
    except Exception:
        pass
    # Excel
    try:
        df = pd.read_excel(io.BytesIO(raw))
        return df, h
    except Exception as e:
        raise ValueError("Файл должен быть CSV, Excel или JSON. Ошибка: " + str(e))

def jaccard_tokens(a: str, b: str) -> float:
    sa = set([t for t in a.split() if t])
    sb = set([t for t in b.split() if t])
    if not sa and not sb:
        return 0.0
    inter = sa & sb
    union = sa | sb
    return len(inter) / len(union) if union else 0.0

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

# ============== Загрузка моделей ==============

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
        # На случай если это уже директория/файл модели
        try:
            shutil.copy(archive_path, model_dir)
        except Exception:
            pass
    return model_dir

# Обёртки для разных источников
@dataclass
class BaseEncoder:
    name: str
    kind: str  # "hf", "gensim", "dict"
    dim: int

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        raise NotImplementedError

class HFEncoder(BaseEncoder):
    def __init__(self, model_path: str):
        self.model = SentenceTransformer(model_path)
        super().__init__(name=model_path, kind="hf", dim=int(self.model.get_sentence_embedding_dimension()))

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)

class GensimEncoder(BaseEncoder):
    def __init__(self, kv: KeyedVectors, name: str = "gensim"):
        self.kv = kv
        super().__init__(name=name, kind="gensim", dim=int(kv.vector_size))

    def _sent_emb(self, text: str) -> np.ndarray:
        toks = [t for t in text.split() if t]
        vecs = []
        for t in toks:
            if t in self.kv.key_to_index:
                vecs.append(self.kv[t])
        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)
        return np.mean(np.stack(vecs, axis=0), axis=0)

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._sent_emb(t) for t in texts], axis=0).astype(np.float32)

class DictEncoder(BaseEncoder):
    """Локальный словарь: token -> vector (например, JSON). Простое усреднение."""
    def __init__(self, token2vec: Dict[str, List[float]], name: str = "dict"):
        self.t2v = {k: np.array(v, dtype=np.float32) for k, v in token2vec.items()}
        dims = {len(v) for v in self.t2v.values()} or {0}
        self._dim = list(dims)[0]
        super().__init__(name=name, kind="dict", dim=self._dim)

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        def sent_emb(text: str) -> np.ndarray:
            toks = [t for t in text.split() if t]
            vecs = [self.t2v[t] for t in toks if t in self.t2v]
            if not vecs:
                return np.zeros(self._dim, dtype=np.float32)
            return np.mean(np.stack(vecs, axis=0), axis=0)
        return np.stack([sent_emb(t) for t in texts], axis=0)

def load_model_from_source(source: str, identifier: str) -> BaseEncoder:
    if source == "huggingface":
        return HFEncoder(identifier)
    elif source == "google_drive":
        path = download_file_from_gdrive(identifier)
        return HFEncoder(path)
    elif source == "gensim_path":
        if not _HAS_GENSIM:
            raise RuntimeError("Gensim недоступен в окружении")
        kv = KeyedVectors.load(identifier) if identifier.endswith(".kv") else KeyedVectors.load_word2vec_format(identifier, binary=identifier.endswith(".bin"))
        return GensimEncoder(kv, name=f"gensim:{os.path.basename(identifier)}")
    elif source == "local_dict_json":
        with open(identifier, "r", encoding="utf-8") as f:
            token2vec = json.load(f)
        return DictEncoder(token2vec, name=f"dict:{os.path.basename(identifier)}")
    else:
        raise ValueError("Unknown model source")

# ============== Вычисления / метрики ==============

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(st_util.cos_sim(a, b).item())

def scores_for_pairs(embs: np.ndarray, pairs: List[Tuple[int, int]]) -> np.ndarray:
    # cos для каждой пары (i,j)
    out = []
    for i, j in pairs:
        out.append(float(st_util.cos_sim(embs[i], embs[j]).item()))
    return np.array(out, dtype=np.float32)

def compute_clf_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    pr_prec, pr_rec, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(pr_rec, pr_prec)
    return {"f1": float(f1), "precision": float(prec), "recall": float(rec), "roc_auc": float(roc_auc), "pr_auc": float(pr_auc)}

def plot_roc_pr(y_true: np.ndarray, y_score: np.ndarray, title_suffix: str = "") -> Tuple[plt.Figure, plt.Figure]:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)

    fig1 = plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr)
    plt.title(f"ROC {title_suffix}".strip())
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.grid(True)

    fig2 = plt.figure(figsize=(5, 4))
    plt.plot(rec, prec)
    plt.title(f"PR {title_suffix}".strip())
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)

    return fig1, fig2

def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 500, seed: int = 42, ci: float = 0.95) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = min(len(a), len(b))
    if n == 0:
        return 0.0, 0.0, 0.0
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(np.mean(a[idx] - b[idx]))
    diffs = np.array(diffs)
    mean_diff = float(np.mean(diffs))
    low = float(np.quantile(diffs, (1 - ci) / 2))
    high = float(np.quantile(diffs, 1 - (1 - ci) / 2))
    return mean_diff, low, high

# ============== Генерация/устойчивость/edge ==============

def generate_pairs_auto(
    texts: List[str],
    n_pos: int = 200,
    n_neg: int = 200,
    synonyms: Optional[Dict[str, str]] = None,
    seed: int = 13,
) -> pd.DataFrame:
    """
    Положительные пары: t & t' (t' — лёгкая пертурбация t: замена на синоним, дроп слова, перестановка).
    Отрицательные пары: случайные тексты.
    """
    rng = random.Random(seed)
    texts = [t for t in texts if t]
    uniq = list(dict.fromkeys(texts))
    if len(uniq) < 2:
        return pd.DataFrame(columns=["phrase_1", "phrase_2", "label", "kind"])

    def perturb(t: str) -> str:
        toks = t.split()
        if not toks:
            return t
        # 1) замена синонима
        if synonyms:
            for k, v in synonyms.items():
                if k in toks or k in t:
                    return t.replace(k, v)
        # 2) дроп слова
        if len(toks) > 1 and rng.random() < 0.5:
            idx = rng.randrange(len(toks))
            return " ".join([w for i, w in enumerate(toks) if i != idx])
        # 3) перестановка пары слов
        if len(toks) > 2:
            i, j = sorted(random.sample(range(len(toks)), 2))
            toks[i], toks[j] = toks[j], toks[i]
            return " ".join(toks)
        return t

    pos_pairs = []
    for _ in range(n_pos):
        t = rng.choice(uniq)
        pos_pairs.append((t, perturb(t)))

    neg_pairs = []
    for _ in range(n_neg):
        a, b = rng.sample(uniq, 2)
        neg_pairs.append((a, b))

    df = pd.DataFrame(
        pos_pairs + neg_pairs,
        columns=["phrase_1", "phrase_2"]
    )
    df["label"] = [1] * len(pos_pairs) + [0] * len(neg_pairs)
    df["kind"] = ["pos"] * len(pos_pairs) + ["neg"] * len(neg_pairs)
    return df

def detect_edge_cases(df: pd.DataFrame, col1="phrase_1", col2="phrase_2") -> pd.DataFrame:
    edges = []
    for _, r in df.iterrows():
        f1 = simple_flags(str(r[col1]))
        f2 = simple_flags(str(r[col2]))
        edge = (f1["has_neg"] != f2["has_neg"]) or (f1["has_num"] != f2["has_num"]) or (f1["has_date"] != f2["has_date"])
        edges.append(edge)
    df = df.copy()
    df["edge_case"] = edges
    return df

def robustness_delta(model: BaseEncoder, pairs: pd.DataFrame, synonyms: Optional[Dict[str, str]] = None, batch_size: int = 64) -> pd.DataFrame:
    """
    Считаем, как меняется косинус, если обе фразы подвергнуть пертурбации (синонимы/дроп слова).
    """
    def apply_syn(text: str) -> str:
        if not synonyms:
            return text
        out = text
        for k, v in synonyms.items():
            out = out.replace(k, v)
        toks = out.split()
        if len(toks) > 3 and random.random() < 0.3:
            toks.pop(random.randrange(len(toks)))
            out = " ".join(toks)
        return out

    texts = list(dict.fromkeys(pairs["phrase_1"].tolist() + pairs["phrase_2"].tolist()))
    embs = model.encode(texts, batch_size=batch_size)
    idx = {t: i for i, t in enumerate(texts)}

    base_scores = []
    pert_scores = []
    for _, r in pairs.iterrows():
        a, b = str(r["phrase_1"]), str(r["phrase_2"])
        base = float(st_util.cos_sim(embs[idx[a]], embs[idx[b]]).item())
        a2, b2 = apply_syn(a), apply_syn(b)
        e2 = model.encode([a2, b2], batch_size=batch_size)
        pert = float(st_util.cos_sim(e2[0], e2[1]).item())
        base_scores.append(base)
        pert_scores.append(pert)
    df = pairs.copy()
    df["score_base"] = base_scores
    df["score_perturbed"] = pert_scores
    df["delta"] = df["score_perturbed"] - df["score_base"]
    return df

def auto_suspicious(df: pd.DataFrame, score_col: str = "score", label_col: str = "label",
                    pos_low: float = 0.30, neg_high: float = 0.75) -> pd.DataFrame:
    """
    Авто-детектор «подозрительных» пар:
      - label=1, но score < pos_low
      - label=0, но score > neg_high
    """
    if label_col not in df.columns or score_col not in df.columns:
        return pd.DataFrame(columns=df.columns)
    mask = ((df[label_col] == 1) & (df[score_col] < pos_low)) | ((df[label_col] == 0) & (df[score_col] > neg_high))
    return df[mask].copy()

# ============== Визуализация эмбеддингов ==============

def reduce_embeddings(embs: np.ndarray, method: str = "pca", dim: int = 2, random_state: int = 42) -> np.ndarray:
    if embs.shape[0] == 0:
        return np.zeros((0, dim))
    if method == "pca":
        reducer = PCA(n_components=dim, random_state=random_state)
    else:
        reducer = TSNE(n_components=dim, random_state=random_state, init="pca", learning_rate="auto")
    return reducer.fit_transform(embs)

def plot_scatter2d(coords: np.ndarray, labels: Optional[List[Any]] = None, title: str = "Embeddings 2D") -> plt.Figure:
    fig = plt.figure(figsize=(6, 5))
    if labels is None:
        plt.scatter(coords[:, 0], coords[:, 1], s=12, alpha=0.8)
    else:
        sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels, s=18, palette="Set2", legend=True)
    plt.title(title)
    plt.tight_layout()
    return fig

def plot_scatter3d(coords3: np.ndarray, labels: Optional[List[Any]] = None, title: str = "Embeddings 3D") -> plt.Figure:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    if labels is None:
        ax.scatter(coords3[:, 0], coords3[:, 1], coords3[:, 2], s=12, alpha=0.8)
    else:
        # цвет по метке
        uniq = list(dict.fromkeys(labels))
        cmap = plt.cm.get_cmap("tab10", len(uniq))
        for i, u in enumerate(uniq):
            idx = [k for k, l in enumerate(labels) if l == u]
            ax.scatter(coords3[idx, 0], coords3[idx, 1], coords3[idx, 2], s=15, alpha=0.9, label=str(u), c=[cmap(i)])
        ax.legend()
    ax.set_title(title)
    plt.tight_layout()
    return fig

def cluster_and_mark(embs: np.ndarray, n_clusters: int = 8, random_state: int = 42) -> np.ndarray:
    if embs.shape[0] < n_clusters:
        return np.arange(embs.shape[0]) % max(1, n_clusters)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    return km.fit_predict(embs)

def model_correlation_heatmap(scores_by_model: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str], plt.Figure]:
    names = list(scores_by_model.keys())
    mat = np.zeros((len(names), len(names)), dtype=np.float32)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                mat[i, j] = 1.0
            else:
                v1, v2 = scores_by_model[a], scores_by_model[b]
                if len(v1) == len(v2) and len(v1) > 1:
                    mat[i, j] = float(np.corrcoef(v1, v2)[0, 1])
                else:
                    mat[i, j] = np.nan
    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(mat, xticklabels=names, yticklabels=names, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Model Score Correlation")
    plt.tight_layout()
    return mat, names, fig

# ============== Экспорт / история ==============

def export_pdf(path: str, title_text: str, summary: Dict[str, Any], figures: List[plt.Figure]) -> None:
    with PdfPages(path) as pdf:
        # титульный лист
        fig0 = plt.figure(figsize=(8.27, 11.69))  # A4 портрет
        plt.axis("off")
        y = 0.95
        plt.text(0.05, y, title_text, fontsize=18, weight="bold")
        y -= 0.06
        for k, v in summary.items():
            plt.text(0.05, y, f"{k}: {v}", fontsize=11)
            y -= 0.035
        pdf.savefig(fig0); plt.close(fig0)

        for f in figures:
            pdf.savefig(f); plt.close(f)

def make_run_record(**kwargs) -> Dict[str, Any]:
    return kwargs
