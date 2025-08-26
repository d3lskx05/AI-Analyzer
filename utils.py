import pandas as pd
import numpy as np
import io, os, json, tempfile, zipfile, tarfile, shutil, re
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import random

# ================== Preprocessing ==================
def preprocess_text(t: Any) -> str:
    if pd.isna(t): return ""
    return " ".join(str(t).lower().strip().split())

# ================== File handling ==================
def read_uploaded_file_bytes(uploaded) -> pd.DataFrame:
    raw = uploaded.read()
    name = (uploaded.name or "").lower()
    if name.endswith(".json"):
        try:
            return pd.read_json(io.BytesIO(raw))
        except:
            return pd.DataFrame(json.loads(raw.decode("utf-8")))
    elif name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw))
    elif name.endswith((".xlsx","xls")):
        return pd.read_excel(io.BytesIO(raw))
    else:
        raise ValueError("Поддерживаются только CSV, Excel или JSON")

# ================== Simple flags ==================
NEG_PAT = re.compile(r"\bне\b|\bни\b|\bнет\b", flags=re.IGNORECASE)
NUM_PAT = re.compile(r"\b\d+\b")
DATE_PAT = re.compile(r"\b\d{1,2}[./-]\d{1,2}([./-]\d{2,4})?\b")

def simple_flags(text: str) -> Dict[str,bool]:
    t = text or ""
    return {
        "has_neg": bool(NEG_PAT.search(t)),
        "has_num": bool(NUM_PAT.search(t)),
        "has_date": bool(DATE_PAT.search(t)),
        "len_char": len(t),
        "len_tok": len(t.split())
    }

# ================== Models ==================
def load_model(source: str, identifier: str):
    if source=="huggingface":
        return SentenceTransformer(identifier)
    elif source=="google_drive":
        return SentenceTransformer(download_file_from_gdrive(identifier))
    else:
        raise ValueError("Unknown source")

def download_file_from_gdrive(file_id: str) -> str:
    import gdown
    tmp_dir = tempfile.gettempdir()
    archive_path = os.path.join(tmp_dir,f"model_gdrive_{file_id}")
    model_dir = os.path.join(tmp_dir,f"model_gdrive_extracted_{file_id}")
    if not os.path.exists(archive_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, archive_path, quiet=True)
    if os.path.exists(model_dir): return model_dir
    os.makedirs(model_dir, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path,'r') as zip_ref: zip_ref.extractall(model_dir)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path,'r:*') as tar_ref: tar_ref.extractall(model_dir)
    else: shutil.copy(archive_path, model_dir)
    return model_dir

def encode_texts_in_batches(model, texts: List[str], batch_size:int=64) -> np.ndarray:
    if not texts: return np.array([])
    return model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)

# ================== Similarity ==================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(util.cos_sim(a, b).item())

def jaccard_tokens(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb: return 0.0
    return len(sa & sb)/len(sa | sb)

# ================== Metrics ==================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, plot=True) -> Dict[str,float]:
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    f1 = f1_score(y_true, y_pred>0.5)
    prec = precision_score(y_true, y_pred>0.5)
    rec = recall_score(y_true, y_pred>0.5)
    if plot:
        fig, axs = plt.subplots(1,2,figsize=(12,5))
        axs[0].plot(fpr,tpr,label=f"F1={f1:.2f}"); axs[0].set_title("ROC"); axs[0].legend(); axs[0].grid(True)
        axs[1].plot(recall,precision,label=f"F1={f1:.2f}"); axs[1].set_title("Precision-Recall"); axs[1].legend(); axs[1].grid(True)
        plt.show()
    return {"f1":f1, "precision":prec, "recall":rec}

# ================== Robustness / Synonym tests ==================
def generate_synonym_pairs(df: pd.DataFrame, text_col: str, n:int=1000) -> pd.DataFrame:
    """Автоматически генерирует пары текстов для тестирования модели"""
    texts = df[text_col].dropna().tolist()
    pairs = []
    for _ in range(n):
        a, b = random.sample(texts, 2)
        pairs.append({"text1":a, "text2":b, "label": int(a==b)})
    return pd.DataFrame(pairs)

def perturb_text(text: str, synonyms: Dict[str,str]) -> str:
    for k,v in synonyms.items(): text = text.replace(k,v)
    return text

def robustness_test(model, df_pairs: pd.DataFrame, text_col1="text1", text_col2="text2", synonyms={}):
    """Проверка устойчивости модели при замене слов"""
    scores = []
    for _, row in df_pairs.iterrows():
        t1 = perturb_text(row[text_col1], synonyms)
        t2 = perturb_text(row[text_col2], synonyms)
        emb1 = encode_texts_in_batches(model,[t1])
        emb2 = encode_texts_in_batches(model,[t2])
        scores.append(cosine_similarity(emb1[0], emb2[0]))
    return np.array(scores)

# ================== Visualization ==================
def plot_embeddings_2d(model, texts: List[str], labels: List[int]=None):
    embs = encode_texts_in_batches(model, texts)
    tsne = TSNE(n_components=2, random_state=42)
    coords = tsne.fit_transform(embs)
    df_plot = pd.DataFrame(coords, columns=["x","y"])
    if labels is not None: df_plot["label"]=labels
    plt.figure(figsize=(8,6))
    if labels is not None:
        sns.scatterplot(x="x",y="y",hue="label",data=df_plot,palette="Set1")
    else:
        plt.scatter(df_plot["x"], df_plot["y"])
    plt.title("2D Embeddings Scatter")
    plt.show()
    return df_plot

def plot_heatmap_scores(scores: np.ndarray, labels: List[str]):
    """Heatmap для сравнения нескольких моделей"""
    plt.figure(figsize=(8,6))
    sns.heatmap(scores, xticklabels=labels, yticklabels=labels, annot=True, cmap="coolwarm")
    plt.title("Model Comparison Heatmap")
    plt.show()
