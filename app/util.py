# app/util.py
import os
from typing import List
import hashlib

import pandas as pd
import numpy as np
import torch


# ---------- Filesystem ----------

def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ---------- Text utils ----------

def split_contexts(raw: str) -> List[str]:
    """
    PubMedQA 'contexts' may contain multiple abstracts/snippets in one cell.
    Split robustly and return unique, non-empty snippets in order.
    """
    if pd.isna(raw):
        return []
    s = str(raw).strip()
    # Try common separators first
    seps = ['","', '" , "', '" ,\'', '||', ' [SEP] ', ';;;']
    parts: List[str] = []
    for sep in seps:
        if sep in s:
            parts = [t.strip().strip('"').strip() for t in s.split(sep)]
            break
    if not parts:
        parts = [s.strip('"').strip()]
    seen, out = set(), []
    for t in parts:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def chunk_text(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    """
    Simple char-based chunking (token-free). Good for abstracts; reranker handles fine detail.
    """
    chunks, n, i = [], len(text), 0
    while i < n:
        j = min(i + max_chars, n)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j == n:
            break
        i = max(0, j - overlap_chars)
    return chunks


def hash_text(txt: str) -> str:
    return hashlib.md5(txt.encode("utf-8")).hexdigest()


# ---------- Vector math ----------

def normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms


# ---------- Devices / Embedding ----------

def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed_query(encoder, text: str) -> np.ndarray:
    with torch.inference_mode():
        vec = encoder.encode(
            [text],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=False,  # we'll normalize manually
            convert_to_numpy=True,
        )[0]
    v = vec.astype("float32")
    v /= (np.linalg.norm(v) + 1e-12)
    return v.reshape(1, -1)


def embed_texts(encoder, texts: List[str], batch_size: int) -> np.ndarray:
    chunks = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        with torch.inference_mode():
            vecs = encoder.encode(
                batch,
                batch_size=len(batch),
                show_progress_bar=False,
                normalize_embeddings=False,
                convert_to_numpy=True,
            )
        chunks.append(vecs)
    return np.vstack(chunks).astype("float32")
