# app/ingest.py
import json
from typing import Tuple

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from app.config import load_config
from app.util import (
    ensure_dir,
    split_contexts,
    chunk_text,
    hash_text,
    embed_texts,
    normalize,
    auto_device,
)


def build_docs_and_queries(cfg) -> Tuple[pd.DataFrame, pd.DataFrame]:
    csv_path = cfg.paths.data_raw
    chunk_chars = int(cfg.ingest.chunk_size_tokens) * 4
    overlap_chars = int(cfg.ingest.chunk_overlap_tokens) * 4
    max_docs = cfg.ingest.get("max_docs")
    dedup = bool(cfg.ingest.get("dedup", True))

    df = pd.read_csv(csv_path)
    col_map = {c.lower(): c for c in df.columns}
    id_col = col_map.get("id", "id")
    q_col = col_map.get("question", "question")
    ctx_col = col_map.get("contexts", "contexts")
    label_col = col_map.get("final_decision", "final_decision")

    docs_rows, queries_rows = [], []

    for _, row in df.iterrows():
        pmid = str(row[id_col]).strip()
        question = str(row[q_col]).strip()
        label = str(row[label_col]).strip().lower() if not pd.isna(row[label_col]) else ""

        source_doc_ids = []
        for s_idx, snippet in enumerate(split_contexts(row[ctx_col])):
            for c_idx, chunk in enumerate(chunk_text(snippet, chunk_chars, overlap_chars)):
                doc_id = f"{pmid}#{s_idx}:{c_idx}"
                source_doc_ids.append(doc_id)
                docs_rows.append(
                    {"doc_id": doc_id, "text": chunk, "source_row_id": pmid, "pmid": pmid}
                )

        queries_rows.append(
            {"query_id": pmid, "question": question, "gold_label": label, "gold_source_ids": "|".join(source_doc_ids)}
        )

    docs = pd.DataFrame(docs_rows)
    queries = pd.DataFrame(queries_rows)

    if dedup and not docs.empty:
        before = len(docs)
        docs["text_hash"] = docs["text"].map(hash_text)
        docs = docs.drop_duplicates(subset=["text_hash"]).drop(columns=["text_hash"])
        after = len(docs)
        print(f"[ingest] deduped docs: {before} -> {after}")

    if isinstance(max_docs, int) and max_docs > 0 and not docs.empty:
        docs = docs.sample(n=min(max_docs, len(docs)), random_state=42).reset_index(drop=True)
        print(f"[ingest] subsampled docs to {len(docs)}")

    return docs, queries


def embed_and_index(docs: pd.DataFrame, cfg):
    model_name = cfg.models.embedding_model
    batch_size = int(cfg.models.get("embedding_batch_size", 128))
    do_norm = bool(cfg.models.get("normalize_embeddings", True))

    device = auto_device()
    print(f"[ingest] loading embedding model: {model_name} on {device}")
    encoder = SentenceTransformer(model_name, device=device)

    X = embed_texts(encoder, docs["text"].tolist(), batch_size)
    if do_norm:
        X = normalize(X)

    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    index_path = cfg.paths.vector_index
    meta_path = cfg.paths.vector_meta
    ensure_dir(index_path)
    ensure_dir(meta_path)
    faiss.write_index(index, index_path)
    with open(meta_path, "w") as f:
        json.dump(
            {
                "embedding_model": model_name,
                "dim": dim,
                "normalize": do_norm,
                "count": int(X.shape[0]),
                "index_type": "IndexFlatIP",
                "embedding_device": device,
                "faiss_device": "cpu",
            },
            f,
            indent=2,
        )

    print(f"[ingest] built FAISS index: {X.shape[0]} vectors, dim={dim}")
    return index_path, meta_path


def main():
    cfg = load_config()

    docs, queries = build_docs_and_queries(cfg)

    ensure_dir(cfg.paths.data_processed_docs)
    docs.to_parquet(cfg.paths.data_processed_docs, index=False)
    ensure_dir(cfg.paths.data_processed_queries)
    queries.to_parquet(cfg.paths.data_processed_queries, index=False)
    print(f"[ingest] wrote processed docs: {len(docs)}; queries: {len(queries)}")

    if len(docs) == 0:
        raise ValueError("No documents were created from contexts; check your CSV and parsing.")

    embed_and_index(docs, cfg)
    print("[ingest] DONE.")


if __name__ == "__main__":
    main()

# Run Python3 -m app.ingest
"""
outputs:

vectorstore/ --> faiss.index and meta.json
data/processed/ --> pubmedqa_train_clean_docs and pubmedqa_train_clean_queries

"""