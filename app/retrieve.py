# app/retrieve.py
from typing import Dict, List, Any

import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from app.config import load_config, print_summary
from app.util import auto_device, embed_query


def _rank_first_relevant(ranked_doc_ids: List[str], gold_ids: List[str]) -> int:
    gold = set(gold_ids)
    for i, did in enumerate(ranked_doc_ids, start=1):
        if did in gold:
            return i
    return 0

def _recall_at_k(ranked_doc_ids: List[str], gold_ids: List[str], k: int) -> int:
    k = min(k, len(ranked_doc_ids))
    topk = set(ranked_doc_ids[:k])
    return int(any(g in topk for g in gold_ids))


class Retriever:
    def __init__(self, cfg):
        self.cfg = cfg
        self.docs = pd.read_parquet(cfg.paths.data_processed_docs)
        self._rowid_to_docid = self.docs["doc_id"].tolist()

        self.index = faiss.read_index(cfg.paths.vector_index)

        self.model_name = cfg.models.embedding_model
        self.device = auto_device()
        self.encoder = SentenceTransformer(self.model_name, device=self.device)
        print(f"[retrieve] encoder={self.model_name} device={self.device}")

        self.primary_k = int(cfg.eval.primary_k)
        self.k_candidates = int(cfg.retrieval.k_candidates)

    def search(self, query_id: str, question: str, gold_source_ids_str: str, k: int = None) -> Dict[str, Any]:
        if k is None:
            k = self.k_candidates

        gold_ids: List[str] = []
        if isinstance(gold_source_ids_str, str) and gold_source_ids_str.strip():
            gold_ids = [s.strip() for s in gold_source_ids_str.split("|") if s.strip()]

        qv = embed_query(self.encoder, question)  # (1, d), normalized
        scores, idxs = self.index.search(qv, k)   # (1, k)
        idxs = idxs[0].tolist()
        scores = scores[0].tolist()

        doc_ids, doc_scores = [], []
        for row_idx, sc in zip(idxs, scores):
            if 0 <= row_idx < len(self._rowid_to_docid):
                doc_ids.append(self._rowid_to_docid[row_idx])
                doc_scores.append(float(sc))

        rank_first = _rank_first_relevant(doc_ids, gold_ids)
        recall_pk = _recall_at_k(doc_ids, gold_ids, self.primary_k)

        return {
            "query_id": query_id,
            "k": k,
            "doc_ids": doc_ids,
            "scores": doc_scores,
            "rank_first_rel_pre": rank_first,
            f"recall_at_{self.primary_k}": recall_pk,
        }

    def batch_search(self, queries_df: pd.DataFrame, k: int = None):
        if k is None:
            k = self.k_candidates
        need = {"query_id", "question", "gold_source_ids"}
        if not need.issubset(queries_df.columns):
            missing = need - set(queries_df.columns)
            raise ValueError(f"[retrieve] queries_df missing columns: {missing}")
        out = []
        for _, row in queries_df.iterrows():
            out.append(self.search(
                query_id=str(row["query_id"]),
                question=str(row["question"]),
                gold_source_ids_str=str(row["gold_source_ids"]) if pd.notna(row["gold_source_ids"]) else "",
                k=k,
            ))
        return out


if __name__ == "__main__":
    cfg = load_config()
    print_summary(cfg)
    queries = pd.read_parquet(cfg.paths.data_processed_queries)
    print(f"[retrieve] loaded {len(queries)} queries")
    r = Retriever(cfg)
    for o in r.batch_search(queries.head(3), k=cfg.retrieval.k_candidates):
        print(
            f"[retrieve] query_id={o['query_id']} "
            f"rank_first_rel_pre={o['rank_first_rel_pre']} "
            f"recall@{r.primary_k}={o[f'recall_at_{r.primary_k}']} "
            f"top1={o['doc_ids'][0] if o['doc_ids'] else 'NA'}"
        )

# Run Python3 -m app.retrieve
