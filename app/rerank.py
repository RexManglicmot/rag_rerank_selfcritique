# app/rerank.py
from typing import Dict, List, Any

import pandas as pd
from sentence_transformers import CrossEncoder

from app.config import load_config, print_summary
from app.util import auto_device


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


class Reranker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.models.reranker_model
        self.device = auto_device()
        self.batch_size = int(cfg.models.get("reranker_batch_size", 64))
        self.max_length = int(cfg.models.get("reranker_max_length", 512))
        self.encoder = CrossEncoder(self.model_name, device=self.device, max_length=self.max_length)
        print(f"[rerank] CrossEncoder={self.model_name} device={self.device}")

        self.primary_k = int(cfg.eval.primary_k)
        self.m_context = int(cfg.retrieval.m_context)

    def rerank(
        self,
        query_id: str,
        question: str,
        candidate_doc_ids: List[str],
        candidate_texts: List[str],
        gold_source_ids_str: str,
    ) -> Dict[str, Any]:
        gold_ids: List[str] = []
        if isinstance(gold_source_ids_str, str) and gold_source_ids_str.strip():
            gold_ids = [s.strip() for s in gold_source_ids_str.split("|") if s.strip()]

        # Build pairs: (question, doc_text)
        pairs = [(question, d) for d in candidate_texts]

        # Score with CrossEncoder
        scores = self.encoder.predict(pairs, batch_size=self.batch_size).tolist()

        # Sort candidates by reranker score
        reranked = sorted(
            zip(candidate_doc_ids, candidate_texts, scores),
            key=lambda x: x[2],
            reverse=True,
        )

        reranked_doc_ids = [r[0] for r in reranked]
        reranked_scores = [float(r[2]) for r in reranked]

        # Compute metrics
        rank_first_post = _rank_first_relevant(reranked_doc_ids, gold_ids)
        recall_post = _recall_at_k(reranked_doc_ids, gold_ids, self.primary_k)

        # Keep only top-m for generator
        top_m_ids = reranked_doc_ids[: self.m_context]
        top_m_texts = [r[1] for r in reranked[: self.m_context]]

        return {
            "query_id": query_id,
            "doc_ids_reranked": reranked_doc_ids,
            "scores_reranked": reranked_scores,
            "rank_first_rel_post": rank_first_post,
            f"recall_at_{self.primary_k}_post": recall_post,
            "context_doc_ids_reranked": top_m_ids,
            "context_texts_reranked": top_m_texts,
        }


if __name__ == "__main__":
    cfg = load_config()
    print_summary(cfg)
    rr = Reranker(cfg)

    # Demo: load a few queries + docs to rerank
    queries = pd.read_parquet(cfg.paths.data_processed_queries)
    docs = pd.read_parquet(cfg.paths.data_processed_docs)

    q = queries.iloc[0]
    qid, question, gold = q["query_id"], q["question"], q["gold_source_ids"]

    # Pick first 5 docs as fake retrieval candidates
    cand_doc_ids = docs.head(5)["doc_id"].tolist()
    cand_texts = docs.head(5)["text"].tolist()

    out = rr.rerank(qid, question, cand_doc_ids, cand_texts, gold)
    print(f"[rerank] query_id={qid} rank_first_rel_post={out['rank_first_rel_post']} recall_post={out[f'recall_at_{rr.primary_k}_post']}")
    print(f"[rerank] top-m ids={out['context_doc_ids_reranked']}")

# Run Python3 -m app.rerank