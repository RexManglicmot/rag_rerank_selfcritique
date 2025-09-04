# app/pipeline.py
# Orchestrates: retrieve -> (optional) rerank -> generate for ONE variant.
# Returns a per-query DataFrame; optionally writes/append to outputs/eval.csv.

import os
from typing import Dict, List, Any

import pandas as pd

from app.config import load_config, print_summary
from app.util import ensure_dir
from app.retrieve import Retriever
from app.rerank import Reranker
from app.generate import LocalGenerator


def _prepare_gold(gold_source_ids: str) -> List[str]:
    if not isinstance(gold_source_ids, str) or not gold_source_ids.strip():
        return []
    return [s.strip() for s in gold_source_ids.split("|") if s.strip()]


def run_pipeline(cfg, variant: Dict[str, Any], queries: pd.DataFrame, docs: pd.DataFrame, save_csv: bool = True) -> pd.DataFrame:
    name = variant["name"]
    use_reranker = bool(variant.get("use_reranker", False))

    # per-variant override for self_critique if provided
    sc_over = variant.get("self_critique")
    if sc_over is not None and isinstance(sc_over, dict):
        cfg.self_critique.enabled = bool(sc_over.get("enabled", cfg.self_critique.enabled))
        if "passes" in sc_over:
            cfg.self_critique.passes = sc_over["passes"]
        if "mode" in sc_over:
            cfg.self_critique.mode = sc_over["mode"]

    retriever = Retriever(cfg)
    reranker = Reranker(cfg) if use_reranker else None
    generator = LocalGenerator(cfg)

    primary_k = int(cfg.eval.primary_k)
    mrr_k = int(cfg.eval.get("mrr_k", 10))
    ndcg_k = int(cfg.eval.get("ndcg_k", 10))
    k_candidates = int(cfg.retrieval.k_candidates)
    m_context = int(cfg.retrieval.m_context)

    # doc_id -> text
    doc_text = dict(zip(docs["doc_id"].tolist(), docs["text"].tolist()))

    rows = []

    for _, q in queries.iterrows():
        qid = str(q["query_id"])
        question = str(q["question"])
        gold_ids = set(_prepare_gold(q.get("gold_source_ids", "")))
        gold_label = str(q.get("gold_label", "")).lower() if pd.notna(q.get("gold_label", "")) else ""

        # retrieve (pre)
        ret = retriever.search(
            query_id=qid,
            question=question,
            gold_source_ids_str=str(q.get("gold_source_ids", "")),
            k=k_candidates,
        )
        pre_doc_ids = ret["doc_ids"]
        pre_scores = ret["scores"]
        pre_rank_first = ret["rank_first_rel_pre"]
        pre_recall_pk = ret[f"recall_at_{primary_k}"]

        cand_ids = pre_doc_ids
        cand_txts = [doc_text.get(did, "") for did in cand_ids]

        # rerank (post) or pass-through
        post_doc_ids = cand_ids
        post_scores = pre_scores
        rank_first_rel_post = 0
        recall_post = 0
        ctx_ids = cand_ids[:m_context]
        ctx_txts = cand_txts[:m_context]

        if use_reranker and reranker is not None and cand_ids:
            rr = reranker.rerank(
                query_id=qid,
                question=question,
                candidate_doc_ids=cand_ids,
                candidate_texts=cand_txts,
                gold_source_ids_str=str(q.get("gold_source_ids", "")),
            )
            post_doc_ids = rr["doc_ids_reranked"]
            post_scores = rr["scores_reranked"]
            rank_first_rel_post = rr["rank_first_rel_post"]
            recall_post = rr[f"recall_at_{primary_k}_post"]
            ctx_ids = rr["context_doc_ids_reranked"]
            ctx_txts = rr["context_texts_reranked"]

        # generate
        gen_out = generator.answer(qid, question, ctx_ids, ctx_txts)

        # metrics (ranking from post if rerank else pre)
        ranked_for_scores = post_doc_ids if use_reranker else pre_doc_ids
        # MRR@k
        mrr = 0.0
        for i, did in enumerate(ranked_for_scores[:mrr_k], start=1):
            if did in gold_ids:
                mrr = 1.0 / i
                break
        # nDCG@k (binary rel, ideal=1 at rank 1)
        ndcg = 0.0
        for i, did in enumerate(ranked_for_scores[:ndcg_k], start=1):
            if did in gold_ids:
                import math
                ndcg = 1.0 / math.log2(i + 1)
                break

        pred_label = str(gen_out.get("pred_label", "")).lower()
        citations = gen_out.get("citations", [])
        correct = int(pred_label == gold_label) if gold_label else 0

        row = {
            "variant": name,
            "query_id": qid,
            "gold_label": gold_label,
            "pred_label": pred_label,
            "is_correct": correct,

            # retrieval (pre)
            "rank_first_rel_pre": pre_rank_first,
            f"recall_at_{primary_k}_pre": pre_recall_pk,
            "doc_ids_pre": "|".join(pre_doc_ids),
            "scores_pre": "|".join([f"{s:.6f}" for s in pre_scores]),

            # rerank (post)
            "rank_first_rel_post": rank_first_rel_post,
            f"recall_at_{primary_k}_post": recall_post,
            "doc_ids_post": "|".join(post_doc_ids),
            "scores_post": "|".join([f"{s:.6f}" for s in post_scores]),

            # ranking metrics (variant-appropriate)
            f"mrr_at_{mrr_k}": mrr,
            f"ndcg_at_{ndcg_k}": ndcg,

            # generation
            "citations": "|".join(citations),
            "answer_text": gen_out.get("answer_text", ""),
            "latency_ms_total": int(gen_out.get("latency_ms_gen", 0)) + int(gen_out.get("latency_ms_critique", 0)),
            "latency_ms_gen": int(gen_out.get("latency_ms_gen", 0)),
            "latency_ms_critique": int(gen_out.get("latency_ms_critique", 0)),
            "critique_groundedness": gen_out.get("critique_groundedness", None),
            "critique_revised": bool(gen_out.get("critique_revised", False)),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if save_csv:
        ensure_dir(cfg.paths.eval_csv)
        # append if file exists and has header; else write fresh
        if os.path.exists(cfg.paths.eval_csv):
            df.to_csv(cfg.paths.eval_csv, mode="a", header=False, index=False)
        else:
            df.to_csv(cfg.paths.eval_csv, index=False)

    return df


if __name__ == "__main__":
    cfg = load_config()
    print_summary(cfg)

    # Load artifacts once
    queries = pd.read_parquet(cfg.paths.data_processed_queries)
    docs = pd.read_parquet(cfg.paths.data_processed_docs)

    # Run ALL variants declared in config, appending to outputs/eval.csv
    all_parts = []
    for variant in cfg.variants:
        print(f"[pipeline] running variant: {variant['name']}")
        part = run_pipeline(cfg, variant, queries, docs, save_csv=True)
        print(f"[pipeline] wrote {len(part)} rows for variant={variant['name']}")
        all_parts.append(part)

    # Print small summary to stdout
    if all_parts:
        full = pd.concat(all_parts, ignore_index=True)
        print(full.head(3).to_string(index=False))


# Run python3 -m app.pipeline
# Start: 5:40pm on Vast.ai

# Got back home around 732pm

