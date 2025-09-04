# app/generate.py
# Local-only generator (transformers). Runs entirely inside your Vast.ai container GPU.
# No HF serverless, no TGI HTTP. Keeps prompts/parsers minimal and aligned to your metrics.

import time
import re
import json
from typing import Dict, List, Any, Tuple, Optional

from app.config import load_config, print_summary, self_critique_enabled
from app.util import auto_device

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ----------------------------- prompt builders -----------------------------

def build_answer_prompt(question: str, ctx_ids: List[str], ctx_texts: List[str], label_space: List[str]) -> str:
    ctx_blocks = []
    for i, (did, txt) in enumerate(zip(ctx_ids, ctx_texts), start=1):
        ctx_blocks.append(f"[DOC {i} | id={did}]\n{txt}")
    ctx_str = "\n\n".join(ctx_blocks)
    labels_str = ", ".join(label_space)
    allowed_ids = ", ".join(ctx_ids)

    return (
        "You are a careful biomedical QA assistant. Use ONLY the context snippets.\n"
        f"Allowed labels: {labels_str}\n"
        "Cite exactly ONE supporting doc id from the list below.\n"
        f"Allowed doc_ids: {allowed_ids}\n\n"
        "STRICT RESPONSE FORMAT (single line):\n"
        "<label> [doc_id]\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{ctx_str}\n\n"
        "Answer:"
    )

def _normalize_citations(raw_cites: List[str], allowed_ids: List[str]) -> List[str]:
    # strip wrappers like "doc_id=..." and keep only allowed ids
    allowed = set(allowed_ids)
    out = []
    for c in raw_cites:
        c2 = c.replace("doc_id=", "").strip()
        if c2 in allowed:
            out.append(c2)
    # de-dup, keep order
    seen = set()
    final = []
    for c in out:
        if c not in seen:
            seen.add(c); final.append(c)
    return final


def build_critique_prompt(question: str, ctx_ids: List[str], ctx_texts: List[str], draft: str) -> str:
    ctx_blocks = []
    for i, (did, txt) in enumerate(zip(ctx_ids, ctx_texts), start=1):
        ctx_blocks.append(f"[DOC {i} | id={did}]\n{txt}")
    ctx_str = "\n\n".join(ctx_blocks)
    return (
        "You are a strict reviewer. Check if the draft answer is grounded in the context.\n"
        "Rate groundedness 1-5 (5 = fully grounded). If <3, provide a corrected answer in the same format.\n\n"
        "Format:\n"
        "Groundedness: <1-5>\n"
        "Revised: <label> [doc_id]   # include only if grounding < 3\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{ctx_str}\n\n"
        f"Draft Answer: {draft}\n\n"
        "Review:"
    )


# ----------------------------- small parsers -----------------------------

_LABEL_RE = re.compile(r"\b(yes|no|maybe)\b", re.IGNORECASE)
_CITE_RE = re.compile(r"\[([^\]]+)\]")

def parse_label_and_cites(text: str, label_space: List[str]) -> Tuple[str, List[str]]:
    label = ""
    m = _LABEL_RE.search(text or "")
    if m:
        cand = m.group(1).lower()
        if cand in {x.lower() for x in label_space}:
            label = cand
    cites: List[str] = []
    m2 = _CITE_RE.search(text or "")
    if m2:
        raw = m2.group(1)
        cites = [c.strip() for c in re.split(r"[,\s]+", raw) if c.strip()]
    return label, cites

def parse_critique(text: str) -> Tuple[int, Optional[str]]:
    g = 5
    m = re.search(r"Groundedness:\s*([1-5])", text or "", re.IGNORECASE)
    if m:
        g = int(m.group(1))
    revised = None
    m2 = re.search(r"Revised:\s*(.+)", text or "", re.IGNORECASE)
    if m2:
        revised = m2.group(1).strip()
    return g, revised


# ----------------------------- local generator -----------------------------

class LocalGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.models.generator.model
        self.label_space = list(cfg.eval.label_space)

        # decoding params
        self.max_new_tokens = int(cfg.models.generator.get("max_new_tokens", 64))
        self.temperature = float(cfg.models.generator.get("temperature", 0.2))
        self.top_p = float(cfg.models.generator.get("top_p", 0.9))
        self.repetition_penalty = float(cfg.models.generator.get("repetition_penalty", 1.05))

        # self-critique controls
        self.sc_enabled = self_critique_enabled(cfg)
        self.sc_passes = int(cfg.self_critique.get("passes", 1)) if self.sc_enabled else 0
        self.sc_mode = cfg.self_critique.get("mode", "revise_if_ungrounded") if self.sc_enabled else "off"
        self.sc_threshold = int(cfg.self_critique.get("groundedness_threshold", 3)) if self.sc_enabled else 0

        print(f"[generate] loading local model: {self.model_name}")
        self.device = auto_device()  # 'cuda' on Vast.ai, else 'mps'/'cpu'
        # device_map="auto" will place weights on GPU when available
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")

        # sdpa attention if backend supports it (safe to leave implicit)

    def _generate_text(self, prompt: str) -> str:
        toks = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **toks,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=False if self.temperature == 0 else True,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # If the model echoes the prompt, strip it
        return text[len(prompt):].strip() if text.startswith(prompt) else text

    def answer(
        self,
        query_id: str,
        question: str,
        context_doc_ids_reranked: List[str],
        context_texts_reranked: List[str],
    ) -> Dict[str, Any]:
        # 1) first pass
        # Needed for Variant 1 and 2
        prompt = build_answer_prompt(question, context_doc_ids_reranked, context_texts_reranked, self.label_space)
        t0 = time.time()
        text = self._generate_text(prompt)
        latency_gen = int((time.time() - t0) * 1000)
        pred_label, citations = parse_label_and_cites(text, self.label_space)
        citations = _normalize_citations(citations, context_doc_ids_reranked)
        if not citations:
            # fallback: force-cite top-1 context
            citations = [context_doc_ids_reranked[0]]


        # 2) self-critique
        # Needed for Variant 3 rag+rerank+refine
        g_score = None
        revised_flag= False
        latency_crit = 0
        if self.sc_enabled and self.sc_passes > 0:
            c_prompt = build_critique_prompt(question, context_doc_ids_reranked, context_texts_reranked, text)
            t1 = time.time()
            c_text = self._generate_text(c_prompt)
            latency_crit = int((time.time() - t1) * 1000)
            g_score, revised = parse_critique(c_text)
            if self.sc_mode == "revise_if_ungrounded" and g_score < self.sc_threshold and revised:
                text = revised
                pred_label, citations = parse_label_and_cites(text, self.label_space)
                revised_flag = True

        return {
            "query_id": query_id,
            "answer_text": text.strip(),
            "pred_label": pred_label,
            "citations": citations,
            "latency_ms_gen": latency_gen,
            "latency_ms_critique": latency_crit if self.sc_enabled else 0,
            "critique_groundedness": g_score if self.sc_enabled else None,
            "critique_revised": revised_flag if self.sc_enabled else False,
        }


# ----------------------------- debug entrypoint -----------------------------

if __name__ == "__main__":
    cfg = load_config()
    print_summary(cfg)

    # Tiny smoke test with the first query and 4 chunks from the same pmid
    import pandas as pd
    queries = pd.read_parquet(cfg.paths.data_processed_queries)
    docs = pd.read_parquet(cfg.paths.data_processed_docs)
    row = queries.iloc[0]
    qid, question = str(row["query_id"]), str(row["question"])
    pmid = qid
    subset = docs[docs["pmid"] == pmid].head(4)
    ctx_ids = subset["doc_id"].tolist()
    ctx_txt = subset["text"].tolist()

    gen = LocalGenerator(cfg)
    out = gen.answer(qid, question, ctx_ids, ctx_txt)
    print(json.dumps(out, indent=2))

# Run Python3 -m app.generate
# works in VastAi...refer to that for any bugs