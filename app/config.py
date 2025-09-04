# app/config.py
import os
import sys
import json
from typing import Any
from dotenv import load_dotenv
load_dotenv()  # this will load .env into os.environ

# Used in industry for getting access to config 
# Replaced al the _get() and _() in previous old versions
from omegaconf import OmegaConf

try:
    import torch
except Exception:
    torch = None


REQUIRED_KEYS = [
    "paths.data_raw",
    "paths.data_processed_docs",
    "paths.data_processed_queries",
    "paths.vector_index",
    "paths.vector_meta",
    "paths.outputs_dir",
    "paths.eval_csv",
    "paths.metrics_summary_csv",
    "paths.plots_dir",
    "runtime.device",
    "models.embedding_model",
]


def _ensure_dirs(cfg: Any) -> None:
    """Create parent directories for all path outputs."""
    paths = [
        cfg.paths.data_processed_docs,
        cfg.paths.data_processed_queries,
        cfg.paths.vector_index,
        cfg.paths.vector_meta,
        cfg.paths.outputs_dir,
        cfg.paths.eval_csv,
        cfg.paths.metrics_summary_csv,
        cfg.paths.plots_dir,
    ]
    for path in paths:
        if not path:
            continue
        target_dir = path if os.path.splitext(path)[1] == "" else os.path.dirname(path)
        if target_dir and not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)


def _validate(cfg: Any) -> None:
    missing = []
    for key in REQUIRED_KEYS:
        try:
            val = OmegaConf.select(cfg, key)
        except Exception:
            val = None
        if val in (None, ""):
            missing.append(key)
    if missing:
        raise ValueError(f"[config] Missing required config keys: {missing}")


def _auto_device(cfg: Any) -> None:
    if cfg.runtime.device == "cuda":
        if not (torch and torch.cuda.is_available()):
            cfg.runtime.device = "cpu"
            print("[config] CUDA not available; falling back to CPU.", file=sys.stderr)


def load_config(path: str = "config.yaml", auto_device: bool = True):
    """Load YAML into OmegaConf DictConfig, validate, ensure dirs, auto-switch device if needed."""
    cfg = OmegaConf.load(path)

    if auto_device:
        _auto_device(cfg)

    _validate(cfg)
    _ensure_dirs(cfg)
    return cfg


# ---------------- Convenience helpers ----------------

def device(cfg) -> str:
    return cfg.runtime.device

def hf_headers():
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}

def generator_provider(cfg) -> str:
    return cfg.models.generator.provider

def tgi_base_url(cfg) -> str:
    return cfg.models.generator.tgi_http.base_url or ""

def self_critique_enabled(cfg) -> bool:
    return bool(cfg.self_critique.enabled)

def save_effective_config(cfg, out_path: str) -> None:
    with open(out_path, "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)

def print_summary(cfg) -> None:
    print(
        f"[config] device={cfg.runtime.device}, "
        f"embed={cfg.models.embedding_model}, "
        f"reranker={cfg.models.reranker_model}, "
        f"gen={cfg.models.generator.model}, "
        f"k={cfg.retrieval.k_candidates}, "
        f"m={cfg.retrieval.m_context}, "
        + (", refine=on" if cfg.self_critique.enabled else ", refine=off")
    )


# ---------------- Debug entrypoint ----------------
if __name__ == "__main__":
    cfg = load_config()
    print_summary(cfg)
    save_effective_config(cfg, "outputs/effective_config.json")
    print("[config] Effective config saved to outputs/effective_config.json")
