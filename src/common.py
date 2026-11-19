"""Shared paths, model registry, prompt loading and seeds."""
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"          # edited checkpoints live here
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Exact revisions used in this study (Hugging Face commit SHAs).
REGISTRY = {
    "qwen3.5-0.8b": ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
    "qwen3.5-2b": ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"),
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    "qwen2.5-0.5b-it": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    "qwen2.5-1.5b-it": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
    "smollm2-1.7b-it": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "31b70e2e869a7173562077fd711b654946d38674"),
    "eurollm-1.7b-it": ("utter-project/EuroLLM-1.7B-Instruct", "a25c7fa65fc2a644e6270b8940dbe295b51da681"),
    "olmo2-1b-it": ("allenai/OLMo-2-0425-1B-Instruct", "48d788eca847d4d7548f375ad03d3c9312f6139e"),
}


def model_path(name: str) -> str:
    """Local snapshot dir for a registry name, or a path to an edited checkpoint."""
    if name in REGISTRY:
        from huggingface_hub import snapshot_download
        repo, rev = REGISTRY[name]
        return snapshot_download(repo, revision=rev if len(rev) == 40 else None, local_files_only=True)
    p = Path(name)
    if not p.is_absolute():
        p = MODELS / name
    assert p.exists(), p
    return str(p)


def load_jsonl(path):
    with open(path, encoding="utf8") as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def seed_all(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def prompt_split(lang: str, split: str):
    """Prompts are split once, by seed 0: for sv, 1/3 dev (method development) and 2/3 test (held out).
    For en/da/nb control prompts: 1/3 dev, 2/3 test as well."""
    rows = load_jsonl(DATA / "prompts" / f"{lang}_prompts.jsonl")
    rng = random.Random(1234)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n_dev = len(rows) // 3
    keep = set(idx[:n_dev]) if split == "dev" else set(idx[n_dev:]) if split == "test" else set(idx)
    return [r for i, r in enumerate(rows) if i in keep]
