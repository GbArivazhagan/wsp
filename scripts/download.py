import os
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from huggingface_hub import snapshot_download
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from common import REGISTRY  # the exact revisions used in the study
models = {repo: rev for repo, rev in REGISTRY.values()}
for m,rev in models.items():
    try:
        p=snapshot_download(m, revision=rev, allow_patterns=["*.json","*.safetensors","*.model","*.txt","*.jinja","tokenizer*","*.py"])
        print("OK",m,p,flush=True)
    except Exception as e: print("FAIL",m,e,flush=True)
for lang in ["swe_Latn","eng_Latn","dan_Latn","nob_Latn"]:
    try:
        p=snapshot_download("facebook/belebele", repo_type="dataset", revision="7899cdfa4e1e0d733fd77c848e2c273cb1d32be2", allow_patterns=[f"data/{lang}*", f"{lang}*", "*.md"])
        print("OK belebele",lang,p,flush=True)
    except Exception as e: print("FAIL belebele",lang,e,flush=True)
