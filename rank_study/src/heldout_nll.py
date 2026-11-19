"""Low-noise supplement to sampled accuracy: mean per-token NLL of held-out DEV solutions under a checkpoint.
Texts: the unedited model's own samples on the 100 DEV problems (rank_study/results/dev_base.jsonl), split into
correct and incorrect Swedish solutions (finished only, first 1024 tokens). A good 'Swedish math' edit should lower
NLL of correct solutions more than of incorrect ones (margin = NLL_incorrect - NLL_correct goes up).
python rank_study/src/heldout_nll.py --model rank_r1d15 --out r1d15
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common import load_jsonl  # noqa: E402
from grade_math import grade_numeric  # noqa: E402
import hf_tools as H  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
model, tok, _ = H.load(a.model, dtype=torch.bfloat16)
rows = [r for r in load_jsonl(ROOT / "rank_study" / "results" / "dev_base.jsonl") if r["finish"] == "stop" and r["resp_lang"] == "sv"]
res = {}
for name, sel in (("correct", True), ("incorrect", False)):
    items = [(H.chat(tok, r["prompt"]), r["response"]) for r in rows if grade_numeric(r["response"], r["answer"]) == sel]
    lp = H.cont_logprobs(model, tok, items, batch=8, max_len=1024)
    res[name] = {"n": len(items), "nll_per_tok": -sum(s for s, _ in lp) / sum(n for _, n in lp)}
res["margin"] = res["incorrect"]["nll_per_tok"] - res["correct"]["nll_per_tok"]
res["model"] = a.model
(ROOT / "rank_study" / "results" / "nll" / f"{a.out}.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res))
