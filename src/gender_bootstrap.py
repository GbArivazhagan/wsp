"""Paired bootstrap over TEST nouns for the gender probe: AUC and accuracy differences (model - base).
python src/gender_bootstrap.py qwen35_gender_best_acts qwen35_gender_best_random [qwen35_gender_lora]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, load_jsonl  # noqa: E402
import hf_tools as H  # noqa: E402
from eval_logprob import auc  # noqa: E402


def scores(name):
    model, tok, _ = H.load(name)
    rows = load_jsonl(DATA / "gender" / "gender_test.jsonl")
    items = []
    for r in rows:
        items += [("", r["good"]), ("", r["bad"])]
    lp = H.cont_logprobs(model, tok, items, batch=64)
    good = np.array([lp[2 * i][0] for i in range(len(rows))])
    bad = np.array([lp[2 * i + 1][0] for i in range(len(rows))])
    neut = np.array([r["gender"] == "Neut" for r in rows])
    del model
    import torch
    torch.cuda.empty_cache()
    return np.where(neut, good - bad, bad - good), good > bad, neut, [r["noun"] for r in rows]


def main():
    base_s, base_c, neut, nouns = scores("qwen3.5-0.8b")
    uniq = sorted(set(nouns))
    idx = {n: [i for i, x in enumerate(nouns) if x == n] for n in uniq}
    rng = np.random.default_rng(0)
    out = {"n_nouns": len(uniq), "base": {"auc": auc(base_s[neut], base_s[~neut]), "acc": float(base_c.mean())}}
    for name in sys.argv[1:]:
        s, c, _, _ = scores(name)
        d_auc, d_acc = [], []
        for _ in range(2000):
            pick = np.concatenate([idx[uniq[j]] for j in rng.integers(0, len(uniq), len(uniq))])
            nb = neut[pick]
            d_auc.append(auc(s[pick][nb], s[pick][~nb]) - auc(base_s[pick][nb], base_s[pick][~nb]))
            d_acc.append(c[pick].mean() - base_c[pick].mean())
        out[name] = {"auc": auc(s[neut], s[~neut]), "acc": float(c.mean()),
                     "d_auc": float(np.mean(d_auc)), "d_auc_ci95": [float(np.percentile(d_auc, 2.5)), float(np.percentile(d_auc, 97.5))],
                     "d_acc": float(np.mean(d_acc)), "d_acc_ci95": [float(np.percentile(d_acc, 2.5)), float(np.percentile(d_acc, 97.5))]}
        print(name, json.dumps(out[name]), flush=True)
    (RESULTS / "gender" / "bootstrap.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
