"""Log-probability evaluations (HF transformers), used for capability and regression checks.

  belebele_<lang> : 4-way reading comprehension (Belebele, CC BY-SA 4.0), zero-shot, the option with the highest
                    length-normalised log-prob wins. 900 questions per language. sv, en, da, nb.
  gender_<split>  : Swedish en/ett minimal pairs from UD Swedish-Talbanken (src/build_gender_pairs.py).
                    acc = P(good) > P(bad); auc = ROC-AUC of s = logP(neuter variant) - logP(common variant)
                    for neuter vs common nouns (insensitive to a global shift toward 'en' or 'ett').
  ppl_<lang>      : mean nats/token on the 488 unique Belebele passages (FLORES sentences), sv and en.

Usage: python src/eval_logprob.py --model qwen3.5-0.8b --out results/logprob/base.json
       [--steer_layers 12 --steer_vec models/vec.pt --steer_alpha 4]
"""
import os
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, load_jsonl  # noqa: E402
import hf_tools as H  # noqa: E402

BB = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")) + "/hub/datasets--facebook--belebele/snapshots/"
          "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2/data")
BBL = {"sv": "swe_Latn", "en": "eng_Latn", "da": "dan_Latn", "nb": "nob_Latn"}
QA = {"sv": ("Fråga", "Svar"), "en": ("Question", "Answer"), "da": ("Spørgsmål", "Svar"), "nb": ("Spørsmål", "Svar")}


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def belebele(model, tok, lang, limit=0):
    rows = load_jsonl(BB / f"{BBL[lang]}.jsonl")
    if limit:
        rows = rows[:limit]
    q, a = QA[lang]
    items, meta = [], []
    for r in rows:
        pre = f"{r['flores_passage']}\n{q}: {r['question']}\n{a}:"
        for k in range(1, 5):
            items.append((pre, " " + r[f"mc_answer{k}"]))
        meta.append(int(r["correct_answer_num"]))
    lp = H.cont_logprobs(model, tok, items, batch=16)
    correct = []
    for i, gold in enumerate(meta):
        s = [lp[4 * i + k][0] / max(lp[4 * i + k][1], 1) for k in range(4)]
        correct.append(int(np.argmax(s)) + 1 == gold)
    return float(np.mean(correct))


def gender(model, tok, split="test"):
    rows = load_jsonl(DATA / "gender" / f"gender_{split}.jsonl")
    items = []
    for r in rows:
        items += [("", r["good"]), ("", r["bad"])]
    lp = H.cont_logprobs(model, tok, items, batch=64)
    good = np.array([lp[2 * i][0] for i in range(len(rows))])
    bad = np.array([lp[2 * i + 1][0] for i in range(len(rows))])
    neut = np.array([r["gender"] == "Neut" for r in rows])
    s = np.where(neut, good - bad, bad - good)       # logP(ett-variant) - logP(en-variant)
    out = {"acc": float(np.mean(good > bad)), "auc": auc(s[neut], s[~neut]),
           "acc_com": float(np.mean((good > bad)[~neut])), "acc_neut": float(np.mean((good > bad)[neut])),
           "mean_ett_pref": float(s.mean())}
    # best achievable accuracy with a global bias shift (threshold sweep): separates calibration from knowledge
    thr = np.unique(s)
    best = max(np.mean((s > t) == neut) for t in np.concatenate([thr - 1e-6, [thr.max() + 1]]))
    out["acc_best_shift"] = float(best)
    by_t = {}
    for t in ("T1", "T2", "T3"):
        m = np.array([r["template"] == t for r in rows])
        by_t[t] = float(np.mean((good > bad)[m]))
    out["acc_by_template"] = by_t
    return out


def ppl(model, tok, lang):
    rows = load_jsonl(BB / f"{BBL[lang]}.jsonl")
    seen, texts = set(), []
    for r in rows:
        if r["link"] not in seen:
            seen.add(r["link"])
            texts.append(r["flores_passage"])
    lp = H.cont_logprobs(model, tok, [("", t) for t in texts], batch=16)
    return float(-sum(x[0] for x in lp) / sum(x[1] for x in lp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="belebele_sv,belebele_en,belebele_da,belebele_nb,gender_test,ppl_sv,ppl_en")
    ap.add_argument("--steer_layers", default="")
    ap.add_argument("--steer_vec", default="")
    ap.add_argument("--steer_alpha", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    model, tok, p = H.load(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    res = {"model": a.model, "model_path": p}
    ctx = None
    if a.steer_vec:
        v = torch.load(a.steer_vec).float()
        ctx = H.Steer(model, [int(x) for x in a.steer_layers.split(",")], a.steer_alpha * v.to(H.DEV))
        ctx.__enter__()
        res["steer"] = {"layers": a.steer_layers, "vec": a.steer_vec, "alpha": a.steer_alpha}
    t0 = time.time()
    for t in a.tasks.split(","):
        if t.startswith("belebele_"):
            res[t] = belebele(model, tok, t.split("_")[1], a.limit)
        elif t.startswith("gender_"):
            res[t] = gender(model, tok, t.split("_")[1])
        elif t.startswith("ppl_"):
            res[t] = ppl(model, tok, t.split("_")[1])
        print(t, res[t], f"{time.time() - t0:.0f}s", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
