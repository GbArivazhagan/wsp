"""Capability probe: can a WSP-style targeted edit improve Swedish en/ett gender agreement on UNSEEN nouns?

Base Qwen3.5-0.8B is already optimally calibrated on this task (accuracy = best accuracy reachable by a global
en/ett bias shift), so a real gain requires better *discrimination* (AUC), i.e. per-noun knowledge.

Procedure (all choices made on DEV nouns; TEST nouns used once at the end):
  1. directions u_l from TRAIN nouns: 'acts' = residual diff-in-means at the last prefix token (after the
     article/adjective) between common- and neuter-article contexts; 'weights' = embedding row of 'en' minus 'ett'.
  2. grid over layer sets x alpha (alpha < 0 amplifies the article-gender signal, alpha > 0 suppresses it),
     edit form W <- W - alpha * outer(u, mask_k*(u@W)) on mlp.down_proj (top-64 columns, as in the paper).
     Each grid point is one DEV evaluation (= outcome feedback; counted).
  3. best DEV config -> TEST; plus the same config with a random direction, the weights-only direction,
     few-shot prompting (inference-time alternative) and the base model.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, load_jsonl  # noqa: E402
import hf_tools as H  # noqa: E402
from eval_logprob import auc  # noqa: E402
from wsp_edit import contrast_items, direction_weights, mlp_act_means  # noqa: E402

FEWSHOT = "Exempel: ett hus, en bil, ett barn, en stol, ett äpple, en katt, ett bord, en gata.\n"


def gender_eval(model, tok, split, prefix=""):
    rows = load_jsonl(DATA / "gender" / f"gender_{split}.jsonl")
    items = []
    for r in rows:
        items += [(prefix, r["good"]), (prefix, r["bad"])]
    lp = H.cont_logprobs(model, tok, items, batch=64)
    good = np.array([lp[2 * i][0] for i in range(len(rows))])
    bad = np.array([lp[2 * i + 1][0] for i in range(len(rows))])
    neut = np.array([r["gender"] == "Neut" for r in rows])
    s = np.where(neut, good - bad, bad - good)
    thr = np.unique(s)
    best = max(np.mean((s > t) == neut) for t in np.concatenate([thr - 1e-6, [thr.max() + 1]]))
    return {"acc": float(np.mean(good > bad)), "auc": auc(s[neut], s[~neut]), "acc_best_shift": float(best),
            "acc_com": float(np.mean((good > bad)[~neut])), "acc_neut": float(np.mean((good > bad)[neut]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--out", default=str(RESULTS / "gender" / "sweep.json"))
    a = ap.parse_args()
    model, tok, _ = H.load(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    L = H.layers(model)
    nL = len(L)
    res = {"model": a.model, "topk": a.topk}
    t0 = time.time()
    res["base"] = {s: gender_eval(model, tok, s) for s in ("dev", "test")}
    res["fewshot"] = {s: gender_eval(model, tok, s, FEWSHOT) for s in ("dev", "test")}
    print("base", res["base"], "\nfewshot", res["fewshot"], f"{time.time() - t0:.0f}s", flush=True)

    # directions from TRAIN nouns
    A, B, which = contrast_items(tok, "gender", a.model, n_max=1200)
    RA = H.residual_means(model, tok, A, which=which)
    RB = H.residual_means(model, tok, B, which=which)
    u_acts = {}
    for i in range(nL):
        u = RA[i + 1].mean(0) - RB[i + 1].mean(0)
        u_acts[i] = (u / u.norm()).to(H.DEV)
    u_w, _ = direction_weights(model, tok, "gender")
    actw = mlp_act_means(model, tok, A + B, list(range(nL)), which)
    orig = {i: L[i].mlp.down_proj.weight.data.clone() for i in range(nL)}

    def apply(layer_set, alpha, src, seed=0):
        for i in range(nL):
            L[i].mlp.down_proj.weight.data.copy_(orig[i])
        for i in layer_set:
            if src == "acts":
                u = u_acts[i]
            elif src == "weights":
                u = u_w
            else:
                g = torch.Generator().manual_seed(seed + i)
                u = torch.randn(u_w.shape[0], generator=g).to(H.DEV)
                u = u / u.norm()
            W = L[i].mlp.down_proj.weight.data
            score = (u.float() @ W.float()).abs() * actw[i].to(H.DEV)
            cols = torch.topk(score, a.topk).indices.cpu() if a.topk else None
            H.rank1_edit_(W, u, alpha, cols)

    layer_sets = {"early": [4, 6], "mid": [10, 12], "p50c": [15, 19], "late": [20, 22],
                  "mid_late_8": [8, 10, 12, 14, 16, 18, 20, 22]}
    alphas = [-8, -4, -2, -1, 1, 2, 6]
    grid = []
    for name, ls in layer_sets.items():
        for al in alphas:
            apply(ls, al, "acts")
            r = gender_eval(model, tok, "dev")
            grid.append({"layers": name, "alpha": al, **r})
            print(name, al, r, f"{time.time() - t0:.0f}s", flush=True)
    res["grid_dev"] = grid
    res["n_feedback_evals"] = len(grid)
    best = max(grid, key=lambda g: (g["auc"], g["acc"]))
    res["best_dev"] = best
    ls = layer_sets[best["layers"]]
    out = {}
    for src in ("acts", "weights", "random"):
        apply(ls, best["alpha"], src)
        out[src] = {s: gender_eval(model, tok, s) for s in ("dev", "test")}
        print(src, out[src], flush=True)
    # weights-only with a fixed, a-priori choice (no feedback): paper's layers, amplify x2
    apply(layer_sets["p50c"], -2, "weights")
    out["weights_apriori_p50c_a-2"] = {s: gender_eval(model, tok, s) for s in ("dev", "test")}
    res["test"] = out
    for i in range(nL):
        L[i].mlp.down_proj.weight.data.copy_(orig[i])
    # export the selected acts edit for regression checks (ppl, belebele) with eval_logprob.py
    apply(ls, best["alpha"], "acts")
    edited = [H.param_name(model, L[i].mlp.down_proj) for i in ls]
    H.export_ckpt(model, a.model, "qwen35_gender_best_acts", edited, {"layers": ls, "alpha": best["alpha"]})
    apply(ls, best["alpha"], "random")
    H.export_ckpt(model, a.model, "qwen35_gender_best_random", edited, {"layers": ls, "alpha": best["alpha"]})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
