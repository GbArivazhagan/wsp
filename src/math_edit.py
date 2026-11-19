"""Targeted (WSP / p50c-style) weight edits aimed at Qwen3.5-0.8B's Swedish math gap.

Directions are computed from the GSM8K TRAIN pool (ids 0-199 of gsm8k_train_sv300.jsonl) only:
  derail : the paper's recipe. u = mean residual over Swedish solutions that end WRONG minus those that end RIGHT
           (teacher-forced under the plain Swedish prompt, first --resp_tokens response tokens).
  rlang  : reasoning-language. u = Swedish-language solution minus English-language solution, both for the
           Swedish problem (the English one was generated with 'answer in English', then teacher-forced under the
           plain prompt). Suppressing u should push the model to reason in English.
  inalign: input alignment. u = Swedish problem text minus its English translation (mean over prompt tokens).
Column (neuron) selection follows p50c: per layer, the top-k down_proj columns by their contribution to the
contrast along u: |(mean act_j | A - mean act_j | B) * (u . W_col_j)|.

  python src/math_edit.py directions --kind derail --gen results/math/pool_sv_train.jsonl
  python src/math_edit.py build --kind derail --layers 15,19 --alpha 6 --topk 64 --out qwen35_m_derail_L15-19_a6
  python src/math_edit.py build --kind random --layers 15,19 --alpha 6 --topk 64 --match derail --out ...
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS, load_jsonl  # noqa: E402
import hf_tools as H  # noqa: E402
from grade_math import grade_numeric  # noqa: E402

DIRS = MODELS / "dirs"


@torch.no_grad()
def resid_and_acts(model, tok, items, resp_tokens, which="response", batch=4, max_len=1536):
    """Mean residual (all layers) and mean signed MLP hidden activation (all layers) over the scored span."""
    L = H.layers(model)
    nL = len(L)
    acts = {i: [] for i in range(nL)}
    spans_cur = {}
    hooks = []
    for i in range(nL):
        def pre(mod, inp, i=i):
            x = inp[0].float()
            m = spans_cur["mask"].to(x)              # [B, T]
            acts[i].append(((x * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)).cpu())
        hooks.append(L[i].mlp.down_proj.register_forward_pre_hook(pre))
    res = []
    tok.padding_side = "right"
    for s in range(0, len(items), batch):
        chunk = items[s:s + batch]
        ids, spans = [], []
        for pre_, cont in chunk:
            a = tok(pre_, add_special_tokens=False)["input_ids"]
            b = tok(cont, add_special_tokens=False)["input_ids"] if cont else []
            x = (a + b)[:max_len]
            ids.append(x)
            spans.append((0, len(a)) if which == "prompt" else (len(a), min(len(x), len(a) + resp_tokens)))
        T = max(map(len, ids))
        inp = torch.full((len(ids), T), tok.pad_token_id or 0, dtype=torch.long)
        att = torch.zeros_like(inp)
        mask = torch.zeros((len(ids), T))
        for j, x in enumerate(ids):
            inp[j, :len(x)] = torch.tensor(x)
            att[j, :len(x)] = 1
            mask[j, spans[j][0]:spans[j][1]] = 1
        spans_cur["mask"] = mask.to(H.DEV)
        hs = model(input_ids=inp.to(H.DEV), attention_mask=att.to(H.DEV), output_hidden_states=True).hidden_states
        Hs = torch.stack(hs, 0).float()               # [L+1, B, T, d]
        m = mask.to(H.DEV)
        res.append(((Hs * m[None, ..., None]).sum(2) / m.sum(1)[None, :, None].clamp(min=1)).cpu())
    for h in hooks:
        h.remove()
    R = torch.cat(res, 1)                              # [L+1, n, d]
    Aacts = {i: torch.cat(v, 0) for i, v in acts.items()}   # [n, d_ff]
    return R, Aacts


def cmd_directions(a):
    model, tok, _ = H.load(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if a.kind == "derail":
        rows = [r for r in load_jsonl(a.gen) if r["finish"] == "stop"]
        A = [(H.chat(tok, r["prompt"]), r["response"]) for r in rows if not grade_numeric(r["response"], r["answer"])]
        B = [(H.chat(tok, r["prompt"]), r["response"]) for r in rows if grade_numeric(r["response"], r["answer"])]
        which = "response"
    elif a.kind == "rlang":
        sv = {r["id"] + str(r["sample"]): r for r in load_jsonl(a.gen) if r["finish"] == "stop" and r["resp_lang"] == "sv"}
        en = {r["id"] + str(r["sample"]): r for r in load_jsonl(a.gen2) if r["finish"] == "stop" and r["resp_lang"] == "en"}
        keys = sorted(set(sv) & set(en))
        A = [(H.chat(tok, sv[k]["prompt"]), sv[k]["response"]) for k in keys]
        B = [(H.chat(tok, sv[k]["prompt"]), en[k]["response"]) for k in keys]
        which = "response"
    else:  # inalign
        sv = {r["id"].rsplit("_", 1)[0]: r for r in load_jsonl(a.gen) if r["sample"] == 0}
        en = {r["id"].rsplit("_", 1)[0]: r for r in load_jsonl(a.gen2) if r["sample"] == 0}
        keys = sorted(set(sv) & set(en))
        A = [(H.chat(tok, sv[k]["prompt"]), None) for k in keys]
        B = [(H.chat(tok, en[k]["prompt"]), None) for k in keys]
        which = "prompt"
    print(f"{a.kind}: |A|={len(A)} |B|={len(B)}", flush=True)
    RA, AA = resid_and_acts(model, tok, A, a.resp_tokens, which)
    RB, AB = resid_and_acts(model, tok, B, a.resp_tokens, which)
    nL = len(H.layers(model))
    out = {"kind": a.kind, "nA": len(A), "nB": len(B), "u": {}, "dact": {}, "dprime": {}, "probe_acc": {}}
    for i in range(nL):
        u = RA[i + 1].mean(0) - RB[i + 1].mean(0)
        u = u / u.norm()
        out["u"][i] = u
        out["dact"][i] = AA[i].mean(0) - AB[i].mean(0)
        pa, pb = RA[i + 1] @ u, RB[i + 1] @ u
        out["dprime"][i] = float((pa.mean() - pb.mean()) / torch.cat([pa, pb]).std())
        # held-out separability (2-fold) of the diff-in-means direction: an honest measure of 'is there a direction'
        n2a, n2b = len(pa) // 2, len(pb) // 2
        u1 = RA[i + 1][:n2a].mean(0) - RB[i + 1][:n2b].mean(0)
        thr = ((RA[i + 1][:n2a] @ u1).mean() + (RB[i + 1][:n2b] @ u1).mean()) / 2
        acc = (torch.cat([(RA[i + 1][n2a:] @ u1) > thr, (RB[i + 1][n2b:] @ u1) <= thr]).float().mean())
        out["probe_acc"][i] = float(acc)
    DIRS.mkdir(parents=True, exist_ok=True)
    torch.save(out, DIRS / f"{a.kind}.pt")
    print(json.dumps({"dprime": out["dprime"], "heldout_probe_acc": out["probe_acc"]}, indent=0))


def cmd_build(a):
    model, tok, _ = H.load(a.model)
    L = H.layers(model)
    lids = [int(x) for x in a.layers.split(",")]
    src = torch.load(DIRS / f"{a.match if a.kind == 'random' else a.kind}.pt")
    g = torch.Generator().manual_seed(a.seed)
    edited, info = [], {**vars(a), "columns": {}}
    for i in lids:
        W = L[i].mlp.down_proj.weight.data
        u = src["u"][i].to(H.DEV)
        score = ((src["dact"][i].to(H.DEV)) * (u @ W.float())).abs()
        cols = torch.topk(score, a.topk).indices.cpu() if a.topk else None
        if a.kind == "random":
            r = torch.randn(u.shape[0], generator=g).to(H.DEV)
            r = r - (r @ u) * u
            r = r / r.norm()
            # norm-matched: the same columns and the same Frobenius norm of the change as the targeted edit
            Wf = W.float()
            mask = torch.zeros(W.shape[1], device=H.DEV)
            mask[cols.to(H.DEV) if cols is not None else slice(None)] = 1
            d_t = a.alpha * torch.outer(u, (u @ Wf) * mask)
            d_r = torch.outer(r, (r @ Wf) * mask)
            d_r = d_r * (d_t.norm() / d_r.norm())
            W.copy_((Wf - d_r).to(W.dtype))
        else:
            H.rank1_edit_(W, u, a.alpha, cols)
        edited.append(H.param_name(model, L[i].mlp.down_proj))
        info["columns"][i] = None if cols is None else cols.tolist()
    info["changed_weights"] = len(lids) * W.shape[0] * (a.topk or W.shape[1])
    print(H.export_ckpt(model, a.model, a.out, edited, info))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["directions", "build"])
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--kind", choices=["derail", "rlang", "inalign", "random"], required=True)
    ap.add_argument("--gen", default="")
    ap.add_argument("--gen2", default="")
    ap.add_argument("--resp_tokens", type=int, default=512)
    ap.add_argument("--layers", default="15,19")
    ap.add_argument("--alpha", type=float, default=6.0)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--match", default="derail")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    {"directions": cmd_directions, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
