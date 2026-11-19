"""numword study, edit B (the paper's recipe applied to number words) and its random twins.

Direction: residual stream at the last token of each number, written as a Swedish word vs as digits, on SYN-train
(prefix = chat user turn up to and including the number). Per layer: u = mean(word) - mean(digit), d' along u.
The two layers in 6..18 with the largest d' are edited (the paper edited two layers). Edit (p50c form, hf_tools):
W <- W - alpha * u (mask * u^T W) on the top-64 columns of that layer's mlp.down_proj by |u^T W|, alpha in {1, 2, 4}.
Twins: a random unit direction, same layers, same columns, same alpha.

  python numword/src/build_edit_B.py
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common import load_jsonl  # noqa: E402
import hf_tools as H  # noqa: E402

OUT = ROOT / "numword" / "results" / "edits"
OUT.mkdir(parents=True, exist_ok=True)


def prefixes(rows, form):
    """one prefix per number: user turn up to and including that number in the given form."""
    items = []
    for r in rows:
        q = r["q_" + form]
        if form == "word":
            # locate number words in order of appearance
            import re
            pos = [m.end() for m in re.finditer(r"[A-Za-zÅÄÖåäö]+", q) if m.group(0).lower() in WORDS and m.group(0).lower() not in ("en", "ett")]
        else:
            import re
            pos = [m.end() for m in re.finditer(r"\d+", q)]
        for p in pos:
            items.append(("<|im_start|>user\n" + q[:p], ""))
    return items


def main():
    torch.manual_seed(0)
    rows = load_jsonl(ROOT / "numword" / "data" / "syn_train.jsonl")
    model, tok, _ = H.load("qwen3.5-0.8b", dtype=torch.float32)
    Pw, Pd = prefixes(rows, "word"), prefixes(rows, "digit")
    assert len(Pw) == len(Pd), (len(Pw), len(Pd))
    Hw = H.residual_means(model, tok, Pw, which="last_prefix")   # [L+1, n, d]
    Hd = H.residual_means(model, tok, Pd, which="last_prefix")
    info = {"n_pairs": len(Pw), "layers": []}
    dprime = []
    for l in range(Hw.shape[0] - 1):
        a, b = Hw[l + 1], Hd[l + 1]
        u = a.mean(0) - b.mean(0)
        un = u / u.norm()
        pa, pb = a @ un, b @ un
        dp = float((pa.mean() - pb.mean()) / ((pa.var() + pb.var()) / 2).sqrt())
        dprime.append(dp)
        info["layers"].append({"layer": l, "dprime": dp, "diff_norm": float(u.norm())})
    cand = sorted(range(6, 19), key=lambda l: -dprime[l])[:2]
    info["edit_layers"] = sorted(cand)
    print("d' per layer:", [round(x, 2) for x in dprime], "-> layers", info["edit_layers"], flush=True)
    U = {l: (Hw[l + 1].mean(0) - Hd[l + 1].mean(0)) for l in info["edit_layers"]}
    U = {l: u / u.norm() for l, u in U.items()}
    L = H.layers(model)
    W0 = {l: L[l].mlp.down_proj.weight.detach().clone() for l in info["edit_layers"]}
    cols = {l: torch.topk((U[l].to(W0[l]) @ W0[l]).abs(), 64).indices.cpu() for l in info["edit_layers"]}
    g = torch.Generator().manual_seed(1)
    R = {l: torch.randn(U[l].shape[0], generator=g) for l in info["edit_layers"]}
    R = {l: r / r.norm() for l, r in R.items()}
    info["cos_u_random"] = {l: float(U[l] @ R[l]) for l in info["edit_layers"]}
    torch.save({"U": U, "R": R, "cols": cols, "dprime": dprime}, OUT / "B_dirs.pt")
    names = [H.param_name(model, L[l].mlp.down_proj) for l in info["edit_layers"]]
    for alpha in (1, 2, 4):
        for kind, D in (("B", U), ("Btwin", R)):
            with torch.no_grad():
                for l in info["edit_layers"]:
                    L[l].mlp.down_proj.weight.copy_(W0[l])
                    H.rank1_edit_(L[l].mlp.down_proj.weight, D[l].to(W0[l]), alpha, cols=cols[l])
            rel = {l: float((L[l].mlp.down_proj.weight - W0[l]).norm() / W0[l].norm()) for l in info["edit_layers"]}
            out = H.export_ckpt(model.to(torch.bfloat16), "qwen3.5-0.8b", f"nw_{kind}_a{alpha}", names,
                                info={"kind": kind, "alpha": alpha, "layers": info["edit_layers"], "rel_change": rel})
            model.to(torch.float32)
            print(kind, alpha, "rel change", rel, "->", out, flush=True)
    json.dump(info, open(OUT / "B_info.json", "w"), indent=1)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from normalize_numbers import WORDS  # noqa: E402
    main()
