"""numword study: norm-matched random twins of edit B (replaces the twins written by build_edit_B.py, which used the
same alpha but changed the weights ~3x less). Twin: same layers, same 64 columns, random unit direction, strength
scaled per layer so that ||dW_twin||_F = ||dW_B||_F.   python numword/src/build_Btwin_matched.py"""
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import hf_tools as H  # noqa: E402
d = torch.load(ROOT / "numword" / "results" / "edits" / "B_dirs.pt")
model, tok, _ = H.load("qwen3.5-0.8b", dtype=torch.float32)
L = H.layers(model)
lay = sorted(d["U"])
W0 = {l: L[l].mlp.down_proj.weight.detach().clone() for l in lay}
names = [H.param_name(model, L[l].mlp.down_proj) for l in lay]


def dW(l, u, alpha):
    W = W0[l].clone()
    H.rank1_edit_(W, u.to(W), alpha, cols=d["cols"][l])
    return W - W0[l]


for alpha in (1, 2, 4):
    rel = {}
    with torch.no_grad():
        for l in lay:
            nb = dW(l, d["U"][l], alpha).norm()
            nt1 = dW(l, d["R"][l], 1.0).norm()
            a_t = alpha * float(nb / nt1)            # dW is linear in alpha
            L[l].mlp.down_proj.weight.copy_(W0[l] + dW(l, d["R"][l], a_t))
            rel[l] = {"alpha_twin": a_t, "fro_B": float(nb), "fro_twin": float((L[l].mlp.down_proj.weight - W0[l]).norm())}
    out = H.export_ckpt(model.to(torch.bfloat16), "qwen3.5-0.8b", f"nw_Btwin_a{alpha}", names,
                        info={"kind": "Btwin (norm-matched)", "alpha_B": alpha, "per_layer": rel})
    model.to(torch.float32)
    print(alpha, rel, out, flush=True)
