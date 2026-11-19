"""numword study: random twin of edit C. dW = s1 * u v^T on the same matrix (layer-15 mlp.down_proj), with u, v random
unit vectors and s1 the (only) singular value of C's dW, so ||dW||_F matches C.   python numword/src/build_Ctwin.py"""
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import hf_tools as H  # noqa: E402
svd = torch.load(ROOT / "numword" / "results" / "train" / "nw_C.svd.pt")
model, tok, _ = H.load("qwen3.5-0.8b", dtype=torch.float32)
pn = svd["param"]
W = dict(model.named_parameters())[pn]
g = torch.Generator().manual_seed(2)
u = torch.randn(W.shape[0], generator=g); v = torch.randn(W.shape[1], generator=g)
dW = float(svd["S"][0]) * torch.outer(u / u.norm(), v / v.norm())
with torch.no_grad():
    W.add_(dW.to(W))
print("C twin: ||dW|| =", float(dW.norm()), "param", pn)
print(H.export_ckpt(model.to(torch.bfloat16), "qwen3.5-0.8b", "nw_Ctwin", [pn], info={"kind": "Ctwin", "fro": float(dW.norm())}))
