"""Controls for the paper's p50c edit (Experiment 8 of 'Sparks of Weight-Space Programming').

p50c: W = W0 - 6 * outer(u, mask64 * (u @ W0)) on mlp.down_proj of layers 15 and 19.
We recover u from the published diff (it is rank one) and build:
  rand   : the same 64 columns per layer, a random unit direction orthogonal to u, Frobenius norm of the change
           matched to p50c's change per layer (an untargeted edit of the same size and location)
Runs with plain numpy (uses st_numpy from the paper's released code for bfloat16 I/O).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data/wsp_logs/exp8_math/code"))
import st_numpy as S  # noqa: E402

SNAP = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")) + "/hub/models--Qwen--Qwen3.5-0.8B/snapshots/"
            "2fc06364715b967f1860aea9cf38778875588b17")
ST = "model.safetensors-00001-of-00001.safetensors"


def make_dir(name):
    out = ROOT / "models" / name
    out.mkdir(parents=True, exist_ok=True)
    for f in SNAP.iterdir():
        if f.name != ST:
            dst = out / f.name
            if not dst.exists():
                os.symlink(os.path.realpath(f), dst)
    return out


def main():
    base = S.load(str(SNAP / ST))
    p50 = S.load(str(ROOT / "models/qwen35_paper_p50c" / ST),
                 [f"model.language_model.layers.{l}.mlp.down_proj.weight" for l in (15, 19)])
    rng = np.random.default_rng(0)
    out_t = dict(base)
    info = {}
    for l in (15, 19):
        n = f"model.language_model.layers.{l}.mlp.down_proj.weight"
        W0 = S.bf16_to_f32(base[n][1]).astype(np.float64)
        W1 = S.bf16_to_f32(p50[n][1]).astype(np.float64)
        D = W1 - W0
        cols = np.where(np.abs(D).sum(0) > 0)[0]
        U, s, Vt = np.linalg.svd(D[:, cols], full_matrices=False)
        u = U[:, 0]
        # recover alpha: D[:,cols] ~= -alpha * outer(u, u @ W0[:,cols])
        proj = u @ W0[:, cols]
        alpha = -float((u @ D[:, cols]) @ proj / (proj @ proj))
        r = rng.standard_normal(W0.shape[0])
        r -= (r @ u) * u
        r /= np.linalg.norm(r)
        Dr = np.zeros_like(W0)
        Dr[:, cols] = -np.outer(r, r @ W0[:, cols])
        c = np.linalg.norm(D) / np.linalg.norm(Dr)
        Dr *= c
        out_t[n] = ("BF16", S.bf16_bits((W0 + Dr).astype(np.float32)).reshape(W0.shape))
        info[l] = {"n_cols": int(len(cols)), "sv_ratio_s2_s1": float(s[1] / s[0]), "alpha_recovered": alpha,
                   "fro_change_p50c": float(np.linalg.norm(D)), "fro_change_rand": float(np.linalg.norm(Dr)),
                   "rel_change": float(np.linalg.norm(D) / np.linalg.norm(W0)), "rand_scale": float(c)}
        np.save(ROOT / "models" / f"p50c_u_layer{l}.npy", u.astype(np.float32))
        np.save(ROOT / "models" / f"p50c_cols_layer{l}.npy", cols)
    out = make_dir("qwen35_paper_p50c_randctrl")
    S.save(str(out / ST), out_t, {"format": "pt"})
    (out / "EDIT_INFO.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
