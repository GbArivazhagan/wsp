# Per-layer read/write gain of the translation-pair language direction d (vs random unit directions).
# read gain  = ||W diag(1+g_norm) d|| / mean_r ||W diag(1+g_norm) r||   (each input matrix separately)
# write gain = ||d^T W_out|| / mean_r ||r^T W_out||                       (each output matrix separately)
import sys; sys.path.insert(0, "./wsp"); import harness as h
h.init("./wsp/runs/zero")
import numpy as np
d = np.load("./wsp/runs/zero/scratch/d_pairs.npy")
gF = 1 + h.weight("model.language_model.norm.weight")
dout = gF * d; dout /= np.linalg.norm(dout)   # residual direction that raises Swedish-word over English-word logits (up to rms)
rng = np.random.default_rng(0); R = rng.standard_normal((1024, 64)); R /= np.linalg.norm(R, axis=0)
pre = "model.language_model.layers."
print("layer type | read gains (inputs) | write gains along d, along dout (outputs)")
for l in range(24):
    full = (l % 4 == 3)
    g1 = 1 + h.weight(pre + f"{l}.input_layernorm.weight"); g2 = 1 + h.weight(pre + f"{l}.post_attention_layernorm.weight")
    ins = ([f"{l}.self_attn.q_proj.weight", f"{l}.self_attn.k_proj.weight", f"{l}.self_attn.v_proj.weight"] if full else
           [f"{l}.linear_attn.in_proj_qkv.weight", f"{l}.linear_attn.in_proj_z.weight"])
    outs = [f"{l}.self_attn.o_proj.weight" if full else f"{l}.linear_attn.out_proj.weight", f"{l}.mlp.down_proj.weight"]
    rd = {}
    for n in ins:
        W = h.weight(pre + n) * g1[None, :]; rd[n.split(".")[-2]] = np.linalg.norm(W @ d) / np.linalg.norm(W @ R, axis=0).mean()
    for n in ["gate_proj", "up_proj"]:
        W = h.weight(pre + f"{l}.mlp.{n}.weight") * g2[None, :]; rd[n] = np.linalg.norm(W @ d) / np.linalg.norm(W @ R, axis=0).mean()
    wr = {}
    for n in outs:
        W = h.weight(pre + n); base = np.linalg.norm(R.T @ W, axis=1).mean()
        wr[n.split(".")[-2]] = (np.linalg.norm(d @ W) / base, np.linalg.norm(dout @ W) / base)
    print("L%02d %s | " % (l, "full" if full else "lin "), " ".join("%s %.2f" % kv for kv in rd.items()), "|",
          " ".join("%s %.2f/%.2f" % (k, *v) for k, v in wr.items()))
