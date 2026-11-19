# Language direction in the (tied) embedding space, estimated from token statistics of the train pool.
import sys; sys.path.insert(0, "./wsp"); import harness as h
h.init("./wsp/runs/zero")
import numpy as np, collections
tok = h.tokenizer(); tr = h.pool("train")
E = h.weight("model.language_model.embed_tokens.weight")
gF = 1 + h.weight("model.language_model.norm.weight")
print("E norms: mean %.3f  median %.3f" % (np.linalg.norm(E, axis=1).mean(), np.median(np.linalg.norm(E, axis=1))))
print("final norm (1+g): mean %.3f min %.3f max %.3f" % (gF.mean(), gF.min(), gF.max()))
def counts(texts):
    c = collections.Counter()
    for t in texts: c.update(tok(t, add_special_tokens=False)["input_ids"])
    return c
rng = np.random.default_rng(0); perm = rng.permutation(len(tr)); A, B = perm[:100], perm[100:]
def sets(idx):
    csv = counts([tr[i]["question_sv"] for i in idx]); cen = counts([tr[i]["question_en"] for i in idx] + [tr[i]["solution_en"] for i in idx])
    sv_only = [t for t in csv if t not in cen]; en_only = [t for t in cen if t not in csv]
    return csv, cen, sv_only, en_only
csvA, cenA, svA, enA = sets(A); csvB, cenB, svB, enB = sets(B)
print("sv-only types A", len(svA), "en-only types A", len(enA))
mu = E.mean(0)
def dirn(sv, en):
    d = E[sv].mean(0) - E[en].mean(0); return d / np.linalg.norm(d)
dA = dirn(svA, enA); dB = dirn(svB, enB)
print("cos(dA,dB) = %.3f" % (dA @ dB))
# held-out separation: project B's sv-only / en-only tokens (excluding those in A's sets) on dA
svB2 = [t for t in svB if t not in set(svA)]; enB2 = [t for t in enB if t not in set(enA)]
ps, pe = E[svB2] @ dA, E[enB2] @ dA
auc = np.mean(ps[:, None] > pe[None, :])
print("held-out AUC on dA: %.3f  (n %d vs %d)" % (auc, len(svB2), len(enB2)))
# full direction
csv = csvA + csvB; cen = cenA + cenB
sv_only = [t for t in csv if t not in cen]; en_only = [t for t in cen if t not in csv]
d = dirn(sv_only, en_only)
np.save("./wsp/runs/zero/scratch/d_lang_emb.npy", d)
proj = (E - mu) @ d
o = np.argsort(proj)
print("TOP + (Swedish side):", [tok.decode([i]) for i in o[::-1][:60]])
print("TOP - (English side):", [tok.decode([i]) for i in o[:60]])
# variance explained of d in the embedding cloud
Ec = E[::7] - mu
print("std of projection over vocab %.4f vs avg per-dim std %.4f" % (np.std(Ec @ d), np.sqrt((Ec**2).mean())))
# what fraction of the sv-vs-en mean difference is captured by projecting
diff = E[sv_only].mean(0) - E[en_only].mean(0)
print("||mean diff|| %.4f ; typical token norm %.4f" % (np.linalg.norm(diff), np.linalg.norm(E[sv_only], axis=1).mean()))
# freq-weighted means (token-level)
def wmean(c):
    ids = np.array(list(c.keys())); w = np.array([c[i] for i in ids], float); return (E[ids] * w[:, None]).sum(0) / w.sum()
msv, men = wmean(csv), wmean(cen)
dt = (msv - men); print("token-level cos with type-level dir: %.3f, ||diff|| %.4f" % (dt @ d / np.linalg.norm(dt), np.linalg.norm(dt)))
# LDA-ish: a few principal directions of the sv_only vs en_only difference -- check rank of language info
X = np.concatenate([E[sv_only], E[en_only]]); y = np.r_[np.ones(len(sv_only)), np.zeros(len(en_only))]
Xc = X - X.mean(0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
for k in range(8):
    p = Xc @ Vt[k]; print("PC%d: var %.4f, corr with lang %.3f, cos with d %.3f" % (k, S[k]**2/len(X), np.corrcoef(p, y)[0,1], Vt[k] @ d))
