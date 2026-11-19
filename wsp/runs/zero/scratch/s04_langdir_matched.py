# Frequency-matched Swedish-vs-English direction in embedding space (the s03 direction was mostly "rare vs frequent").
import sys; sys.path.insert(0, "./wsp"); import harness as h
h.init("./wsp/runs/zero")
import numpy as np, collections
tok = h.tokenizer(); tr = h.pool("train")
E = h.weight("model.language_model.embed_tokens.weight"); V = E.shape[0]
mu = E.mean(0)
def counts(texts):
    c = collections.Counter()
    for t in texts: c.update(tok(t, add_special_tokens=False)["input_ids"])
    return c
dec = {}
def D(i):
    if i not in dec: dec[i] = tok.decode([i])
    return dec[i]
def wordlike(i):
    s = D(i).strip(); return len(s) >= 2 and s.isalpha()
rng = np.random.default_rng(0); perm = rng.permutation(len(tr)); halves = [perm[:100], perm[100:]]
def build(idx):
    csv = counts([tr[i]["question_sv"] for i in idx]); cen = counts([tr[i]["question_en"] for i in idx] + [tr[i]["solution_en"] for i in idx])
    sv = [t for t in csv if t not in cen and wordlike(t)]; en = [t for t in cen if t not in csv and wordlike(t)]
    return sv, en
bins = np.r_[0, 2000, 5000, 10000, 20000, 40000, 70000, 110000, 160000, 260000]
def matched_dir(sv, en):
    sv, en = np.array(sv), np.array(en); ds = []; ws = []
    for a, b in zip(bins[:-1], bins[1:]):
        s = sv[(sv >= a) & (sv < b)]; e = en[(en >= a) & (en < b)]
        if len(s) >= 5 and len(e) >= 5:
            ds.append(E[s].mean(0) - E[e].mean(0)); ws.append(min(len(s), len(e)))
    d = np.average(ds, axis=0, weights=ws); return d / np.linalg.norm(d)
(svA, enA), (svB, enB) = build(halves[0]), build(halves[1])
dA, dB = matched_dir(svA, enA), matched_dir(svB, enB)
print("cos(dA,dB) %.3f" % (dA @ dB))
svB2 = np.array([t for t in svB if t not in set(svA)]); enB2 = np.array([t for t in enB if t not in set(enA)])
aucs = []
for a, b in zip(bins[:-1], bins[1:]):
    s = svB2[(svB2 >= a) & (svB2 < b)]; e = enB2[(enB2 >= a) & (enB2 < b)]
    if len(s) >= 5 and len(e) >= 5:
        ps, pe = E[s] @ dA, E[e] @ dA; aucs.append((a, len(s), len(e), round(float(np.mean(ps[:, None] > pe[None, :])), 3)))
print("held-out within-bin AUC (bin_start, n_sv, n_en, auc):", aucs)
sv = list(set(svA) | set(svB)); en = list(set(enA) | set(enB))
d = matched_dir(sv, en)
np.save("./wsp/runs/zero/scratch/d_sv_en_matched.npy", d)
s03 = np.load("./wsp/runs/zero/scratch/d_lang_emb.npy")
print("cos with unmatched s03 direction %.3f" % (d @ s03))
proj = (E - mu) @ d
o = np.argsort(proj)
wl = [i for i in o[::-1][:20000] if wordlike(i)]
print("TOP Swedish side (word-like):", [D(i) for i in wl[:80]])
wl = [i for i in o[:20000] if wordlike(i)]
print("TOP English side (word-like):", [D(i) for i in wl[:80]])
# Does it separate Swedish from other languages (German/Danish/Norwegian/Dutch)? quick probe with hand-made word lists
probe = {"sv": " och att det som för är med till av på inte har jag kan om ett men var vi så från eller hur mycket många varje kvar".split(),
         "no/da": " ikke hvad hvordan mange hver kun efter også være blev".split(),
         "de": " und nicht das ist mit auch sich wie wird nach einer über wurde".split(),
         "nl": " het een niet zijn voor maar wordt hebben naar".split(),
         "en": " and not that is with also how many each after over been".split()}
for k, ws in probe.items():
    ids = [tok(" " + w, add_special_tokens=False)["input_ids"] for w in ws]; ids = [i[0] for i in ids if len(i) == 1]
    print(k, "n=%d mean proj %.4f" % (len(ids), ((E[ids] - mu) @ d).mean()))
print("vocab proj std %.4f" % proj.std())
