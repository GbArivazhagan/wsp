# Swedish-vs-English direction from WORD-INITIAL tokens only (leading space), frequency-matched, to remove the
# "sub-word fragment vs whole word" confound found in s04.
import sys; sys.path.insert(0, "./wsp"); import harness as h
h.init("./wsp/runs/zero")
import numpy as np, collections
tok = h.tokenizer(); tr = h.pool("train")
E = h.weight("model.language_model.embed_tokens.weight"); mu = E.mean(0)
dec = {}
def D(i):
    if i not in dec: dec[i] = tok.decode([i])
    return dec[i]
def wi(i):
    s = D(i); return s.startswith(" ") and len(s.strip()) >= 2 and s.strip().isalpha()
def counts(texts):
    c = collections.Counter()
    for t in texts: c.update(tok(t, add_special_tokens=False)["input_ids"])
    return c
bins = np.r_[0, 2000, 5000, 10000, 20000, 40000, 70000, 110000, 160000, 260000]
def build(idx):
    csv = counts([tr[i]["question_sv"] for i in idx]); cen = counts([tr[i]["question_en"] for i in idx] + [tr[i]["solution_en"] for i in idx])
    return [t for t in csv if t not in cen and wi(t)], [t for t in cen if t not in csv and wi(t)]
def matched_dir(sv, en):
    sv, en = np.array(sv), np.array(en); ds = []; ws = []
    for a, b in zip(bins[:-1], bins[1:]):
        s = sv[(sv >= a) & (sv < b)]; e = en[(en >= a) & (en < b)]
        if len(s) >= 3 and len(e) >= 3: ds.append(E[s].mean(0) - E[e].mean(0)); ws.append(min(len(s), len(e)))
    d = np.average(ds, axis=0, weights=ws); return d / np.linalg.norm(d)
rng = np.random.default_rng(1); perm = rng.permutation(len(tr))
(svA, enA), (svB, enB) = build(perm[:100]), build(perm[100:])
print("n word-initial sv/en types:", len(set(svA)|set(svB)), len(set(enA)|set(enB)))
dA, dB = matched_dir(svA, enA), matched_dir(svB, enB); print("cos(dA,dB) %.3f" % (dA @ dB))
svB2 = np.array([t for t in svB if t not in set(svA)]); enB2 = np.array([t for t in enB if t not in set(enA)])
ps, pe = E[svB2] @ dA, E[enB2] @ dA; print("held-out AUC (all bins pooled) %.3f n=%d/%d" % (np.mean(ps[:, None] > pe[None, :]), len(ps), len(pe)))
d = matched_dir(list(set(svA)|set(svB)), list(set(enA)|set(enB)))
np.save("./wsp/runs/zero/scratch/d_sv_en_wordinit.npy", d)
for f in ["d_lang_emb", "d_sv_en_matched"]:
    print("cos with", f, "%.3f" % (d @ np.load(f"./wsp/runs/zero/scratch/{f}.npy")))
proj = (E - mu) @ d; o = np.argsort(proj)
top = [i for i in o[::-1][:30000] if wi(i)][:100]; bot = [i for i in o[:30000] if wi(i)][:60]
print("Swedish side:", [D(i) for i in top])
print("English side:", [D(i) for i in bot])
# other-language probes (single-token words)
probe = {"sv": "och att det som för är med till av på inte har kan om ett men var vi så från eller hur mycket många varje kvar".split(),
         "da/no": "ikke hvad hvordan mange hver kun efter også være blev".split(),
         "de": "und nicht das ist mit auch sich wie wird nach einer über wurde".split(),
         "nl": "het een niet zijn voor maar wordt hebben naar".split(),
         "fi": "ja että ei mutta kun myös".split(), "es": "que los las del para por pero como más".split(),
         "en": "and not that is with also how many each after over been".split()}
for k, ws in probe.items():
    ids = [tok(" " + w, add_special_tokens=False)["input_ids"] for w in ws]; ids = [i[0] for i in ids if len(i) == 1]
    print("%-6s n=%2d mean proj %+.4f (vocab std %.4f)" % (k, len(ids), proj[ids].mean(), proj.std()))
# fraction of Swedish-vs-English token variance this one direction explains
sv = list(set(svA)|set(svB)); en = list(set(enA)|set(enB))
diff = E[sv].mean(0) - E[en].mean(0); print("share of raw mean-diff along d: %.3f" % ((diff @ d)**2 / (diff @ diff)))
