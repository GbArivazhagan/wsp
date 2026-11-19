# Cleaner language direction from hand-made Swedish/English TRANSLATION PAIRS (single tokens, leading space):
# the pair difference cancels meaning, leaving language identity. Then: which layers write / read it.
import sys; sys.path.insert(0, "./wsp"); import harness as h
h.init("./wsp/runs/zero")
import numpy as np
tok = h.tokenizer()
E = h.weight("model.language_model.embed_tokens.weight"); mu = E.mean(0)
pairs = """och and|att to|det it|som which|för for|är is|med with|till to|av of|på on|inte not|har has|kan can|om if|ett a|men but|var was|vi we|så so|från from|eller or|hur how|mycket much|många many|varje each|kvar left|antal number|totalt total|svar answer|steg step|hon she|han he|de they|hade had|får gets|fick got|mer more|än than|efter after|innan before|dag day|vecka week|år year|timme hour|pengar money|kostar costs|varje every|alla all|två two|tre three|fyra four|fem five|sex six|sju seven|åtta eight|nio nine|tio ten|hälften half|dubbelt double|gånger times|summa sum|lika equal|varje per|mindre less|större larger|första first|andra second|sista last|också also|bara only|sedan then|när when|där there|här here|vad what|vilken which|varför why|måste must|skulle would|kommer will|blir becomes|ger gives|tar takes|köper buys|säljer sells|betalar pays|tjänar earns|äpplen apples|bok book|bil car|hus house|barn children|skola school|vatten water|mat food|stor big|liten small|ny new|gammal old|vecka week|minuter minutes|procent percent|kronor dollars|resten rest|tillsammans together|varandra each""".split("|")
P = []
for p in pairs:
    a, b = p.split(); ia = tok(" " + a, add_special_tokens=False)["input_ids"]; ib = tok(" " + b, add_special_tokens=False)["input_ids"]
    if len(ia) == 1 and len(ib) == 1 and ia != ib: P.append((a, b, ia[0], ib[0]))
print("usable single-token pairs:", len(P), [p[0] for p in P])
diffs = np.stack([E[a] - E[b] for _, _, a, b in P])
rng = np.random.default_rng(0); cs = []
for r in range(20):
    m = rng.permutation(len(P)); A, B = m[:len(P)//2], m[len(P)//2:]
    dA = diffs[A].mean(0); dA /= np.linalg.norm(dA)
    acc = np.mean([(E[P[i][2]] - E[P[i][3]]) @ dA > 0 for i in B]); cs.append(acc)
print("held-out pair-sign accuracy %.3f" % np.mean(cs))
d = diffs.mean(0); print("||mean pair diff|| %.4f, mean ||pair diff|| %.4f  (coherence %.3f)" % (np.linalg.norm(d), np.linalg.norm(diffs, axis=1).mean(), np.linalg.norm(d)/np.linalg.norm(diffs, axis=1).mean()))
d /= np.linalg.norm(d); np.save("./wsp/runs/zero/scratch/d_pairs.npy", d)
proj = (E - mu) @ d
print("vocab proj std %.4f; pairs: sv mean %+.4f en mean %+.4f" % (proj.std(), proj[[p[2] for p in P]].mean(), proj[[p[3] for p in P]].mean()))
probe = {"da/no": "ikke hvad hvordan mange hver kun efter også være blev".split(),
         "de": "und nicht das ist mit auch sich wie wird nach einer über wurde".split(),
         "nl": "het een niet zijn voor maar wordt hebben naar".split(),
         "fi": "ja että ei mutta kun myös".split(), "es": "que los las del para por pero como más".split(),
         "fr": "les des est pas pour une dans sur avec".split()}
for k, ws in probe.items():
    ids = [tok(" " + w, add_special_tokens=False)["input_ids"] for w in ws]; ids = [i[0] for i in ids if len(i) == 1]
    print("%-6s n=%2d mean proj %+.4f" % (k, len(ids), proj[ids].mean()))
dec = lambda i: tok.decode([i])
o = np.argsort(proj)
print("TOP sv side:", [dec(i) for i in o[::-1][:80]])
print("TOP en side:", [dec(i) for i in o[:40]])
for f in ["d_lang_emb", "d_sv_en_matched", "d_sv_en_wordinit"]:
    print("cos with", f, "%+.3f" % (d @ np.load(f"./wsp/runs/zero/scratch/{f}.npy")))
