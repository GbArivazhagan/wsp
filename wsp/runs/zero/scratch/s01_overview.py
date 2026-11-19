import sys; sys.path.insert(0, "./wsp"); import harness as h
h.init("./wsp/runs/zero")
import numpy as np
from safetensors import safe_open
with safe_open(str(h.ST), "pt") as f:
    ks = [k for k in f.keys() if k.startswith("model.language_model.")]
    for k in ks:
        if ".layers." in k and not (".layers.0." in k or ".layers.3." in k): continue
        print(k, tuple(f.get_slice(k).get_shape()))
    print("n tensors", len(ks))
    other = [k for k in f.keys() if not k.startswith("model.language_model.") and not k.startswith("model.visual")]
    print("other keys:", other[:20])
tok = h.tokenizer()
tr = h.pool("train")
print("n train", len(tr), "keys", list(tr[0].keys()))
nsv = [len(tok(r["question_sv"])["input_ids"]) for r in tr]
nen = [len(tok(r["question_en"])["input_ids"]) for r in tr]
wsv = [len(r["question_sv"].split()) for r in tr]; wen = [len(r["question_en"].split()) for r in tr]
print("tokens sv/en", np.mean(nsv), np.mean(nen), "words", np.mean(wsv), np.mean(wen), "fertility", np.sum(nsv)/np.sum(wsv), np.sum(nen)/np.sum(wen))
for w in ["bananer", " bananer", " bananter", " klasar", " klassar", " högar", " högare", " apor", " apa", " bananas", " monkeys", " tyg", " kvadratfot", " återstående", " varje", " each", " Resonera", " steg", " för", " slutgiltiga", " svar", " inom"]:
    ids = tok(w, add_special_tokens=False)["input_ids"]
    print(repr(w), ids, [tok.decode([i]) for i in ids])
