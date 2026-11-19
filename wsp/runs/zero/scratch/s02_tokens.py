# Presence-penalty hypothesis: in the example Swedish outputs, which corrupted word forms appear, and had the
# correct continuation token already been emitted earlier in the output (=> penalised by presence_penalty)?
import sys; sys.path.insert(0, "./wsp"); import harness as h
h.init("./wsp/runs/zero")
import re, numpy as np
tok = h.tokenizer()
ctx = open("./wsp/runs/zero/context.md").read()
parts = re.split(r"\n## ", ctx)[1:]
for p in parts:
    head = p.split("\n")[0]
    ans = p.split("ANSWER:\n", 1)[1]
    prob = p.split("PROBLEM: ", 1)[1].split("ANSWER:")[0]
    ids = tok(ans, add_special_tokens=False)["input_ids"]
    pids = set(tok(prob, add_special_tokens=False)["input_ids"])
    print("==", head, "n_out_tokens", len(ids), "unique", len(set(ids)), "frac tokens that were already emitted:",
          round(np.mean([ids[i] in set(ids[:i]) for i in range(len(ids))]), 3))
    # how many digit tokens
    dig = [i for i in ids if tok.decode([i]).strip().isdigit()]
    print("   digit tokens", len(dig), "distinct", len(set(dig)))
    # words in the output not in the problem and that look like corrupted forms of problem words
    seen = []
    for i, t in enumerate(ids):
        s = tok.decode([t])
        if not s.startswith(" ") and s.isalpha() and i > 0:
            prev = ids[:i]
            seen.append((i, tok.decode(ids[max(0,i-2):i+1]), s, t in prev, t in pids))
    # print continuation pieces that are new in the output (not emitted before, not in prompt)
    new = [x for x in seen if not x[3] and not x[4]]
    print("   continuation pieces new to output & prompt:", [x[1] for x in new][:60])
