"""Build Swedish en/ett grammatical-gender minimal pairs from UD Swedish-Talbanken (CC BY-SA 4.0).

For each noun lemma that occurs (in the treebank) only with one gender, as an indefinite singular nominative
form identical to its lemma, we make sentence pairs that differ only in agreement:
   T1  "Det här är {en|ett} {noun}."
   T2  "Hon pratade länge om {en ny|ett nytt} {noun} igår."
   T3  "Där borta står {en stor|ett stort} {noun}."
Lemmas are split (seeded, stratified by gender and frequency) into train / dev / test.
"""
import collections, json, random, re, sys
from pathlib import Path

UD = Path(__file__).resolve().parents[1] / "data" / "ud"
OUT = Path(__file__).resolve().parents[1] / "data" / "gender"
gender = collections.defaultdict(collections.Counter)
freq = collections.Counter()
for split in ["train", "dev", "test"]:
    for line in open(UD / f"sv_talbanken-ud-{split}.conllu", encoding="utf8"):
        if not line.strip() or line.startswith("#"):
            continue
        c = line.rstrip("\n").split("\t")
        if len(c) != 10 or not c[0].isdigit() or c[3] != "NOUN":
            continue
        form, lemma, feats = c[1], c[2], dict(kv.split("=") for kv in c[5].split("|") if "=" in kv)
        g = feats.get("Gender")
        if g not in ("Com", "Neut"):
            continue
        gender[lemma][g] += 1
        freq[lemma] += 1
        if feats.get("Number") == "Sing" and feats.get("Definite") == "Ind" and feats.get("Case") == "Nom":
            gender[lemma]["_indef_sing:" + form.lower()] += 1

rows = []
for lemma, cnt in gender.items():
    if not re.fullmatch(r"[a-zåäöé]{3,20}", lemma):
        continue
    gs = [g for g in ("Com", "Neut") if cnt[g] > 0]
    if len(gs) != 1:                       # ambiguous gender in the treebank -> drop
        continue
    if cnt["_indef_sing:" + lemma] == 0:   # need the bare indefinite singular form attested
        continue
    rows.append({"noun": lemma, "gender": gs[0], "freq": freq[lemma]})

rng = random.Random(0)
by_g = collections.defaultdict(list)
for r in rows:
    by_g[r["gender"]].append(r)
splits = {"train": [], "dev": [], "test": []}
for g, lst in by_g.items():
    lst.sort(key=lambda r: (-r["freq"], r["noun"]))
    # stratify by frequency: walk in frequency order, assign in shuffled blocks of 10 -> 6/1/3 train/dev/test
    for i in range(0, len(lst), 10):
        block = lst[i:i + 10]
        rng.shuffle(block)
        for j, r in enumerate(block):
            splits["train" if j < 6 else "dev" if j < 7 else "test"].append(r)

TEMPL = [
    ("T1", "Det här är {art} {noun}.", {"Com": "en", "Neut": "ett"}),
    ("T2", "Hon pratade länge om {art} {noun} igår.", {"Com": "en ny", "Neut": "ett nytt"}),
    ("T3", "Där borta står {art} {noun}.", {"Com": "en stor", "Neut": "ett stort"}),
]
for s, lst in splits.items():
    with open(OUT / f"gender_{s}.jsonl", "w", encoding="utf8") as f:
        for r in sorted(lst, key=lambda r: r["noun"]):
            other = "Neut" if r["gender"] == "Com" else "Com"
            for tid, t, arts in TEMPL:
                f.write(json.dumps({"noun": r["noun"], "gender": r["gender"], "freq": r["freq"], "template": tid,
                                    "good": t.format(art=arts[r["gender"]], noun=r["noun"]),
                                    "bad": t.format(art=arts[other], noun=r["noun"])}, ensure_ascii=False) + "\n")
    c = collections.Counter(r["gender"] for r in lst)
    print(s, len(lst), dict(c), "median freq", sorted(r["freq"] for r in lst)[len(lst)//2])
