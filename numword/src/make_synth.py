"""Synthetic Swedish word problems with number words (numword study). Nothing comes from GSM8K.

Each item: problem with numbers written as Swedish words, the same problem with digits, the answer, and a short
Swedish solution that restates the numbers as digits ("sju" -> 7) and computes the answer.
SYN-train: templates T1-T8, names/objects set A. SYN-test: all templates T1-T10 (T9, T10 only here), names/objects set B.

  python numword/src/make_synth.py      # -> numword/data/syn_train.jsonl, syn_test.jsonl
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize_numbers import WORDS  # noqa: E402

NUM2W = {}
for w, v in WORDS.items():
    if w in ("en", "ett", "enhundra", "entusen", "ettusen", "etttusen", "etthundra"):
        continue
    NUM2W.setdefault(v, w)  # first spelling generated: the standard one
NAMES = {"A": ["Elsa", "Oskar", "Maja", "Hugo", "Alva", "Liam", "Ebba", "Noah", "Saga", "Viktor"],
         "B": ["Freja", "Axel", "Wilma", "Ludvig", "Agnes", "Theo", "Stina", "Melvin", "Tyra", "Sixten"]}
OBJ = {"A": ["äpplen", "pennor", "kulor", "böcker", "frimärken", "kakor", "stenar", "ballonger"],
       "B": ["päron", "suddgummin", "klistermärken", "tidningar", "muttrar", "bullar", "snäckor", "pussel"]}


def w(n, cap=False):
    s = NUM2W[n]
    return s.capitalize() if cap else s


def items(rng, names, objs, templates, k):
    out = []
    while len(out) < k:
        t = rng.choice(templates)
        N, M = rng.sample(names, 2)
        o = rng.choice(objs)
        a, b, c = rng.randint(2, 60), rng.randint(2, 30), rng.randint(2, 20)
        if t == 1:
            qw = f"{N} har {{a}} {o}. {N} får {{b}} till av {M}. Hur många {o} har {N} nu?"; ans = a + b
            sol = f"{N} har {a} {o} och får {b} till: {a} + {b} = {ans}."
        elif t == 2:
            a, b = rng.randint(2, 12), rng.randint(2, 12)
            qw = f"{N} köper {{a}} paket {o}. Varje paket innehåller {{b}} {o}. Hur många {o} köper {N} totalt?"; ans = a * b
            sol = f"{a} paket med {b} {o} i varje: {a} · {b} = {ans}."
        elif t == 3:
            a, b = max(a, b) + 5, min(a, b)
            qw = f"{N} hade {{a}} {o} men gav bort {{b}} till {M}. Hur många {o} har {N} kvar?"; ans = a - b
            sol = f"{N} hade {a} {o} och gav bort {b}: {a} − {b} = {ans}."
        elif t == 4:
            a, b = rng.randint(2, 12), rng.randint(2, 15)
            qw = f"Det finns {{a}} lådor med {{b}} {o} i varje. Hur många {o} finns det sammanlagt?"; ans = a * b
            sol = f"{a} lådor med {b} {o} i varje: {a} · {b} = {ans}."
        elif t == 5:
            a, b = rng.randint(5, 90), rng.randint(2, 12)
            qw = f"{N} sparar {{a}} kronor varje vecka i {{b}} veckor. Hur många kronor har {N} sparat?"; ans = a * b
            sol = f"{a} kronor per vecka i {b} veckor: {a} · {b} = {ans} kronor."
        elif t == 6:
            qw = f"{N} läser {{a}} sidor på måndagen och {{b}} sidor på tisdagen. Hur många sidor läser {N} totalt?"; ans = a + b
            sol = f"{a} sidor och {b} sidor: {a} + {b} = {ans}."
        elif t == 7:
            a, b = rng.randint(2, 9), rng.randint(2, 12)
            qw = "{C} mer än {a} gånger {b} är hur mycket?"; ans = c + a * b
            sol = f"{a} gånger {b} är {a * b}, och {c} mer än det är {a * b} + {c} = {ans}."
        elif t == 8:
            b, k2 = rng.randint(2, 9), rng.randint(2, 9); a = b * k2
            qw = f"{N} har {{a}} {o} och delar dem lika mellan {{b}} vänner. Hur många {o} får varje vän?"; ans = k2
            sol = f"{a} {o} delat på {b} vänner: {a} / {b} = {ans}."
        elif t == 9:
            a, b = rng.randint(5, 50), rng.randint(2, 30)
            qw = f"{N} är {{a}} år gammal. {M} är {{b}} år äldre än {N}. Hur gammal är {M}?"; ans = a + b
            sol = f"{M} är {a} + {b} = {ans} år."
        else:
            a, b = rng.randint(2, 9), rng.randint(2, 12); c = rng.randint(2, a * b - 1)
            qw = "{C} färre än {a} gånger {b} är hur mycket?"; ans = a * b - c
            sol = f"{a} gånger {b} är {a * b}, och {c} färre är {a * b} − {c} = {ans}."
        q_word = qw.format(a=w(a), b=w(b), C=w(c, cap=True))
        q_dig = qw.format(a=a, b=b, C=c)
        if q_word[0].islower():
            q_word = q_word[0].upper() + q_word[1:]
        out.append({"template": t, "q_word": q_word, "q_digit": q_dig, "answer": str(ans),
                    "solution_sv": sol + f"\n\nSvaret är \\boxed{{{ans}}}.",
                    "numbers": [x for x, ph in ((a, "{a}"), (b, "{b}"), (c, "{C}")) if ph in qw]})
    return out


def main():
    rng = random.Random(20261119)
    out = ROOT / "numword" / "data"
    tr = items(rng, NAMES["A"], OBJ["A"], list(range(1, 9)), 480)
    te = items(rng, NAMES["B"], OBJ["B"], list(range(1, 11)), 200)
    for name, rows in (("syn_train", tr), ("syn_test", te)):
        with open(out / f"{name}.jsonl", "w") as f:
            for i, r in enumerate(rows):
                f.write(json.dumps({"id": f"{name}_{i:03d}", **r}, ensure_ascii=False) + "\n")
    for r in tr[:3] + te[-2:]:
        print(r["q_word"], "|", r["q_digit"], "|", r["answer"])


if __name__ == "__main__":
    main()
