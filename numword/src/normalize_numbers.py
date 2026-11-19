"""Swedish cardinal number words -> digits (0-9999), and the NUM/NONUM split of the Swedish GSM8K sets.

Whole words only; standalone 'en'/'ett' are left alone (they are also articles). Compounds are handled
('tjugosju' -> 27, 'tvåhundrafemtio' -> 250, 'tretusen' -> 3000). Ordinals and fractions ('tredje', 'hälften',
'dubbelt') are not cardinals and are not changed.

  python numword/src/normalize_numbers.py      # writes data files under numword/data and prints the split sizes
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UNITS = {"noll": 0, "en": 1, "ett": 1, "två": 2, "tre": 3, "fyra": 4, "fem": 5, "sex": 6, "sju": 7, "åtta": 8, "nio": 9,
         "tio": 10, "elva": 11, "tolv": 12, "tretton": 13, "fjorton": 14, "femton": 15, "sexton": 16, "sjutton": 17,
         "arton": 18, "nitton": 19}
TENS = {"tjugo": 20, "trettio": 30, "fyrtio": 40, "femtio": 50, "sextio": 60, "sjuttio": 70, "åttio": 80, "nittio": 90}


def _below100():
    d = {k: v for k, v in UNITS.items()}
    for t, tv in TENS.items():
        d[t] = tv
        for u, uv in UNITS.items():
            if 1 <= uv <= 9:
                d[t + u] = tv + uv
    return d


def build():
    b100 = _below100()
    words = dict(b100)
    hundreds = {"hundra": 100, "etthundra": 100, "enhundra": 100}
    for u, uv in UNITS.items():
        if 2 <= uv <= 9:
            hundreds[u + "hundra"] = 100 * uv
    b1000 = dict(b100)
    for h, hv in hundreds.items():
        b1000[h] = hv
        for r, rv in b100.items():
            if rv >= 1 and r not in ("en",):
                b1000[h + r] = hv + rv
    words.update(b1000)
    thousands = {"tusen": 1000, "ettusen": 1000, "etttusen": 1000, "entusen": 1000}
    for u, uv in UNITS.items():
        if 2 <= uv <= 9:
            thousands[u + "tusen"] = 1000 * uv
    for t, tv in thousands.items():
        words[t] = tv
        for r, rv in b1000.items():
            if rv >= 1 and r not in ("en",):
                words[t + r] = tv + rv
    return words


WORDS = build()
SKIP = {"en", "ett"}
TOKEN = re.compile(r"[A-Za-zÅÄÖåäöÉé]+")


def to_digits(text):
    n = 0

    def rep(m):
        nonlocal n
        w = m.group(0)
        lw = w.lower()
        if lw in SKIP or lw not in WORDS:
            return w
        n += 1
        return str(WORDS[lw])
    return TOKEN.sub(rep, text), n


def main():
    sys.path.insert(0, str(ROOT / "src"))
    from common import DATA, load_jsonl, save_jsonl
    out = ROOT / "numword" / "data"
    test = load_jsonl(DATA / "gsm8k" / "gsm8k_sv200.jsonl")
    split, dig = {}, []
    for r in test:
        q, n = to_digits(r["question_sv"])
        split[r["id"]] = "NUM" if n else "NONUM"
        dig.append({**r, "question_sv": q, "n_number_words": n})
    save_jsonl(dig, out / "gsm8k_sv200_digits.jsonl")
    pool = load_jsonl(DATA / "gsm8k" / "gsm8k_train_sv300.jsonl")
    psplit = {}
    for r in pool:
        _, n = to_digits(r["question_sv"])
        psplit[r["id"]] = "NUM" if n else "NONUM"
    json.dump({"test": split, "pool": psplit}, open(out / "num_split.json", "w"), indent=0)
    nt = sum(v == "NUM" for v in split.values())
    nd = sum(v == "NUM" and int(k.split("_")[1]) >= 200 for k, v in psplit.items())
    print(f"TEST: NUM {nt}, NONUM {len(split) - nt}; DEV pool (ids 200-299) NUM {nd}")
    ex = [(r["question_sv"], d["question_sv"]) for r, d in zip(test, dig) if d["n_number_words"]][:4]
    for a, b in ex:
        print("-", a[:140], "\n ", b[:140])


if __name__ == "__main__":
    main()
