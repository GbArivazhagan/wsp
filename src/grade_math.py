"""Grade GSM8K-style (numeric) and MATH-500 answers from generations; summarise loops / truncation / language.

python src/grade_math.py results/gsm/*.jsonl
"""
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_jsonl  # noqa: E402


def last_boxed(text: str):
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
        k += 1
    return None


def num(s: str):
    if s is None:
        return None
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:mathrm|textbf|mbox)\{([^}]*)\}", r"\1", s)
    s = s.replace("\\$", "").replace("$", "").replace("\\%", "").replace("%", "").replace("\\,", "").replace("\\!", "")
    s = re.sub(r"(?i)\b(kr|kronor|dollar|dollars|usd|sek|st|stycken|minuter|timmar|cent|euro)\b", "", s)
    s = s.strip().rstrip(".")
    m = re.findall(r"-?\d[\d\s.,]*", s)
    if not m:
        return None
    t = m[-1].strip().rstrip(".,")
    if re.fullmatch(r"-?\d{1,3}([ ,.]\d{3})+", t) and not re.fullmatch(r"-?\d+[.,]\d{1,2}", t):
        t = re.sub(r"[ ,.]", "", t)
    t = t.replace(" ", "").replace(",", ".")
    if t.count(".") > 1:
        t = t.replace(".", "")
    try:
        return float(t)
    except ValueError:
        return None


def grade_numeric(resp: str, gold: str):
    pred = num(last_boxed(resp.split("</think>")[-1]) or last_boxed(resp))
    g = float(gold.replace(",", ""))
    return pred is not None and abs(pred - g) < 1e-6


def grade_math(resp: str, gold: str):
    # math-verify is required: a string-equality fallback would silently under-count (39.4 % vs 45.6 % on MATH-500).
    from math_verify import parse, verify
    b = last_boxed(resp.split("</think>")[-1]) or last_boxed(resp)
    if b is None:
        return False
    try:
        return bool(verify(parse("$" + gold + "$"), parse("$" + b + "$")))
    except Exception:
        return False


def summarise(path):
    rows = load_jsonl(path)
    from langid import label
    out = collections.OrderedDict()
    n = len(rows)
    corr = []
    for r in rows:
        if r["category"] == "math500":
            corr.append(grade_math(r["response"], r["answer"]))
        else:
            corr.append(grade_numeric(r["response"], r["answer"]))
    cut = [r["finish"] == "length" for r in rows]
    closed = ["</think>" in r["response"] for r in rows]
    out["n"] = n
    out["acc"] = sum(corr) / n
    out["cut_share"] = sum(cut) / n
    out["acc_on_finished"] = (sum(c for c, k in zip(corr, cut) if not k) / max(1, sum(1 for k in cut if not k)))
    out["mean_tokens"] = sum(r["n_tokens"] for r in rows) / n
    langs = collections.Counter(r.get("resp_lang", label(r["response"])) for r in rows)
    out["answer_lang"] = {k: v / n for k, v in langs.most_common()}
    think_l = collections.Counter(label(r["response"].split("</think>")[0][:4000]) for r in rows if "</think>" in r["response"])
    out["think_lang"] = {k: v / max(1, sum(closed)) for k, v in think_l.most_common()}
    out["think_closed"] = sum(closed) / n
    return out, corr


if __name__ == "__main__":
    for p in sys.argv[1:]:
        s, _ = summarise(p)
        print(Path(p).stem, json.dumps(s, ensure_ascii=False))
