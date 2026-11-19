"""Swedish-thinking study: accuracy, thinking language and paired bootstrap (sv-think vs en-think).
  venvs/vllm/bin/python swthink/analyze.py [--md swthink/results.md]
Thinking language: lingua label of the think text (prefill removed, math stripped by langid.clean), first 3000 chars.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grade_math import grade_numeric  # noqa: E402
from langid import label  # noqa: E402

R = ROOT / "swthink" / "results"
PF = ("Okej, jag löser det här steg för steg och tänker på svenska.", "Okay, I'll solve this step by step and think in English.")


def load(tag, lg, suffix=""):
    rows = []
    for s in (0, 1):
        f = R / f"{tag}_{lg}think_s{s}{suffix}.jsonl"
        if f.exists():
            rows += [dict(json.loads(l), seed=s) for l in open(f)]
    return rows


def summary(rows):
    acc = collections.defaultdict(list)
    tl, closed = [], []
    for r in rows:
        acc[r["id"]].append(grade_numeric(r["response"], r["answer"]))
        think, sep, _ = r["response"].partition("</think>")
        for p in PF:
            think = think.replace(p, "")
        tl.append(label(think[:3000]))
        closed.append(bool(sep) and not r.get("forced", False))
    return ({k: float(np.mean(v)) for k, v in acc.items()}, collections.Counter(tl), float(np.mean(closed)),
            float(np.mean([r.get("resp_lang") == "sv" for r in rows])), len({r["seed"] for r in rows}))


def boot(a, b, n=5000, seed=0):
    ks = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in ks])
    bs = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return 100 * d.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5), len(ks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    L = []
    L.append("| model / loop control | thinking | seeds | accuracy | think traces in Swedish | think closed by itself | answers in Swedish |")
    L.append("|---|---|---|---|---|---|---|")
    S = {}
    for tag, suf, name in (("p50", "", "p50c edit, 16k"), ("base", "_bf8192", "unedited + budget forcing 8k")):
        for lg in ("sv", "en"):
            rows = load(tag, lg, suf)
            if not rows:
                continue
            acc, tl, cl, sv, ns = summary(rows)
            S[(tag, lg)] = acc
            L.append(f"| {name} | {lg} | {ns} | {100 * np.mean(list(acc.values())):.1f} % | {100 * tl['sv'] / sum(tl.values()):.0f} % "
                     f"| {100 * cl:.0f} % | {100 * sv:.0f} % |")
    L.append("")
    for tag, name in (("p50", "p50c edit (primary)"), ("base", "unedited + budget forcing 8k")):
        if (tag, "sv") in S and (tag, "en") in S:
            d, lo, hi, n = boot(S[(tag, "sv")], S[(tag, "en")])
            verdict = "matches English (lower bound > −5)" if lo > -5 else ("worse than English" if hi < 0 else "inconclusive")
            L.append(f"- **{name}:** Δ(sv-think − en-think) = {d:+.1f} [{lo:+.1f}, {hi:+.1f}] points, n = {n} problems → {verdict}")
    txt = "\n".join(L)
    print(txt)
    if a.md:
        Path(a.md).write_text(txt + "\n")


if __name__ == "__main__":
    main()
