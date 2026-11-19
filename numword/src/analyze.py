"""numword study: accuracy tables with paired bootstrap CIs.  python numword/src/analyze.py [--md out.md]"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from grade_math import grade_numeric  # noqa: E402

T = ROOT / "numword" / "results" / "test"
S = ROOT / "numword" / "results" / "syn"
D = ROOT / "numword" / "results" / "dev"


def per_problem(path):
    acc, sv, tr = collections.defaultdict(list), [], []
    for l in open(path):
        r = json.loads(l)
        acc[r["id"]].append(grade_numeric(r["response"], r["answer"]))
        sv.append(r.get("resp_lang") == "sv")
        tr.append(r.get("finish") != "stop")
    return {k: float(np.mean(v)) for k, v in acc.items()}, float(np.mean(sv)), float(np.mean(tr))


def boot(a, b, n=5000, seed=0):
    ks = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in ks])
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return 100 * d.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def fmt(t):
    return "%+.1f [%+.1f, %+.1f]" % t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    lines = []
    out = lines.append
    tags = ["base", "digits", "B", "Btwin", "C", "Ctwin", "D", "C2", "D2"]
    for cell, cname in (("svsv", "Swedish problem, default (Swedish) reasoning"), ("sven", "Swedish problem, English answer prefill")):
        out(f"\n### {cname}\n")
        out("| condition | NUM acc | Δ NUM vs base [95% CI] | Δ NUM vs twin | NONUM acc | Δ NONUM vs base | sv answers | cut off |")
        out("|---|---|---|---|---|---|---|---|")
        R = {}
        for t in tags:
            fn, fo = T / f"{t}_{cell}_NUM.jsonl", T / f"{t}_{cell}_NONUM.jsonl"
            if not fn.exists():
                continue
            R[t] = {"NUM": per_problem(fn)}
            if fo.exists():
                R[t]["NONUM"] = per_problem(fo)
        if "base" not in R:
            continue
        bN = R["base"]["NUM"][0]
        bO = R["base"].get("NONUM", (None,))[0]
        for t, v in R.items():
            n_acc, n_sv, n_tr = v["NUM"]
            twin = {"B": "Btwin", "C": "Ctwin"}.get(t)
            dt = fmt(boot(n_acc, R[twin]["NUM"][0])) if twin in R else ""
            o = v.get("NONUM") or (R["base"]["NONUM"] if t == "digits" and "NONUM" in R["base"] else None)
            oacc = "%.1f" % (100 * np.mean(list(o[0].values()))) if o else ""
            do = fmt(boot(o[0], bO)) if o and bO and t not in ("base", "digits") else ""
            sv = n_sv if not v.get("NONUM") else (n_sv * len(n_acc) * 8 + v["NONUM"][1] * len(v["NONUM"][0]) * 4) / (len(n_acc) * 8 + len(v["NONUM"][0]) * 4)
            out(f"| {t} | {100 * np.mean(list(n_acc.values())):.1f} | {fmt(boot(n_acc, bN)) if t != 'base' else ''} | {dt} | {oacc} | {do} | {100 * sv:.0f} % | {100 * n_tr:.1f} % |")
    out("\n### Synthetic held-out problems (SYN-test, greedy)\n")
    out("| condition | word form | digit form | word, unseen templates T9-T10 |")
    out("|---|---|---|---|")
    for t in tags:
        fw, fd = S / f"{t}_word.jsonl", S / f"{t}_digit.jsonl"
        if not fw.exists():
            continue
        rw = [json.loads(l) for l in open(fw)]
        accw = np.mean([grade_numeric(r["response"], r["answer"]) for r in rw])
        unseen = [grade_numeric(r["response"], r["answer"]) for r in rw if r.get("template", 0) >= 9]
        accd = np.mean([grade_numeric(json.loads(l)["response"], json.loads(l)["answer"]) for l in open(fd)]) if fd.exists() else float("nan")
        out(f"| {t} | {100 * accw:.1f} | {100 * accd:.1f} | {100 * np.mean(unseen):.1f} (n={len(unseen)}) |")
    out("\n### Regression checks (log-prob tasks; English GSM8K TEST first 50 problems x 4 samples)\n")
    out("| condition | FLORES sv perplexity | Δ ppl | en/ett AUC | Belebele sv acc | English GSM8K acc | Δ English vs base [95% CI] |")
    out("|---|---|---|---|---|---|---|")
    L, E = ROOT / "numword" / "results" / "logprob", ROOT / "numword" / "results" / "en"
    lp = {"base": ROOT / "results" / "logprob" / "qwen35_base.json"}
    lp.update({t: L / f"{t}.json" for t in tags if t != "base"})
    eb = per_problem(E / "base.jsonl")[0] if (E / "base.jsonl").exists() else None
    p0 = json.load(open(lp["base"]))["ppl_sv"]
    for t, f in lp.items():
        if not f.exists():
            continue
        j = json.load(open(f))
        fe = E / f"{t}.jsonl"
        ea = per_problem(fe)[0] if fe.exists() else None
        eacc = "%.1f" % (100 * np.mean(list(ea.values()))) if ea else ""
        de = fmt(boot(ea, eb)) if ea and eb and t != "base" else ""
        dp = "%+.1f %%" % (100 * (j["ppl_sv"] / p0 - 1)) if t != "base" else ""
        out(f"| {t} | {j['ppl_sv']:.3f} | {dp} | {j['gender_test']['auc']:.3f} | {100 * j['belebele_sv']:.1f} | {eacc} | {de} |")
    if D.exists():
        out("\n### DEV-NUM (edit B strength selection, default cell, 4 samples)\n")
        out("| model | DEV-NUM acc | sv answers |")
        out("|---|---|---|")
        for f in sorted(D.glob("*_devNUM.jsonl")):
            acc, sv, _ = per_problem(f)
            out(f"| {f.name.replace('_devNUM.jsonl', '')} | {100 * np.mean(list(acc.values())):.1f} | {100 * sv:.0f} % |")
    text = "\n".join(lines)
    print(text)
    if a.md:
        Path(a.md).write_text(text + "\n")


if __name__ == "__main__":
    main()
