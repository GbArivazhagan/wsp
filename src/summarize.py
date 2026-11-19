"""Summaries -> markdown tables (stdout) and results/*.csv.

python src/summarize.py scan | expD | lang | gender
"""
import collections
import glob
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, load_jsonl  # noqa: E402


def se(p, n):
    return math.sqrt(max(p * (1 - p), 1e-9) / max(n, 1))


def lang_table(files, title):
    print(f"\n### {title}\n")
    print("| run | prompts | n | sv | en | da | nb+nn | other/empty | mean tokens | cut |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    rows_out = []
    for f in files:
        rows = load_jsonl(f)
        by = collections.defaultdict(list)
        for r in rows:
            by[r["lang"]].append(r)
        for l, rs in sorted(by.items(), key=lambda x: ["sv", "en", "da", "nb"].index(x[0]) if x[0] in "sv en da nb" else 9):
            c = collections.Counter(r["resp_lang"] for r in rs)
            n = len(rs)
            sh = {k: c[k] / n for k in c}
            other = 1 - sum(sh.get(k, 0) for k in ("sv", "en", "da", "nb", "nn"))
            mt = sum(r["n_tokens"] for r in rs) / n
            cut = sum(r["finish"] == "length" for r in rs) / n
            print(f"| {Path(f).stem} | {l} | {n} | {sh.get('sv', 0):.2f} | {sh.get('en', 0):.2f} | {sh.get('da', 0):.2f} | "
                  f"{sh.get('nb', 0) + sh.get('nn', 0):.2f} | {other:.2f} | {mt:.0f} | {cut:.2f} |")
            rows_out.append({"run": Path(f).stem, "prompts": l, "n": n, **{k: round(v, 4) for k, v in sh.items()},
                             "mean_tokens": round(mt, 1), "cut": round(cut, 4)})
    return rows_out


def scan():
    files = sorted(glob.glob(str(RESULTS / "scan" / "*.jsonl")))
    out = lang_table(files, "Response language by model (DEV prompts, thinking off, T=0.7, 2 samples)")
    Path(RESULTS / "scan" / "summary.json").write_text(json.dumps(out, indent=1))


def expD():
    from grade_math import grade_math, grade_numeric
    from langid import label
    print("\n### Exp D: paper's p50c edit on Swedish vs English GSM8K (same 200 problems) and MATH-500\n")
    print("| run | set | n | acc | ±SE | cut share | acc on finished | answer lang sv/en | think lang sv/en | mean tokens |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    out = []
    for f in sorted(glob.glob(str(RESULTS / "expD" / "*.jsonl"))):
        rows = load_jsonl(f)
        by = collections.defaultdict(list)
        for r in rows:
            by[r["lang"] if r["category"] == "gsm8k" else "math500"].append(r)
        for k, rs in by.items():
            corr = [grade_math(r["response"], r["answer"]) if k == "math500" else grade_numeric(r["response"], r["answer"]) for r in rs]
            n = len(rs)
            cut = [r["finish"] == "length" for r in rs]
            fin = [c for c, x in zip(corr, cut) if not x]
            al = collections.Counter(r["resp_lang"] for r in rs if r["finish"] != "length")
            nf = max(1, sum(al.values()))
            tl = collections.Counter(label(r["response"].split("</think>")[0][:3000]) for r in rs if "</think>" in r["response"] and len(r["response"].split("</think>")[0]) > 50)
            nt = max(1, sum(tl.values()))
            acc = sum(corr) / n
            row = {"run": Path(f).stem, "set": k, "n": n, "acc": acc, "se": se(acc, n), "cut": sum(cut) / n,
                   "acc_finished": sum(fin) / max(1, len(fin)), "ans_sv": al["sv"] / nf, "ans_en": al["en"] / nf,
                   "think_sv": tl["sv"] / nt, "think_en": tl["en"] / nt, "think_n": sum(tl.values()),
                   "mean_tokens": sum(r["n_tokens"] for r in rs) / n}
            out.append(row)
            print(f"| {row['run']} | {k} | {n} | {acc:.3f} | {row['se']:.3f} | {row['cut']:.2f} | {row['acc_finished']:.3f} | "
                  f"{row['ans_sv']:.2f}/{row['ans_en']:.2f} | {row['think_sv']:.2f}/{row['think_en']:.2f} (n={row['think_n']}) | {row['mean_tokens']:.0f} |")
    Path(RESULTS / "expD" / "summary.json").write_text(json.dumps(out, indent=1))


def lang():
    files = sorted(glob.glob(str(RESULTS / "lang" / "*.jsonl")))
    out = lang_table(files, "Language-fidelity edits (TEST prompts)")
    Path(RESULTS / "lang" / "summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__" and sys.argv[1] != "paired":
    {"scan": scan, "expD": expD, "lang": lang}[sys.argv[1]]()


def per_problem_acc(path, lang=None):
    from grade_math import grade_math, grade_numeric
    rows = load_jsonl(path)
    acc = collections.defaultdict(list)
    for r in rows:
        if lang and r["lang"] != lang:
            continue
        key = r["id"].replace("gsm_sv_", "").replace("gsm_en_", "")
        ok = grade_math(r["response"], r["answer"]) if r["category"] == "math500" else grade_numeric(r["response"], r["answer"])
        acc[key].append(ok)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def paired(a, b, lang=None, n_boot=5000, seed=0):
    """Paired difference (b - a) in per-problem accuracy, with a bootstrap 95% CI over problems."""
    import random
    A, B = per_problem_acc(a, lang), per_problem_acc(b, lang)
    keys = sorted(set(A) & set(B))
    d = [B[k] - A[k] for k in keys]
    rng = random.Random(seed)
    boots = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(n_boot))
    return {"n": len(keys), "acc_a": sum(A[k] for k in keys) / len(keys), "acc_b": sum(B[k] for k in keys) / len(keys),
            "diff": sum(d) / len(d), "ci95": (boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)])}


if __name__ == "__main__" and sys.argv[1] == "paired":
    print(json.dumps(paired(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None), indent=0))
