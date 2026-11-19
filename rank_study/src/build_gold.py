"""Build rank_study/data/gold_sv_train.jsonl from the Swedish step-by-step solutions that Claude translated from
GSM8K solution_en (calculator annotations removed), one per TRAIN problem (ids 0-199). Adds the final boxed answer."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from grade_math import grade_numeric  # noqa: E402

D = ROOT / "rank_study" / "data"
sol = {}
for f in sorted(D.glob("gold_sv_part*.txt")):
    for block in f.read_text().split("### ")[1:]:
        k, body = block.split("\n", 1)
        sol[int(k)] = body.strip()
rows = [json.loads(l) for l in open(ROOT / "data" / "gsm8k" / "gsm8k_train_sv300.jsonl")][:200]
out, bad = [], []
for r in rows:
    k = int(r["id"].split("_")[1])
    s = sol[k] + f"\n\nSvaret är \\boxed{{{r['answer']}}}."
    if not grade_numeric(s, r["answer"]):
        bad.append(k)
    out.append({"id": r["id"], "question_sv": r["question_sv"], "solution_sv": s, "answer": r["answer"]})
assert len(out) == 200 and not bad, bad
with open(D / "gold_sv_train.jsonl", "w") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print(len(out), "gold solutions; mean chars", sum(len(o["solution_sv"]) for o in out) / len(out))
