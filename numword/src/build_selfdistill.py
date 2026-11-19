"""numword study (exploratory): SYN-train word-form prompts paired with the unedited model's own correct solutions of the
digit form. -> numword/data/syn_train_sd.jsonl.   python numword/src/build_selfdistill.py"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from grade_math import grade_numeric  # noqa: E402
syn = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "numword" / "data" / "syn_train.jsonl")}
out, per = [], {}
for l in open(ROOT / "numword" / "results" / "syn" / "selfsol_train_digit.jsonl"):
    r = json.loads(l)
    sid = r["id"].rsplit("_", 1)[0]
    if r.get("finish") == "stop" and r.get("resp_lang") == "sv" and grade_numeric(r["response"], r["answer"]):
        if per.get(sid, 0) < 1:
            per[sid] = 1
            out.append({"id": sid, "q_word": syn[sid]["q_word"], "solution_sv": r["response"]})
with open(ROOT / "numword" / "data" / "syn_train_sd.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(out), "self-distilled examples from", len(syn), "problems")
