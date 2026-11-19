"""numword study: pre-registered choice of edit B's strength on DEV-NUM (answers >= 95 % Swedish, then highest
accuracy, ties -> smaller alpha). Writes numword/results/edits/B_selected.json and symlinks models/nw_B_sel,
models/nw_Btwin_sel.   python numword/src/select_B.py"""
import json, os, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import per_problem  # noqa: E402
rows = []
for a in (1, 2, 4):
    acc, sv, _ = per_problem(ROOT / "numword" / "results" / "dev" / f"nw_B_a{a}_devNUM.jsonl")
    rows.append({"alpha": a, "acc": float(np.mean(list(acc.values()))), "sv": sv})
ok = [r for r in rows if r["sv"] >= 0.95]
best = sorted(ok, key=lambda r: (-round(r["acc"], 6), r["alpha"]))[0]
bf = ROOT / "numword" / "results" / "dev" / "base_devNUM.jsonl"   # reference only, not used by the rule
base_dev = float(np.mean(list(per_problem(bf)[0].values()))) if bf.exists() else None
json.dump({"candidates": rows, "selected": best, "base_dev": base_dev},
          open(ROOT / "numword" / "results" / "edits" / "B_selected.json", "w"), indent=1)
for k in ("B", "Btwin"):
    dst = ROOT / "models" / f"nw_{k}_sel"
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(f"nw_{k}_a{best['alpha']}", dst)
print(rows, "->", best)
