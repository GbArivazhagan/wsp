"""Apply the pre-registered selection rule (report/hypothesis_and_design.md) to the DEV grid.
Writes results/mtest/selected.json {label: config} and prints the choice.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS  # noqa: E402

LABELS = {"derail": "edit: derail (paper's recipe)", "rlang": "edit: reasoning-language", "inalign": "edit: input alignment"}

rows = json.loads((RESULTS / "msweep" / "table.json").read_text())
base = next(r for r in rows if r["cfg"] == "msw_base")
sel = {}
for fam, lab in LABELS.items():
    cand = [r for r in rows if r["cfg"].split("_")[1] == fam and r["lang_ok_sv"] >= 0.95]
    allfam = [r for r in rows if r["cfg"].split("_")[1] == fam]
    if not cand:
        print(f"{fam}: no config keeps >=95% Swedish answers ({len(allfam)} configs)")
        continue
    best = max(cand, key=lambda r: (r["acc_sv"], r["acc_en"]))
    sel[lab] = best["cfg"]
    print(f"{fam}: {best['cfg']} DEV sv {best['acc_sv']:.3f} (base {base['acc_sv']:.3f}), en {best['acc_en']:.3f} "
          f"(base {base['acc_en']:.3f}); {len(cand)}/{len(allfam)} configs eligible")
(RESULTS / "mtest").mkdir(exist_ok=True)
(RESULTS / "mtest" / "selected.json").write_text(json.dumps(sel, indent=1))
