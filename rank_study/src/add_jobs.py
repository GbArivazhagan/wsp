"""Append train / DEV / TEST jobs for one condition to rank_study/queue.txt.
python rank_study/src/add_jobs.py NAME --target down15 --r 1 --data rft --lr 2e-4 --epochs 3 [--seed 0] [--dev] [--test] [--after JOB]"""
import argparse
from pathlib import Path

RS = Path(__file__).resolve().parents[1]
POOL = "results/math/pool_sv_train.jsonl,rank_study/results/pool_sv_train_s1.jsonl"
GEN = ("$PY src/gen_vllm.py --model rank_{n} --think off --temperature 0.7 --top_p 0.8 --top_k 20 "
       "--presence_penalty 1.5 --n 4 --seed 0 --gpu_mem $GPU_MEM")
ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--target", required=True)
ap.add_argument("--r", type=int, required=True)
ap.add_argument("--data", default="rft")
ap.add_argument("--lr", default="2e-4")
ap.add_argument("--epochs", default="3")
ap.add_argument("--seed", default="0")
ap.add_argument("--dev", action="store_true")
ap.add_argument("--test", action="store_true")
ap.add_argument("--after", default="-", help="extra dependency for the training job")
a = ap.parse_args()
n = a.name
pool = {"rft": POOL, "gold": "-", "posctrl": "rank_study/results/posctrl_pool.jsonl"}[a.data]
dep = {"rft": "pool_s1", "gold": "-", "posctrl": "posctrl_pool"}[a.data]
deps = ",".join(x for x in (dep, a.after) if x != "-") or "-"
lines = [f"tr_{n}|{deps}|source scripts/env_hf.sh; $PY rank_study/src/lora_rank.py --data {a.data} --pool {pool} "
         f"--target {a.target} --r {a.r} --lr {a.lr} --epochs {a.epochs} --seed {a.seed} --out rank_{n}"]
if a.dev:
    lines.append(f"dev_{n}|tr_{n}|{GEN.format(n=n)} --sets gsmtr_sv:dev --max_tokens 2048 --out rank_study/results/dev_{n}.jsonl")
if a.test:
    lines.append(f"test_{n}|tr_{n}|{GEN.format(n=n)} --sets gsm_sv,gsm_en --max_tokens 4096 --out rank_study/results/test_{n}.jsonl")
with open(RS / "queue.txt", "a") as f:
    for l in lines:
        f.write(l + "\n")
print("\n".join(lines))
