"""Grid of WSP-style edits for the Swedish math gap: build checkpoints (HF, one process) and evaluate them on the
DEV pool (vLLM, one process per checkpoint). Every DEV evaluation is outcome feedback and is counted.

  python src/math_sweep.py build --grid grid1 --part 0/2     # builds models/msw_<cfg>/ for its share of the grid
  python src/math_sweep.py eval  --grid grid1 --part 0/2     # DEV: gsmtr_sv:dev n=2, gsmtr_en:dev n=1
  python src/math_sweep.py table --grid grid1
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS, RESULTS, load_jsonl  # noqa: E402

LAYER_SETS = {"L15-19": [15, 19], "L10-14": [10, 14], "L6-18x4": [6, 10, 14, 18], "L20-22": [20, 22]}
GRIDS = {
    "grid1": [(k, ls, a) for k in ("derail", "rlang", "inalign") for ls in ("L15-19", "L10-14", "L6-18x4")
              for a in (1.0, 2.0, 6.0)],
}
GRIDS["all"] = GRIDS["grid1"]
NT = ["--think", "off", "--temperature", "0.7", "--top_p", "0.8", "--top_k", "20", "--presence_penalty", "1.5",
      "--max_tokens", "2048"]


def cfg_name(c):
    return f"msw_{c[0]}_{c[1]}_a{c[2]:g}"


def my_part(grid, part):
    i, n = map(int, part.split("/"))
    return [c for j, c in enumerate(GRIDS[grid]) if j % n == i]


def build(a):
    import torch
    import hf_tools as H
    model, tok, _ = H.load("qwen3.5-0.8b")
    L = H.layers(model)
    orig = {i: L[i].mlp.down_proj.weight.data.clone() for i in range(len(L))}
    dirs = {}
    for c in my_part(a.grid, a.part):
        out = MODELS / cfg_name(c)
        if (out / "EDIT_INFO.json").exists() or (RESULTS / "msweep" / f"{cfg_name(c)}.jsonl").exists():
            continue
        kind, ls, alpha = c
        if kind not in dirs:
            dirs[kind] = torch.load(MODELS / "dirs" / f"{kind}.pt")
        for i in range(len(L)):
            L[i].mlp.down_proj.weight.data.copy_(orig[i])
        cols_info, edited = {}, []
        for i in LAYER_SETS[ls]:
            W = L[i].mlp.down_proj.weight.data
            u = dirs[kind]["u"][i].to(H.DEV)
            score = (dirs[kind]["dact"][i].to(H.DEV) * (u @ W.float())).abs()
            cols = torch.topk(score, 64).indices.cpu()
            H.rank1_edit_(W, u, alpha, cols)
            cols_info[i] = cols.tolist()
            edited.append(H.param_name(model, L[i].mlp.down_proj))
        H.export_ckpt(model, "qwen3.5-0.8b", cfg_name(c), edited,
                      {"kind": kind, "layers": LAYER_SETS[ls], "alpha": alpha, "topk": 64, "columns": cols_info})
        print("built", cfg_name(c), flush=True)


def evaluate(a):
    py = sys.executable
    for c in my_part(a.grid, a.part):
        name = cfg_name(c)
        out = RESULTS / "msweep" / f"{name}.jsonl"
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [py, "src/gen_vllm.py", "--model", name, "--sets", "gsmtr_sv:dev,gsmtr_en:dev", "--n", "2",
               "--gpu_mem", str(a.gpu_mem), "--out", str(out)] + NT
        subprocess.run(cmd, check=False)
        if out.exists() and not a.keep:
            ck = MODELS / name
            info = ck / "EDIT_INFO.json"
            if info.exists():
                shutil.copy(info, out.with_suffix(".edit.json"))
            shutil.rmtree(ck, ignore_errors=True)


def build_one(a):
    """Build one config (called in a subprocess by worker)."""
    import torch
    import hf_tools as H
    c = next(c for c in GRIDS["all"] if cfg_name(c) == a.cfg)
    kind, ls, alpha = c
    model, tok, _ = H.load("qwen3.5-0.8b")
    L = H.layers(model)
    d = torch.load(MODELS / "dirs" / f"{kind}.pt")
    cols_info, edited = {}, []
    for i in LAYER_SETS[ls]:
        W = L[i].mlp.down_proj.weight.data
        u = d["u"][i].to(H.DEV)
        score = (d["dact"][i].to(H.DEV) * (u @ W.float())).abs()
        cols = torch.topk(score, 64).indices.cpu()
        H.rank1_edit_(W, u, alpha, cols)
        cols_info[i] = cols.tolist()
        edited.append(H.param_name(model, L[i].mlp.down_proj))
    H.export_ckpt(model, "qwen3.5-0.8b", cfg_name(c), edited,
                  {"kind": kind, "layers": LAYER_SETS[ls], "alpha": alpha, "topk": 64, "columns": cols_info})


def worker(a):
    """Claim-based: take any config of the grid that is neither evaluated, built elsewhere, nor claimed."""
    import os
    hpy = str(Path(__file__).resolve().parents[1] / "venvs/hf/bin/python")
    locks = RESULTS / "msweep" / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    for c in GRIDS[a.grid]:
        name = cfg_name(c)
        out = RESULTS / "msweep" / f"{name}.jsonl"
        if out.exists() or (MODELS / name).exists() or not (MODELS / "dirs" / f"{c[0]}.pt").exists():
            continue
        try:
            os.mkdir(locks / name)
        except FileExistsError:
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        subprocess.run([hpy, "src/math_sweep.py", "build_one", "--cfg", name], env=env, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cmd = [sys.executable, "src/gen_vllm.py", "--model", name, "--sets", "gsmtr_sv:dev,gsmtr_en:dev", "--n", "2",
               "--gpu_mem", str(a.gpu_mem), "--out", str(out)] + NT
        subprocess.run(cmd, check=False)
        info = MODELS / name / "EDIT_INFO.json"
        if out.exists() and info.exists():
            shutil.copy(info, out.with_suffix(".edit.json"))
        if not a.keep:
            shutil.rmtree(MODELS / name, ignore_errors=True)


def table(a):
    from grade_math import grade_numeric
    rows = []
    for f in sorted((RESULTS / "msweep").glob("*.jsonl")):
        R = load_jsonl(f)
        res = {"cfg": f.stem}
        for l in ("sv", "en"):
            rs = [r for r in R if r["lang"] == l]
            res[f"acc_{l}"] = sum(grade_numeric(r["response"], r["answer"]) for r in rs) / max(1, len(rs))
            res[f"lang_ok_{l}"] = sum(r["resp_lang"] == l for r in rs) / max(1, len(rs))
            res[f"cut_{l}"] = sum(r["finish"] == "length" for r in rs) / max(1, len(rs))
        rows.append(res)
    rows.sort(key=lambda r: -r["acc_sv"])
    print("| config | sv acc | sv answers in sv | en acc | en answers in en | cut sv |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['cfg']} | {r['acc_sv']:.3f} | {r['lang_ok_sv']:.2f} | {r['acc_en']:.3f} | {r['lang_ok_en']:.2f} | {r['cut_sv']:.2f} |")
    (RESULTS / "msweep" / "table.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "eval", "table", "build_one", "worker"])
    ap.add_argument("--cfg", default="")
    ap.add_argument("--grid", default="grid1")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--gpu_mem", type=float, default=0.45)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    {"build": build, "eval": evaluate, "table": table, "build_one": build_one, "worker": worker}[a.cmd](a)
