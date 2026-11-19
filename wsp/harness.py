"""Harness for an LLM 'weight programmer' (mirrors the paper's Exp 8 harness, scaled down).

The editor gets:
  * read access to Qwen3.5-0.8B's text-model weights (CPU, float32 numpy) and its tokenizer;
  * save_edit(name, array) to replace a tensor (embeddings, the final norm and lm_head are refused, as in the paper);
  * a budget of FORWARD PASSES (0 for the weights-only condition). One pass is either
       generate(ids, which)      : the model (base / edited / both) answers up to 100 DEV problems (Swedish, 2 samples)
       activations(texts, layers): teacher-forced read of up to 256 texts: mean residual stream per layer and mean
                                   MLP activation per layer (base or edited weights)
  * the TRAIN/DEV problem pools (data/gsm8k/gsm8k_train_sv300.jsonl, ids 0-199 train, 200-299 dev). Never the test set.
Forbidden: gradients/training, running the model outside these calls. Every call is logged to <workdir>/harness_log.jsonl.

Usage inside the editor's Python:
    import sys; sys.path.insert(0, "./wsp"); import harness as h
    h.init("./wsp/runs/<run>")      # reads budget from <workdir>/budget.json
    W = h.weight("model.language_model.layers.15.mlp.down_proj.weight")   # float32 numpy, [out, in]
    h.save_edit(name, new_array)
    h.generate([200, 201, ...], which="both")   # costs 1 pass
"""
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SNAP = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")) + "/hub/models--Qwen--Qwen3.5-0.8B/snapshots/"
            "2fc06364715b967f1860aea9cf38778875588b17")
ST = SNAP / "model.safetensors-00001-of-00001.safetensors"
FROZEN = ("model.language_model.embed_tokens.weight", "model.language_model.norm.weight", "lm_head.weight")
_W = {"dir": None}


def _log(rec):
    rec["time"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(Path(_W["dir"]) / "harness_log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def init(workdir):
    _W["dir"] = str(workdir)
    Path(workdir, "edit").mkdir(parents=True, exist_ok=True)
    b = Path(workdir) / "budget.json"
    st = json.loads(b.read_text())
    print(f"harness ready: {st['passes_left']} forward passes left (of {st['passes_total']})")
    return st


def names():
    from safetensors import safe_open
    with safe_open(str(ST), "pt") as f:
        return [k for k in f.keys() if k.startswith("model.language_model.")]


def weight(name, edited=True):
    """float32 numpy copy of a tensor (the edited version if one was saved and edited=True)."""
    p = Path(_W["dir"]) / "edit" / f"{name}.npy"
    if edited and p.exists():
        return np.load(p)
    from safetensors import safe_open
    with safe_open(str(ST), "pt") as f:
        return f.get_tensor(name).float().numpy()


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(SNAP))


def save_edit(name, arr):
    assert name not in FROZEN, f"{name} may not be edited"
    base = weight(name, edited=False)
    arr = np.asarray(arr, dtype=np.float32)
    assert arr.shape == base.shape, (arr.shape, base.shape)
    assert np.isfinite(arr).all()
    np.save(Path(_W["dir"]) / "edit" / f"{name}.npy", arr)
    rel = float(np.linalg.norm(arr - base) / np.linalg.norm(base))
    _log({"call": "save_edit", "name": name, "rel_change": rel, "changed_entries": int((arr != base).sum())})
    print(f"saved {name}: relative change {rel:.4f}")


def reset_edit(name):
    p = Path(_W["dir"]) / "edit" / f"{name}.npy"
    if p.exists():
        p.unlink()
    _log({"call": "reset_edit", "name": name})


def _spend(kind, detail):
    b = Path(_W["dir"]) / "budget.json"
    st = json.loads(b.read_text())
    if st["passes_left"] <= 0:
        raise RuntimeError("forward-pass budget exhausted")
    st["passes_left"] -= 1
    b.write_text(json.dumps(st))
    _log({"call": kind, **detail, "passes_left": st["passes_left"]})
    return st["passes_left"]


def _build_ckpt(tag):
    """Write a vLLM/HF-loadable checkpoint with the current edits."""
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    from safetensors.torch import load_file, save_file
    out = ROOT / "models" / f"agent_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    for f in SNAP.iterdir():
        if f.name != ST.name and not (out / f.name).exists():
            os.symlink(os.path.realpath(f), out / f.name)
    t = load_file(str(ST))
    for p in (Path(_W["dir"]) / "edit").glob("*.npy"):
        t[p.stem] = torch.from_numpy(np.load(p)).to(torch.bfloat16)
    save_file(t, str(out / ST.name), metadata={"format": "pt"})
    return out


def generate(ids, which="both", n=2, max_tokens=2048):
    """DEV problems only (pool ids 200-299). Returns per-model accuracy and writes texts to <workdir>/gen_<k>.jsonl."""
    ids = [int(i) for i in ids]
    assert all(200 <= i < 300 for i in ids) and len(ids) <= 100, "generate() takes up to 100 DEV ids (200-299)"
    left = _spend("generate", {"ids": ids, "which": which})
    k = 0
    while (Path(_W["dir"]) / f"gen_{k}_base.jsonl").exists() or (Path(_W["dir"]) / f"gen_{k}_edited.jsonl").exists():
        k += 1
    res = {}
    models = {"base": "qwen3.5-0.8b", "edited": None}
    for w in (["base", "edited"] if which == "both" else [which]):
        m = models[w] or str(_build_ckpt(Path(_W["dir"]).name))
        out = Path(_W["dir"]) / f"gen_{k}_{w}.jsonl"
        idfile = Path(_W["dir"]) / f"gen_{k}_ids.json"
        idfile.write_text(json.dumps(ids))
        env = dict(os.environ, AGENT_IDS=str(idfile))
        cmd = [str(ROOT / "venvs/vllm/bin/python"), str(ROOT / "wsp/agent_gen.py"), m, str(idfile), str(out),
               str(n), str(max_tokens)]
        subprocess.run(cmd, check=True, env=env, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rows = [json.loads(l) for l in open(out)]
        res[w] = {"acc": float(np.mean([r["correct"] for r in rows])), "n": len(rows),
                  "cut": float(np.mean([r["finish"] == "length" for r in rows])),
                  "answer_in_swedish": float(np.mean([r["resp_lang"] == "sv" for r in rows])), "file": str(out)}
    _log({"call": "generate_result", **{w: {k2: v for k2, v in r.items() if k2 != 'file'} for w, r in res.items()}})
    print(json.dumps(res, indent=1), f"\n{left} passes left")
    return res


def activations(texts, layers, which="base", max_tokens=1024):
    """Teacher-forced read of up to 256 texts. Returns dict layer -> {'resid': [n, 1024] mean residual over tokens,
    'mlp': [n, 3584] mean MLP hidden activation}. Texts are raw strings (apply the chat template yourself)."""
    assert len(texts) <= 256
    left = _spend("activations", {"n_texts": len(texts), "layers": list(layers), "which": which})
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = str(SNAP) if which == "base" else str(_build_ckpt(Path(_W["dir"]).name))
    tok = AutoTokenizer.from_pretrained(path)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(dev).eval()
    L = model.model.layers
    out = {l: {"resid": [], "mlp": []} for l in layers}
    cur = {}
    hooks = [L[l].mlp.down_proj.register_forward_pre_hook(
        lambda m, i, l=l: cur.__setitem__(l, i[0][0].float().mean(0).cpu().numpy())) for l in layers]
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"][:, :max_tokens].to(dev)
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states
            for l in layers:
                out[l]["resid"].append(hs[l + 1][0].float().mean(0).cpu().numpy())
                out[l]["mlp"].append(cur[l])
    for h in hooks:
        h.remove()
    print(f"{left} passes left")
    return {l: {k: np.stack(v) for k, v in d.items()} for l, d in out.items()}


def pool(split="train"):
    """The Swedish/English problem pool with answers (train = ids 0-199, dev = 200-299)."""
    rows = [json.loads(l) for l in open(ROOT / "data/gsm8k/gsm8k_train_sv300.jsonl")]
    k = lambda r: int(r["id"].split("_")[1])
    return [r for r in rows if (k(r) < 200) == (split == "train")]
