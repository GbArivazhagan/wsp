"""Edit-rank study: LoRA of a given rank on a given set of matrices (gradient descent, full supervision),
then merge, export a vLLM checkpoint, and save ||dW|| + the SVD of dW for single-matrix edits.

Copied from src/lora_train.py and changed:
  --target down15    : rank-r LoRA on mlp.down_proj of layer 15 only (the paper's edit family, but optimal)
           alldown   : rank-r LoRA on every layer's mlp.down_proj
           alllinear : rank-r LoRA on every linear layer of the 24 decoder layers (attn/linear-attn/MLP)
  --data rft    : the model's own CORRECT Swedish TRAIN-pool solutions (all pool files given), <=2 per problem
         gold   : Swedish gold solutions of the TRAIN problems (rank_study/data/gold_sv_train.jsonl)
         posctrl: plain ENGLISH prompt -> the model's own SWEDISH answer (generated with a 'reason in Swedish' suffix)
  micro-batches with gradient accumulation (token-weighted loss), max_len 1536 (answers are not cut).
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common import DATA, RESULTS, load_jsonl  # noqa: E402
import hf_tools as H  # noqa: E402

RS = ROOT / "rank_study"
EN_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
SV_SUFFIX = "\nResonera steg för steg och skriv ditt slutgiltiga svar inom \\boxed{}."
NOT_VIS = r"^(?!.*(visual|mtp)).*"
TARGETS = {
    "down15": NOT_VIS + r"\.layers\.15\.mlp\.down_proj$",
    "alldown": NOT_VIS + r"\.layers\.\d+\.mlp\.down_proj$",
    "alllinear": NOT_VIS + r"\.layers\.\d+\.(mlp\.(down|up|gate)_proj|self_attn\.[qkvo]_proj|"
                           r"linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj))$",
}


def build_examples(a, tok):
    from grade_math import grade_numeric
    eos = tok.eos_token or ""
    if a.data == "rft":
        per = {}
        for f in a.pool.split(","):
            for r in load_jsonl(f):
                if r["split"] != "train":
                    continue
                if r["finish"] == "stop" and r["resp_lang"] == "sv" and grade_numeric(r["response"], r["answer"]):
                    per.setdefault(r["id"], [])
                    # keep <=2 distinct correct solutions per problem
                    if len(per[r["id"]]) < 2 and r["response"] not in [x["response"] for x in per[r["id"]]]:
                        per[r["id"]].append(r)
        ex = [(H.chat(tok, r["prompt"]), r["response"] + eos) for v in per.values() for r in v]
        print(f"RFT: {len(ex)} examples from {len(per)} problems", flush=True)
    elif a.data == "gold":
        rows = load_jsonl(RS / "data" / "gold_sv_train.jsonl")
        ex = [(H.chat(tok, r["question_sv"] + SV_SUFFIX), r["solution_sv"] + eos) for r in rows]
        print(f"gold: {len(ex)} examples", flush=True)
    else:  # posctrl: plain English prompt (default GSM instruction) -> Swedish answer
        per = {}
        for r in load_jsonl(a.pool):
            if r["split"] != "train" or r["finish"] != "stop" or r["resp_lang"] != "sv":
                continue
            per.setdefault(r["id"], []).append(r)
        ex = []
        for k, v in per.items():
            v.sort(key=lambda r: -grade_numeric(r["response"], r["answer"]))  # prefer correct ones
            for r in v[:2]:
                plain = r["prompt"].rsplit("\n", 1)[0] + EN_SUFFIX
                ex.append((H.chat(tok, plain), r["response"] + eos))
        print(f"posctrl: {len(ex)} examples from {len(per)} problems", flush=True)
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--data", choices=["rft", "gold", "posctrl"], required=True)
    ap.add_argument("--pool", default=str(RESULTS / "math" / "pool_sv_train.jsonl"))
    ap.add_argument("--target", choices=list(TARGETS), required=True)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_steps", type=int, default=0, help="debug: stop after this many optimizer steps")
    ap.add_argument("--no_export", action="store_true")
    ap.add_argument("--dry", action="store_true", help="build data + wrap model, then exit")
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    from peft import LoraConfig, get_peft_model
    model, tok, p = H.load(a.model, dtype=torch.float32)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ex = build_examples(a, tok)
    cfg = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.0, target_modules=TARGETS[a.target])
    model = get_peft_model(model, cfg)
    lora_mods = {n: m for n, m in model.named_modules() if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {})}
    print(f"LoRA modules: {len(lora_mods)}; e.g. {list(lora_mods)[:3]}", flush=True)
    n_train = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print(f"trainable params: {n_train}", flush=True)
    if a.dry:
        return
    # original weights of the adapted matrices (to compute dW after merge)
    W0 = {n: m.base_layer.weight.detach().float().cpu().clone() for n, m in lora_mods.items()}
    # LoRA's A factor starts random: record how far it moves (if it barely moves, the input side of a rank-r edit
    # is set by the random init rather than learned)
    A0 = {n: m.lora_A["default"].weight.detach().float().cpu().clone() for n, m in lora_mods.items()}
    opt = torch.optim.AdamW([x for x in model.parameters() if x.requires_grad], lr=a.lr, weight_decay=0.0)
    model.train()
    steps, t0, losses, ep_loss = 0, time.time(), [], []
    done = False
    for ep in range(a.epochs):
        random.shuffle(ex)
        el, en = 0.0, 0
        for i in range(0, len(ex), a.bs):
            chunk = ex[i:i + a.bs]
            enc = []
            for pre, comp in chunk:
                x = tok(pre, add_special_tokens=False)["input_ids"]
                y = tok(comp, add_special_tokens=False)["input_ids"]
                enc.append(((x + y)[:a.max_len], ([-100] * len(x) + y)[:a.max_len]))
            ntok_total = sum(sum(t != -100 for t in lab[1:]) for _, lab in enc)
            tot = 0.0
            for j in range(0, len(enc), a.micro):
                mb = enc[j:j + a.micro]
                L = max(len(s) for s, _ in mb)
                inp = torch.full((len(mb), L), tok.pad_token_id, dtype=torch.long)
                lab = torch.full((len(mb), L), -100, dtype=torch.long)
                att = torch.zeros_like(inp)
                for k, (s, l) in enumerate(mb):
                    inp[k, :len(s)] = torch.tensor(s)
                    lab[k, :len(l)] = torch.tensor(l)
                    att[k, :len(s)] = 1
                ntok = int((lab[:, 1:] != -100).sum())
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=inp.to(H.DEV), attention_mask=att.to(H.DEV), labels=lab.to(H.DEV)).loss
                (loss * ntok / ntok_total).backward()
                tot += loss.item() * ntok / ntok_total
            opt.step()
            opt.zero_grad()
            steps += 1
            losses.append(tot)
            el += tot * ntok_total
            en += ntok_total
            if steps % 10 == 0:
                print(f"ep {ep} step {steps} loss {sum(losses[-10:]) / 10:.4f} {time.time() - t0:.0f}s", flush=True)
            if a.max_steps and steps >= a.max_steps:
                done = True
                break
        ep_loss.append(el / max(en, 1))
        print(f"epoch {ep} token-mean loss {ep_loss[-1]:.4f}", flush=True)
        if done:
            break
    a_move = []
    for n, m in lora_mods.items():
        A1 = m.lora_A["default"].weight.detach().float().cpu()
        a_move.append({"rel_change": float((A1 - A0[n]).norm() / A0[n].norm()),
                       "subspace_overlap": float(torch.linalg.svdvals(torch.linalg.qr(A1.T)[0].T @ torch.linalg.qr(A0[n].T)[0]).pow(2).mean())})
    A_move = {k: sum(x[k] for x in a_move) / len(a_move) for k in ("rel_change", "subspace_overlap")}
    print("A-factor movement from init:", A_move, flush=True)
    model = model.merge_and_unload()
    # dW per adapted matrix (merged - original)
    named = dict(model.named_parameters())
    dW_info, sq, dW_full = {}, 0.0, {}
    svd = {}
    for n in lora_mods:
        pn = n.replace("base_model.model.", "") + ".weight"
        Wn = named[pn].detach().float().cpu()
        d = Wn - W0[n]
        dW_full[pn] = d
        fro = float(d.norm())
        sq += fro ** 2
        dW_info[pn] = {"fro": fro, "rel": fro / float(W0[n].norm())}
        if a.target == "down15":
            U, S, Vh = torch.linalg.svd(d, full_matrices=False)
            k = min(64, a.r)
            svd = {"param": pn, "U": U[:, :k].clone(), "S": S[:k].clone(), "Vh": Vh[:k].clone()}
    model = model.to(torch.bfloat16)
    # the exported checkpoint is bf16: measure how much of dW survives rounding (small edits can round away)
    named = dict(model.named_parameters())
    num, den_e, den_d = 0.0, 0.0, 0.0
    for n in lora_mods:
        pn = n.replace("base_model.model.", "") + ".weight"
        eff = named[pn].detach().float().cpu() - W0[n]
        d = dW_full[pn]
        num += float((eff * d).sum())
        den_e += float(eff.norm()) ** 2
        den_d += float(d.norm()) ** 2
    bf16_retention = {"eff_fro": den_e ** 0.5, "ratio": (den_e / max(den_d, 1e-30)) ** 0.5,
                      "cos": num / max((den_e * den_d) ** 0.5, 1e-30)}
    print("bf16 retention of dW:", bf16_retention, flush=True)
    edited = [n.replace("base_model.model.", "") + ".weight" for n in lora_mods]
    info = {**vars(a), "n_examples": len(ex), "trainable_params": n_train, "steps": steps, "n_lora_modules": len(lora_mods),
            "epoch_loss": ep_loss, "final_loss": sum(losses[-10:]) / max(1, len(losses[-10:])),
            "dW_fro_total": sq ** 0.5, "bf16_retention": bf16_retention, "A_move": A_move, "dW_rel_mean": sum(v["rel"] for v in dW_info.values()) / max(1, len(dW_info)),
            "train_seconds": round(time.time() - t0)}
    outdir = RS / "results" / "train"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{a.out}.train.json").write_text(json.dumps({**info, "dW": dW_info}, indent=1))
    if svd:
        torch.save(svd, outdir / f"{a.out}.svd.pt")
    if not a.no_export:
        out = H.export_ckpt(model, a.model, a.out, edited, info)
        print(out)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
