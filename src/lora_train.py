"""Conventional fine-tuning comparator: LoRA (gradient descent), then merge and export a vLLM-loadable checkpoint.

  --data lang   : context distillation (data/contrast/<model>_lang_distill.jsonl): chat prompt WITHOUT system
                  prompt -> the Swedish answer the model itself gave WITH 'Svara alltid på svenska' (+ English
                  DEV prompts -> its own English answers, to keep English behaviour). DEV prompts only.
  --data gender : LM loss on the correct sentences of the TRAIN nouns only (data/gender/gender_train.jsonl);
                  test nouns are never seen. Measures whether gradient training generalises gender to new nouns.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, load_jsonl  # noqa: E402
import hf_tools as H  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--data", choices=["lang", "gender", "rft"], required=True)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_steps", type=int, default=0, help="debug: stop after this many steps")
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    from peft import LoraConfig, get_peft_model
    model, tok, p = H.load(a.model, dtype=torch.float32)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if a.data == "lang":
        rows = load_jsonl(DATA / "contrast" / f"{a.model}_lang_distill.jsonl")
        ex = [(H.chat(tok, r["prompt"]), r["completion"] + (tok.eos_token or "")) for r in rows]
    elif a.data == "rft":
        # rejection-sampling fine-tuning: the model's own CORRECT Swedish solutions on the TRAIN pool (no other data)
        from grade_math import grade_numeric
        from common import RESULTS
        rows = load_jsonl(RESULTS / "math" / "pool_sv_train.jsonl")
        per = {}
        for r in rows:
            if r["finish"] == "stop" and r["resp_lang"] == "sv" and grade_numeric(r["response"], r["answer"]):
                per.setdefault(r["id"], [])
                if len(per[r["id"]]) < 2:
                    per[r["id"]].append(r)
        ex = [(H.chat(tok, r["prompt"]), r["response"] + (tok.eos_token or "")) for v in per.values() for r in v]
        print(f"RFT: {len(ex)} examples from {len(per)} problems")
    else:
        rows = load_jsonl(DATA / "gender" / "gender_train.jsonl")
        ex = [("", r["good"]) for r in rows]
    targets = ["down_proj", "up_proj", "gate_proj", "o_proj", "q_proj", "k_proj", "v_proj", "out_proj"]
    cfg = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.0, target_modules=targets)
    model = get_peft_model(model, cfg)
    n_train = sum(x.numel() for x in model.parameters() if x.requires_grad)
    opt = torch.optim.AdamW([x for x in model.parameters() if x.requires_grad], lr=a.lr)
    model.train()
    steps, t0, losses = 0, time.time(), []
    for ep in range(a.epochs):
        random.shuffle(ex)
        for i in range(0, len(ex), a.bs):
            chunk = ex[i:i + a.bs]
            ids, labs = [], []
            for pre, comp in chunk:
                x = tok(pre, add_special_tokens=False)["input_ids"] if pre else []
                y = tok(comp, add_special_tokens=False)["input_ids"]
                seq = (x + y)[:768]
                lab = ([-100] * len(x) + y)[:768]
                if not pre:
                    lab[0] = -100
                ids.append(seq)
                labs.append(lab)
            L = max(map(len, ids))
            inp = torch.full((len(ids), L), tok.pad_token_id, dtype=torch.long)
            lab = torch.full((len(ids), L), -100, dtype=torch.long)
            att = torch.zeros_like(inp)
            for j in range(len(ids)):
                inp[j, :len(ids[j])] = torch.tensor(ids[j])
                lab[j, :len(labs[j])] = torch.tensor(labs[j])
                att[j, :len(ids[j])] = 1
            with torch.autocast("cuda" if H.DEV == "cuda" else "cpu", dtype=torch.bfloat16):
                loss = model(input_ids=inp.to(H.DEV), attention_mask=att.to(H.DEV), labels=lab.to(H.DEV)).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            steps += 1
            losses.append(loss.item())
            if a.max_steps and steps >= a.max_steps:
                break
            if steps % 20 == 0:
                print(f"ep {ep} step {steps} loss {sum(losses[-20:]) / 20:.4f} {time.time() - t0:.0f}s", flush=True)
    model = model.merge_and_unload().to(torch.bfloat16)
    edited = [n for n, _ in model.named_parameters() if any(f".{t}." in n for t in targets)]
    info = {**vars(a), "n_examples": len(ex), "trainable_params": n_train, "steps": steps,
            "final_loss": sum(losses[-20:]) / max(1, len(losses[-20:])), "train_seconds": round(time.time() - t0)}
    out = H.export_ckpt(model, a.model, a.out, edited, info)
    print(out, json.dumps(info))


if __name__ == "__main__":
    main()
