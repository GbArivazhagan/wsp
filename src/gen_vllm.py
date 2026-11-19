"""Generate chat responses with vLLM for a set of prompt files, then label the response language.

Example:
  python src/gen_vllm.py --model qwen3.5-0.8b --sets sv:dev,en:dev --think off --out results/scan/qwen35.jsonl
  python src/gen_vllm.py --model models/qwen35_edit_x --sets gsm_sv --think on --max_tokens 32768 ...

--sets: comma list of <lang>:<split> for the chat prompt sets (data/prompts), or 'gsm_sv' / 'gsm_en' (data/gsm8k).
--system: none | sv (a Swedish system prompt asking for Swedish answers) | match (answer in the user's language)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
from common import DATA, REGISTRY, load_jsonl, model_path, prompt_split, save_jsonl  # noqa: E402

SYSTEMS = {
    "none": None,
    "sv": "Du är en hjälpsam assistent. Svara alltid på svenska.",
    "match": "You are a helpful assistant. Always answer in the same language as the user's message.",
}
GSM_SUFFIX = {
    "en": "\nPlease reason step by step, and put your final answer within \\boxed{}.",
    "sv": "\nResonera steg för steg och skriv ditt slutgiltiga svar inom \\boxed{}.",
}


def load_sets(spec):
    rows = []
    for s in spec.split(","):
        if s.startswith("gsmtr"):
            # GSM8K TRAIN pool translated to Swedish (300): ids 0-199 = 'train' (directions, LoRA), 200-299 = 'dev'
            # (method selection). spec: gsmtr_sv:train | gsmtr_en:dev | gsmtr_sv:all
            name, split = (s.split(":") + ["all"])[:2]
            lang = name.split("_")[1]
            keep = None
            if split.endswith("NUM"):  # numword study: devNUM = DEV problems whose Swedish text has number words
                spl = json.load(open(DATA.parent / "numword" / "data" / "num_split.json"))["pool"]
                keep = {k for k, v in spl.items() if v == "NUM"}
                split = split[:-3]
            for r in load_jsonl(DATA / "gsm8k" / "gsm8k_train_sv300.jsonl"):
                if keep is not None and r["id"] not in keep:
                    continue
                k = int(r["id"].split("_")[1])
                if split == "train" and k >= 200 or split == "dev" and k < 200:
                    continue
                q = r["question_sv"] if lang == "sv" else r["question_en"]
                rows.append({"id": r["id"] + "_" + lang, "lang": lang, "category": "gsm8k", "prompt": q + GSM_SUFFIX[lang],
                             "answer": r["answer"], "split": split})
        elif s.startswith("gsm_"):
            # gsm_sv | gsm_en | gsm_svdig (number words -> digits, numword study); optional :NUM / :NONUM subset
            name, sub = (s.split(":") + [""])[:2]
            lang = name.split("_")[1]
            src = DATA.parent / "numword" / "data" / "gsm8k_sv200_digits.jsonl" if lang == "svdig" else \
                DATA / "gsm8k" / f"gsm8k_{lang}200.jsonl"
            keep = None
            if sub:
                spl = json.load(open(DATA.parent / "numword" / "data" / "num_split.json"))["test"]
                keep = {k for k, v in spl.items() if v == sub}
            lang = "sv" if lang == "svdig" else lang
            for r in load_jsonl(src):
                if keep is not None and r["id"] not in keep:
                    continue
                q = r["question_sv"] if lang == "sv" else r["question"]
                rows.append({"id": r["id"], "lang": lang, "category": "gsm8k", "prompt": q + GSM_SUFFIX[lang],
                             "answer": r["answer"]})
        elif s.startswith("syn"):
            # numword study synthetic problems: syn_test:word | syn_test:digit | syn_train:word
            name, form = s.split(":")
            for r in load_jsonl(DATA.parent / "numword" / "data" / f"{name}.jsonl"):
                rows.append({"id": r["id"] + "_" + form, "lang": "sv", "category": "syn", "template": r["template"],
                             "prompt": r["q_" + form] + GSM_SUFFIX["sv"], "answer": r["answer"]})
        elif s.startswith("math500"):
            for r in load_jsonl(DATA / "math500" / "math500.jsonl"):
                rows.append({"id": r["id"], "lang": "en", "category": "math500",
                             "prompt": r["problem"] + GSM_SUFFIX["en"], "answer": r["answer"]})
        else:
            lang, split = s.split(":")
            rows += prompt_split(lang, split)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--system", default="none")
    ap.add_argument("--think", default="off", choices=["on", "off", "na"])
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--presence_penalty", type=float, default=0.0)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=768)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_mem", type=float, default=0.80)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--user_suffix", default="", help="replace the GSM instruction line with this text (e.g. reason in English)")
    ap.add_argument("--prefill", default="", help="text the assistant turn starts with (counted as part of the response)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    rows = load_sets(a.sets)
    if a.limit:
        rows = rows[: a.limit]
    mpath = model_path(a.model)
    kw = dict(model=mpath, dtype="bfloat16", seed=a.seed, gpu_memory_utilization=a.gpu_mem,
              max_model_len=max(a.max_model_len, a.max_tokens + 1024), enable_prefix_caching=True)
    if "qwen3.5" in a.model.lower() or "qwen35" in a.model.lower() or "Qwen3_5" in open(Path(mpath) / "config.json").read():
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    sysmsg = SYSTEMS.get(a.system, a.system)
    texts = []
    for r in rows:
        prompt = r["prompt"]
        if a.user_suffix:
            prompt = prompt.rsplit("\n", 1)[0] + "\n" + a.user_suffix.replace("\\n", "\n")
            r["prompt"] = prompt
        msgs = ([{"role": "system", "content": sysmsg}] if sysmsg else []) + [{"role": "user", "content": prompt}]
        extra = {} if a.think == "na" else {"enable_thinking": a.think == "on"}
        texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **extra) + a.prefill)
    sp = SamplingParams(n=a.n, temperature=a.temperature, top_p=a.top_p if a.temperature > 0 else 1.0,
                        top_k=a.top_k if a.temperature > 0 else -1, presence_penalty=a.presence_penalty,
                        repetition_penalty=a.repetition_penalty, max_tokens=a.max_tokens, seed=a.seed)
    t0 = time.time()
    outs = llm.generate(texts, sp)
    dt = time.time() - t0
    from langid import label
    res = []
    for r, o in zip(rows, outs):
        for k, c in enumerate(o.outputs):
            res.append({**r, "sample": k, "response": a.prefill + c.text, "n_tokens": len(c.token_ids),
                        "finish": c.finish_reason, "resp_lang": label(c.text)})
    save_jsonl(res, a.out)
    cfg = {**vars(a), "model_path": mpath, "registry": REGISTRY.get(a.model), "system_text": sysmsg,
           "n_rows": len(rows), "gen_seconds": round(dt, 1), "example_prompt": texts[0]}
    Path(a.out).with_suffix(".config.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
    # quick summary
    import collections
    by = collections.defaultdict(collections.Counter)
    for r in res:
        by[r["lang"]][r["resp_lang"]] += 1
    for l, c in by.items():
        n = sum(c.values())
        print(f"[{Path(a.out).stem}] prompts={l} n={n} " + " ".join(f"{k}:{v/n:.2f}" for k, v in c.most_common()))
    print(f"generated in {dt:.0f}s")


if __name__ == "__main__":
    main()
