"""Build DEV contrast pairs for the language-fidelity direction and DEV/TRAIN data for the LoRA baseline.

For each Swedish DEV prompt, generate with vLLM
   resp_sv : with the system prompt 'Svara alltid på svenska.'
   resp_en : with the system prompt 'Always answer in English.'
and keep pairs whose language ID confirms sv / en. These pairs are later teacher-forced under the *plain*
prompt (no system prompt), so the contrast isolates the response-language decision.
Also writes LoRA context-distillation data (prompt without system prompt -> Swedish answer produced with it),
plus English DEV prompts -> the model's own English answers (to preserve English behaviour).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, model_path, prompt_split, save_jsonl  # noqa: E402
from langid import label  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--gpu_mem", type=float, default=0.5)
    ap.add_argument("--max_tokens", type=int, default=384)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    mp = model_path(a.model)
    kw = dict(model=mp, dtype="bfloat16", seed=0, gpu_memory_utilization=a.gpu_mem, max_model_len=2048)
    if "Qwen3_5" in open(Path(mp) / "config.json").read():
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    llm = LLM(**kw)
    tok = llm.get_tokenizer()

    def fmt(p, sysmsg):
        m = ([{"role": "system", "content": sysmsg}] if sysmsg else []) + [{"role": "user", "content": p}]
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    sv = prompt_split("sv", "dev")
    en = prompt_split("en", "dev")
    sp = SamplingParams(n=a.n, temperature=0.7, top_p=0.8, top_k=20, max_tokens=a.max_tokens, seed=0)
    o_sv = llm.generate([fmt(r["prompt"], "Du är en hjälpsam assistent. Svara alltid på svenska.") for r in sv], sp)
    o_en = llm.generate([fmt(r["prompt"], "You are a helpful assistant. Always answer in English.") for r in sv], sp)
    o_enp = llm.generate([fmt(r["prompt"], None) for r in en], sp)
    pairs, distill = [], []
    for r, s, e in zip(sv, o_sv, o_en):
        S = [c.text for c in s.outputs if c.finish_reason == "stop" and label(c.text) == "sv"]
        E = [c.text for c in e.outputs if c.finish_reason == "stop" and label(c.text) == "en"]
        if S and E:
            pairs.append({"id": r["id"], "prompt": r["prompt"], "resp_sv": S[0], "resp_en": E[0]})
        for t in S:
            distill.append({"id": r["id"], "prompt": r["prompt"], "completion": t, "kind": "sv_distill"})
    for r, e in zip(en, o_enp):
        for c in e.outputs:
            if c.finish_reason == "stop" and label(c.text) == "en":
                distill.append({"id": r["id"], "prompt": r["prompt"], "completion": c.text, "kind": "en_keep"})
    save_jsonl(pairs, DATA / "contrast" / f"{a.model}_lang_contrast.jsonl")
    save_jsonl(distill, DATA / "contrast" / f"{a.model}_lang_distill.jsonl")
    print(f"pairs {len(pairs)} / {len(sv)} ; distill rows {len(distill)}")


if __name__ == "__main__":
    main()
