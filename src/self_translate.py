"""Inference-time baseline 'self-translate' (Etxaniz et al. 2023): the model first translates the Swedish problem
into English (greedy), then solves its own translation in English. No weights change.
python src/self_translate.py --model qwen3.5-0.8b --out results/x2/base_P-sv_selftranslate.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, load_jsonl, model_path, save_jsonl  # noqa: E402
from gen_vllm import GSM_SUFFIX  # noqa: E402
from langid import label  # noqa: E402

TR = "Översätt följande matematikproblem till engelska. Skriv bara översättningen, ingenting annat.\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--gpu_mem", type=float, default=0.45)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    mp = model_path(a.model)
    llm = LLM(model=mp, dtype="bfloat16", seed=0, gpu_memory_utilization=a.gpu_mem, max_model_len=6144,
              limit_mm_per_prompt={"image": 0, "video": 0}, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    fmt = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
    rows = load_jsonl(DATA / "gsm8k" / "gsm8k_sv200.jsonl")
    tr = llm.generate([fmt(TR + r["question_sv"]) for r in rows], SamplingParams(temperature=0, max_tokens=512))
    trans = [o.outputs[0].text.strip() for o in tr]
    sp = SamplingParams(n=a.n, temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, max_tokens=4096, seed=0)
    outs = llm.generate([fmt(t + GSM_SUFFIX["en"]) for t in trans], sp)
    res = []
    for r, t, o in zip(rows, trans, outs):
        for k, c in enumerate(o.outputs):
            res.append({"id": r["id"], "lang": "sv", "category": "gsm8k", "prompt": r["question_sv"], "translation": t,
                        "translation_lang": label(t), "answer": r["answer"], "sample": k, "response": c.text,
                        "n_tokens": len(c.token_ids), "finish": c.finish_reason, "resp_lang": label(c.text)})
    save_jsonl(res, a.out)
    print(f"[selftranslate] n={len(res)} translations in English: {sum(label(t) == 'en' for t in trans)}/{len(trans)}")


if __name__ == "__main__":
    main()
