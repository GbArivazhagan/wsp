"""Inference-time control for thinking-mode looping ("budget forcing", Muennighoff et al. 2025). No weights change.
Take the unedited model's 16k-token thinking generations. Wherever the thinking is longer than B tokens, or never
closed, cut it at B tokens, close it with </think> and let the model write its final answer. Thinking that closed
within B tokens is kept as it is. Budgets are fixed in advance (4096 and 8192) and both are reported; there is no
selection on TEST.
python src/budget_force.py --src results/expD/gsm_think16k_base.jsonl --budget 4096 --out results/expD/gsm_think16k_base_bf4096.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_jsonl, model_path, save_jsonl  # noqa: E402
from grade_math import grade_numeric  # noqa: E402
from langid import label  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--src", required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--answer_tokens", type=int, default=1024)
    ap.add_argument("--gpu_mem", type=float, default=0.8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_path(a.model), dtype="bfloat16", seed=0, gpu_memory_utilization=a.gpu_mem,
              max_model_len=a.budget + a.answer_tokens + 1024, limit_mm_per_prompt={"image": 0, "video": 0},
              enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    rows = load_jsonl(a.src)
    out, jobs = [], []
    for r in rows:
        think, sep, _ = r["response"].partition("</think>")
        ids = tok.encode(think, add_special_tokens=False)
        if sep and len(ids) <= a.budget:
            out.append(dict(r, forced=False, budget=a.budget))
            continue
        cut = tok.decode(ids[: a.budget])
        head = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
        jobs.append((len(out), cut, head + cut + "\n</think>\n\n"))
        out.append(dict(r, forced=True, budget=a.budget))
    sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=a.answer_tokens, seed=0)
    t0 = time.time()
    gens = llm.generate([j[2] for j in jobs], sp)
    for (i, cut, _), g in zip(jobs, gens):
        c = g.outputs[0]
        out[i].update(response=cut + "\n</think>\n\n" + c.text, finish=c.finish_reason, resp_lang=label(c.text),
                      n_tokens=len(tok.encode(cut, add_special_tokens=False)) + len(c.token_ids))
    save_jsonl(out, a.out)
    cfg = dict(vars(a), n_rows=len(out), n_forced=len(jobs), gen_seconds=round(time.time() - t0, 1),
               sampling=dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, seed=0))
    Path(a.out).with_suffix(".config.json").write_text(json.dumps(cfg, indent=1))
    for lang in ("sv", "en"):
        R = [r for r in out if r["lang"] == lang]
        acc = sum(grade_numeric(r["response"], r["answer"]) for r in R) / len(R)
        print(f"[budget {a.budget}] {lang}: acc {acc:.3f}, forced {sum(r['forced'] for r in R)}/{len(R)}, "
              f"answers in {lang} {sum(r['resp_lang'] == lang for r in R) / len(R):.2f}")


if __name__ == "__main__":
    main()
