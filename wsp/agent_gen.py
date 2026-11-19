"""Called by harness.generate(): answer DEV pool problems (Swedish) with vLLM and grade them.
argv: model idfile out n max_tokens"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import load_jsonl, model_path  # noqa: E402
from gen_vllm import GSM_SUFFIX  # noqa: E402
from grade_math import grade_numeric  # noqa: E402
from langid import label  # noqa: E402

model, idfile, out, n, max_tokens = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
ids = set(json.loads(Path(idfile).read_text()))
rows = [r for r in load_jsonl(ROOT / "data/gsm8k/gsm8k_train_sv300.jsonl") if int(r["id"].split("_")[1]) in ids]
from vllm import LLM, SamplingParams  # noqa: E402
mp = model_path(model)
llm = LLM(model=mp, dtype="bfloat16", seed=0, gpu_memory_utilization=0.4, max_model_len=4096,
          limit_mm_per_prompt={"image": 0, "video": 0})
tok = llm.get_tokenizer()
texts = [tok.apply_chat_template([{"role": "user", "content": r["question_sv"] + GSM_SUFFIX["sv"]}], tokenize=False,
                                 add_generation_prompt=True, enable_thinking=False) for r in rows]
sp = SamplingParams(n=n, temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, max_tokens=max_tokens, seed=0)
outs = llm.generate(texts, sp)
with open(out, "w") as f:
    for r, o in zip(rows, outs):
        for c in o.outputs:
            f.write(json.dumps({"id": r["id"], "response": c.text, "finish": c.finish_reason, "answer": r["answer"],
                                "correct": grade_numeric(c.text, r["answer"]), "resp_lang": label(c.text)},
                               ensure_ascii=False) + "\n")
