#!/bin/bash
# Exp D: does the paper's loop-removal edit (p50c) transfer to Swedish prompts, and how does it compare with
# simply using the model's default non-thinking mode? Thinking ON uses the paper's sampling (T 0.6, top_p 0.95,
# top_k 20, presence 1.5) but a 16,384-token cap (paper: 32,768) to fit our GPUs; thinking OFF uses Qwen's
# non-thinking setting (T 0.7, top_p 0.8, top_k 20, presence 1.5), as in the paper's own 'qwen35' profile.
source ${WSP_ROOT:-$PWD}/scripts/env_vllm.sh
GM=${GM:-0.45}
run() { out=$1; shift; [ -f $out ] && return; $PY src/gen_vllm.py "$@" --gpu_mem $GM --out $out 2>&1 | grep -E "^\[|generated|Error|error:|Traceback"; }
M_base=qwen3.5-0.8b; M_p50=qwen35_paper_p50c; M_rnd=qwen35_paper_p50c_randctrl
NT="--think off --temperature 0.7 --top_p 0.8 --top_k 20 --presence_penalty 1.5 --max_tokens 4096"
TH="--think on --temperature 0.6 --top_p 0.95 --top_k 20 --presence_penalty 1.5 --max_tokens 16384 --max_model_len 17408"
for m in base p50 rnd; do eval M=\$M_$m; run results/expD/gsm_nothink_${m}.jsonl --model $M --sets gsm_sv,gsm_en $NT; done
for m in base p50; do eval M=\$M_$m; run results/expD/math500_nothink_${m}.jsonl --model $M --sets math500 --think off --temperature 0.7 --top_p 0.8 --top_k 20 --presence_penalty 1.5 --max_tokens 8192; done
for m in base p50 rnd; do eval M=\$M_$m; run results/expD/gsm_think16k_${m}.jsonl --model $M --sets gsm_sv,gsm_en $TH; done
echo EXPD_DONE
