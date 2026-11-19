#!/bin/bash
# Exp D non-thinking with 4 samples per problem (paired comparisons)
source ${WSP_ROOT:-$PWD}/scripts/env_vllm.sh
GM=${GM:-0.5}
run() { out=$1; shift; [ -f $out ] && return; $PY src/gen_vllm.py "$@" --gpu_mem $GM --out $out 2>&1 | grep -E "^\[|generated|Error|error:|Traceback"; }
NT="--think off --temperature 0.7 --top_p 0.8 --top_k 20 --presence_penalty 1.5 --max_tokens 4096 --n 4 --seed 1"
run results/expD/gsm_nothink_n4_base.jsonl --model qwen3.5-0.8b --sets gsm_sv,gsm_en $NT
run results/expD/gsm_nothink_n4_p50.jsonl --model qwen35_paper_p50c --sets gsm_sv,gsm_en $NT
run results/expD/gsm_nothink_n4_rnd.jsonl --model qwen35_paper_p50c_randctrl --sets gsm_sv,gsm_en $NT
echo Q2_DONE
