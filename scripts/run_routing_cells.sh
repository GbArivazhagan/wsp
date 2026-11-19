#!/bin/bash
# Controlled routing cells (claim-based): Swedish problem + English-reasoning prefill; self-translate pipeline.
source ${WSP_ROOT:-$PWD}/scripts/env_vllm.sh
GM=${GM:-0.45}
mkdir -p results/locks
o=results/x2/base_P-sv_R-enPrefill.jsonl
[ -f $o ] || { mkdir results/locks/$(basename $o) 2>/dev/null && $PY src/gen_vllm.py --model qwen3.5-0.8b --sets gsm_sv --think off --temperature 0.7 --top_p 0.8 --top_k 20 --presence_penalty 1.5 --max_tokens 4096 --n 4 --prefill "Let's solve this step by step in English.

" --gpu_mem $GM --out $o 2>&1 | grep -E "^\[|generated|Traceback"; }
o=results/x2/base_P-sv_selftranslate.jsonl
[ -f $o ] || { mkdir results/locks/$(basename $o) 2>/dev/null && $PY src/self_translate.py --gpu_mem $GM --out $o 2>&1 | grep -E "^\[|Traceback"; }
echo ROUTING_DONE
