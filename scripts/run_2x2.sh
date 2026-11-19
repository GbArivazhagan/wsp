#!/bin/bash
# 2x2 diagnostic on the same 200 GSM8K problems: problem language (sv/en) x reasoning language (sv/en, via system
# prompt). Non-thinking (default) mode, Qwen non-thinking sampling, 4 samples per problem.
source ${WSP_ROOT:-$PWD}/scripts/env_vllm.sh
GM=${GM:-0.45}
run() { out=$1; shift; [ -f $out ] && return; $PY src/gen_vllm.py "$@" --gpu_mem $GM --out $out 2>&1 | grep -E "^\[|generated|Error|error:|Traceback"; }
NT="--think off --temperature 0.7 --top_p 0.8 --top_k 20 --presence_penalty 1.5 --max_tokens 4096 --n 4"
SV="Du är en hjälpsam assistent. Svara alltid på svenska."
EN="You are a helpful assistant. Always answer in English."
M=${M:-qwen3.5-0.8b}; tag=${TAG:-base}
run results/x2/${tag}_P-sv_R-sv.jsonl --model $M --sets gsm_sv --system "$SV" $NT
run results/x2/${tag}_P-sv_R-en.jsonl --model $M --sets gsm_sv --system "$EN" $NT
run results/x2/${tag}_P-en_R-en.jsonl --model $M --sets gsm_en --system "$EN" $NT
run results/x2/${tag}_P-en_R-sv.jsonl --model $M --sets gsm_en --system "$SV" $NT
run results/x2/${tag}_P-sv_R-none.jsonl --model $M --sets gsm_sv $NT
run results/x2/${tag}_P-en_R-none.jsonl --model $M --sets gsm_en $NT
echo X2_DONE
