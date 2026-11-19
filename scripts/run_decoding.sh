#!/bin/bash
# (1) Decoding-setting control (hypothesis raised by the 0-pass editor agent): does presence_penalty=1.5 hurt
#     Swedish more than English (Swedish words split into more sub-word tokens)?
# (2) Reasoning-language cells with the instruction in the USER turn (Qwen3.5 ignores a system prompt asking for
#     English when the problem is Swedish), and the TRAIN-pool English-reasoning data for the 'rlang' direction.
source ${WSP_ROOT:-$PWD}/scripts/env_vllm.sh
GM=${GM:-0.45}
run() { out=$1; shift; [ -f $out ] && return; mkdir -p results/locks; mkdir results/locks/$(basename $out) 2>/dev/null || return; $PY src/gen_vllm.py "$@" --gpu_mem $GM --out $out 2>&1 | grep -E "^\[|generated|Error|error:|Traceback"; }
B="--think off --temperature 0.7 --top_p 0.8 --top_k 20 --max_tokens 4096"
NT="$B --presence_penalty 1.5"
ENI="Reason step by step in English, and put your final answer within \\boxed{}."
SVI="Resonera steg för steg på svenska och skriv ditt slutgiltiga svar inom \\boxed{}."
run results/x2/base_P-sv_R-none_pp0.jsonl --model qwen3.5-0.8b --sets gsm_sv $B --presence_penalty 0 --n 4
run results/x2/base_P-en_R-none_pp0.jsonl --model qwen3.5-0.8b --sets gsm_en $B --presence_penalty 0 --n 4
run results/x2/base_P-sv_R-enU.jsonl --model qwen3.5-0.8b --sets gsm_sv $NT --n 4 --user_suffix "$ENI"
run results/x2/base_P-en_R-svU.jsonl --model qwen3.5-0.8b --sets gsm_en $NT --n 4 --user_suffix "$SVI"
run results/x2/base_P-sv_R-none_greedy.jsonl --model qwen3.5-0.8b --sets gsm_sv --think off --temperature 0 --max_tokens 4096 --n 1
run results/x2/base_P-en_R-none_greedy.jsonl --model qwen3.5-0.8b --sets gsm_en --think off --temperature 0 --max_tokens 4096 --n 1
run results/math/pool_sv_train_RenU.jsonl --model qwen3.5-0.8b --sets gsmtr_sv:train --n 2 --think off --temperature 0.7 --top_p 0.8 --top_k 20 --presence_penalty 1.5 --max_tokens 2048 --user_suffix "$ENI"
source scripts/env_hf.sh
[ -f models/dirs/rlang.pt ] || $PY src/math_edit.py directions --kind rlang --gen results/math/pool_sv_train.jsonl --gen2 results/math/pool_sv_train_RenU.jsonl 2>&1 | grep -vE "Warning|recommended|causal_conv1d|Loading"
echo DECODING_DONE
