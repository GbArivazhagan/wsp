#!/bin/bash
# Stage 1: TRAIN-pool generations -> directions -> build grid part 0 on CPU -> evaluate part 0 on DEV.
source ${WSP_ROOT:-$PWD}/scripts/env_vllm.sh
VPY=$PY; HPY=$P/venvs/hf/bin/python
GM=${GM:-0.45}
NT="--think off --temperature 0.7 --top_p 0.8 --top_k 20 --presence_penalty 1.5 --max_tokens 2048"
EN="You are a helpful assistant. Always answer in English."
mkdir -p results/math results/msweep
run() { out=$1; shift; [ -f $out ] && return; $VPY src/gen_vllm.py "$@" --gpu_mem $GM --out $out 2>&1 | grep -E "^\[|generated|Error|error:|Traceback"; }
run results/math/pool_sv_train.jsonl --model qwen3.5-0.8b --sets gsmtr_sv:train --n 4 $NT
run results/math/pool_sv_train_Ren.jsonl --model qwen3.5-0.8b --sets gsmtr_sv:train --n 2 --system "$EN" $NT
run results/math/pool_en_train.jsonl --model qwen3.5-0.8b --sets gsmtr_en:train --n 1 --think off --max_tokens 8
run results/msweep/msw_base.jsonl --model qwen3.5-0.8b --sets gsmtr_sv:dev,gsmtr_en:dev --n 2 $NT
source scripts/env_hf.sh
[ -f models/dirs/derail.pt ] || $HPY src/math_edit.py directions --kind derail --gen results/math/pool_sv_train.jsonl 2>&1 | grep -vE "Warning|recommended|causal_conv1d|Loading"
[ -f models/dirs/rlang.pt ] || $HPY src/math_edit.py directions --kind rlang --gen results/math/pool_sv_train.jsonl --gen2 results/math/pool_sv_train_Ren.jsonl 2>&1 | grep -vE "Warning|recommended|causal_conv1d|Loading"
[ -f models/dirs/inalign.pt ] || $HPY src/math_edit.py directions --kind inalign --gen results/math/pool_sv_train.jsonl --gen2 results/math/pool_en_train.jsonl 2>&1 | grep -vE "Warning|recommended|causal_conv1d|Loading"
touch results/msweep/DIRS_READY
CUDA_VISIBLE_DEVICES="" $HPY src/math_sweep.py build --part 0/2 2>&1 | grep -E "built|Error|Traceback"
source scripts/env_vllm.sh
$VPY src/math_sweep.py eval --part 0/2 --gpu_mem $GM 2>&1 | grep -E "^\[|generated|Error|error:|Traceback"
echo STAGE1_DONE
