#!/bin/bash
# Stage 2: TEST evaluation of the DEV-selected edits (each evaluated once on TEST, 4 samples per problem),
# their norm-matched random-direction twins (same layers, columns, alpha and Frobenius norm of the change), and
# regression checks (Belebele sv/en, FLORES perplexity sv/en, en/ett gender test).
# usage: SEL="msw_derail_L15-19_a1 msw_rlang_L10-14_a2 ..." bash scripts/run_math_stage2.sh
source ${WSP_ROOT:-$PWD}/scripts/env_vllm.sh
VPY=$PY; HPY=$P/venvs/hf/bin/python
GM=${GM:-0.45}
NT="--think off --temperature 0.7 --top_p 0.8 --top_k 20 --presence_penalty 1.5 --max_tokens 4096 --n 4"
mkdir -p results/mtest results/logprob
for C in $SEL; do
  kind=$(echo $C | cut -d_ -f2); ls=$(echo $C | cut -d_ -f3); a=$(echo $C | cut -d_ -f4 | tr -d a)
  layers=$($VPY -c "import sys; sys.path.insert(0,'src'); from math_sweep import LAYER_SETS; print(','.join(map(str,LAYER_SETS['$ls'])))")
  [ -d models/$C ] || CUDA_VISIBLE_DEVICES="" $HPY src/math_sweep.py build_one --cfg $C > /dev/null 2>&1
  R=${C/msw_/mrnd_}
  [ -d models/$R ] || CUDA_VISIBLE_DEVICES="" $HPY src/math_edit.py build --kind random --match $kind --layers $layers --alpha $a --topk 64 --out $R > /dev/null 2>&1
  for M in $C $R; do
    [ -f results/mtest/$M.jsonl ] || $VPY src/gen_vllm.py --model $M --sets gsm_sv,gsm_en $NT --gpu_mem $GM --out results/mtest/$M.jsonl 2>&1 | grep -E "^\[|generated|Traceback"
    [ -f results/logprob/$M.json ] || $HPY src/eval_logprob.py --model $M --tasks belebele_sv,belebele_en,ppl_sv,ppl_en,gender_test --out results/logprob/$M.json 2>&1 | grep -E "^(belebele|ppl|gender)|Traceback"
  done
done
echo STAGE2_DONE
