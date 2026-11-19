#!/bin/bash
# numword study: synthetic held-out problems (SYN-test), word and digit forms, greedy, 1 sample. usage: run_syn.sh MODEL TAG [GM]
source ${WSP_ROOT:-$PWD}/scripts/env_vllm.sh > /dev/null
M=$1; TAG=$2; GM=${3:-0.35}; O=numword/results/syn; mkdir -p $O
for f in word digit; do
  out=$O/${TAG}_${f}.jsonl; [ -f $out ] && continue
  $PY src/gen_vllm.py --model $M --sets syn_test:$f --think off --temperature 0 --max_tokens 1536 --max_model_len 4096 --n 1 --seed 0 \
      --gpu_mem $GM --out $out.tmp 2>&1 | grep -E "Error|error:|Traceback" | tail -n 3
  [ -f $out.tmp ] && mv $out.tmp $out
done
echo "$(date) syn $TAG done"
