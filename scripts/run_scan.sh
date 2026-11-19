#!/bin/bash
# Baseline scan: response-language fidelity of small open models on dev prompts (method-development split).
source ${WSP_ROOT:-$PWD}/scripts/env_vllm.sh
GM=${GM:-0.5}
for m in qwen3.5-0.8b qwen2.5-0.5b-it qwen2.5-1.5b-it qwen3-0.6b smollm2-1.7b-it olmo2-1b-it eurollm-1.7b-it qwen3.5-2b; do
  for sys in none sv; do
    sets="sv:dev,en:dev,da:dev,nb:dev"; [ $sys = sv ] && sets="sv:dev"
    out=results/scan/${m}_sys-${sys}.jsonl
    [ -f $out ] && continue
    $PY src/gen_vllm.py --model $m --sets $sets --system $sys --think off --n 2 --max_tokens 512 --gpu_mem $GM --out $out 2>&1 | grep -E "^\[|generated|Error|error:|Traceback" 
  done
done
echo SCAN_DONE
