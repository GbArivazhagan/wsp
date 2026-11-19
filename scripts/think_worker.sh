#!/bin/bash
# Claim-based worker for the thinking-mode GSM8K runs (sv+en TEST, 1 sample, 16k cap).
# usage: think_worker.sh <gpu_mem> [pid_to_wait_for | pidfile_to_wait_for]
mkdir -p logs
source ${WSP_ROOT:-$PWD}/scripts/env_vllm.sh
GM=${1:-0.55}; W=${2:-}
L=results/expD/locks; mkdir -p $L
alive() { if [ -f "$W" ]; then kill -0 $(cat "$W") 2>/dev/null; else [ -n "$W" ] && kill -0 $W 2>/dev/null; fi; }
while alive; do sleep 60; done
for m in base:qwen3.5-0.8b p50:qwen35_paper_p50c rnd:qwen35_paper_p50c_randctrl; do t=${m%%:*}; M=${m#*:}; o=results/expD/gsm_think16k_$t.jsonl
  mkdir $L/think_$t 2>/dev/null || continue
  hostname > $L/think_$t/info
  echo "$(date +%T) $(hostname) start $t"
  rm -f $o
  $PY src/gen_vllm.py --model $M --sets gsm_sv,gsm_en --think on --temperature 0.6 --top_p 0.95 --top_k 20 --presence_penalty 1.5 --max_tokens 16384 --max_model_len 17408 --gpu_mem $GM --out $o > logs/think16k_$t.$(hostname).log 2>&1
  rc=$?; echo "$(date +%T) $(hostname) end $t rc=$rc"
  [ $rc -ne 0 ] && rm -rf $L/think_$t
done
echo THINK_WORKER_DONE
