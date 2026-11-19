#!/bin/bash
# Swedish-thinking study: one generation run. usage: run.sh MODEL TAG LANG(sv|en) SEED GM
source ${WSP_ROOT:-$PWD}/scripts/env_vllm.sh > /dev/null
M=$1; TAG=$2; LG=$3; SEED=$4; GM=${5:-0.75}
if [ $LG = sv ]; then PF="Okej, jag löser det här steg för steg och tänker på svenska.

"; else PF="Okay, I'll solve this step by step and think in English.

"; fi
out=swthink/results/${TAG}_${LG}think_s${SEED}.jsonl
[ -f $out ] && exit 0
MT=16384; [ $TAG = base ] && MT=8192   # unedited model: generate 8k, then budget-force at 8k (time limit, see prereg)
$PY src/gen_vllm.py --model $M --sets gsm_sv --think on --temperature 0.6 --top_p 0.95 --top_k 20 --presence_penalty 1.5 \
  --max_tokens $MT --max_model_len $((MT + 1024)) --n 1 --seed $SEED --gpu_mem $GM --prefill "$PF" --out $out.tmp 2>&1 | grep -E "Error|Traceback|generated" | tail -n 3
[ -f $out.tmp ] && mv $out.tmp $out && mv $out.tmp.config.json $out.config.json 2>/dev/null
if [ $TAG = base ]; then
  $PY src/budget_force.py --src $out --budget 8192 --gpu_mem $GM --out swthink/results/${TAG}_${LG}think_s${SEED}_bf8192.jsonl 2>&1 | grep -E "budget|Error" | tail -n 2
fi
echo "$(date) $TAG $LG s$SEED done"
