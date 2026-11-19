#!/bin/bash
# Step 3: (a) train C and D in parallel, (b) synthetic probe of base and DEV runs for edit B alpha selection,
mkdir -p numword/logs
# (c) build C twin. Run on one GPU.
cd ${WSP_ROOT:-$PWD}
L=numword/logs
( source scripts/env_hf.sh > /dev/null; $PY -u numword/src/lora_syn.py --data syn --target down15 --r 1 --out nw_C > $L/train_C.log 2>&1
  $PY -u numword/src/build_Ctwin.py > $L/build_Ctwin.log 2>&1; echo "$(date) C + twin done" ) &
( source scripts/env_hf.sh > /dev/null; $PY -u numword/src/lora_syn.py --data syn --target alllinear --r 16 --out nw_D > $L/train_D.log 2>&1; echo "$(date) D done" ) &
( numword/scripts/run_syn.sh qwen3.5-0.8b base 0.18 > $L/syn_base.log 2>&1
  source scripts/env_vllm.sh > /dev/null
  NT="--think off --temperature 0.7 --top_p 0.8 --top_k 20 --presence_penalty 1.5 --max_tokens 4096 --max_model_len 6144 --seed 0 --n 4"
  mkdir -p numword/results/dev
  for m in base nw_B_a1 nw_B_a2 nw_B_a4; do
    mm=$([ $m = base ] && echo qwen3.5-0.8b || echo $m)
    out=numword/results/dev/${m}_devNUM.jsonl; [ -f $out ] && continue
    $PY src/gen_vllm.py --model $mm --sets gsmtr_sv:devNUM $NT --gpu_mem 0.18 --out $out.tmp 2>&1 | grep -E "Error|Traceback" | tail -n 2
    [ -f $out.tmp ] && mv $out.tmp $out
  done; echo "$(date) DEV done" ) &
wait
echo "$(date) stage2 done"
