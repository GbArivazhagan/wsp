#!/bin/bash
# numword study: TEST evaluation of one model. NUM subset 8 samples, NONUM 4 samples; two cells:
#   svsv = Swedish problem, default (Swedish) reasoning;  sven = Swedish problem, English answer prefill.
# usage: run_eval.sh MODEL TAG [SET=gsm_sv] [GM=0.4] [CELLS="svsv sven"] [PARTS="NUM NONUM"]
source ${WSP_ROOT:-$PWD}/scripts/env_vllm.sh > /dev/null
M=$1; TAG=$2; SET=${3:-gsm_sv}; GM=${4:-0.4}; CELLS=${5:-"svsv sven"}; PARTS=${6:-"NUM NONUM"}
O=numword/results/test; mkdir -p $O
NT="--think off --temperature 0.7 --top_p 0.8 --top_k 20 --presence_penalty 1.5 --max_tokens 4096 --max_model_len 6144 --seed 0"
for c in $CELLS; do for part in $PARTS; do
  n=$([ $part = NUM ] && echo 8 || echo 4)
  out=$O/${TAG}_${c}_${part}.jsonl
  [ -f $out ] && continue
  pf=(); [ $c = sven ] && pf=(--prefill "Let's solve this step by step in English.

")
  $PY src/gen_vllm.py --model $M --sets $SET:$part $NT --n $n --gpu_mem $GM "${pf[@]}" --out $out.tmp 2>&1 | grep -E "Error|error:|Traceback|generated" | tail -n 3
  [ -f $out.tmp ] && mv $out.tmp $out
done; done
echo "$(date) $TAG done"
