#!/bin/bash
# usage: bash run.sh <workdir> <script.py>   -- archives the script and its output in <workdir>/code/, prints output
W=$1; S=$2
mkdir -p $W/code
k=$(ls $W/code/*.py 2>/dev/null | wc -l); k=$(printf "%03d" $k)
cp "$S" $W/code/$k.py
source ${WSP_ROOT:-$PWD}/scripts/env_hf.sh >/dev/null
if [ "$(cat $W/budget.json | python3 -c 'import sys,json; print(json.load(sys.stdin)["passes_total"])')" = "0" ]; then export CUDA_VISIBLE_DEVICES=""; fi
cd $W && timeout 1800 $PY -u $W/code/$k.py 2>&1 | grep -vE "Warning|recommended|causal_conv1d|Loading weights" > $W/code/$k.out
echo "[run $k, exit ${PIPESTATUS[0]}]"; head -c 20000 $W/code/$k.out
