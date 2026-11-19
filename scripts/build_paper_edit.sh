#!/bin/bash
# Rebuild the paper's Experiment 8 edit (p50c) bit-exact from Qwen3.5-0.8B rev 2fc0636 with the authors'
# apply_edit.py, then build the norm-matched random twin of p50c. Run from the repo root after scripts/download.py.
set -e
source scripts/env_vllm.sh > /dev/null
SNAP=$($PY -c "import sys; sys.path.insert(0,'src'); import common; print(common.model_path('qwen3.5-0.8b'))")
ST=model.safetensors-00001-of-00001.safetensors
for r in p50c; do
  D=models/qwen35_paper_$r; mkdir -p $D
  for f in $SNAP/*; do [ "$(basename $f)" = $ST ] || ln -sf "$(readlink -f $f)" $D/; done   # config + tokenizer files
  [ -f $D/$ST ] || $PY data/wsp_logs/exp8_math/code/apply_edit.py $SNAP/$ST data/wsp_logs/exp8_math/runs/$r/edit $D/$ST
done
venvs/hf/bin/python src/build_paper_controls.py     # -> models/qwen35_paper_p50c_randctrl (+ hash check)
