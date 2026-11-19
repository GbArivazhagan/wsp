#!/bin/bash
# Step 1: build edit B (+ twins), then train C (rank-1 LoRA, layer-15 down_proj) and D (LoRA r16, all linear layers)
mkdir -p numword/logs
source ${WSP_ROOT:-$PWD}/scripts/env_hf.sh > /dev/null
$PY -u numword/src/build_edit_B.py > numword/logs/build_B.log 2>&1
echo "$(date) B built" 
