#!/bin/bash
# Step 2: synthetic probe of the unedited model; after edit B is built, rebuild its twins norm-matched
mkdir -p numword/logs
cd ${WSP_ROOT:-$PWD}
numword/scripts/run_syn.sh qwen3.5-0.8b base 0.35 > numword/logs/syn_base_h.log 2>&1
until grep -q "B built" numword/logs/step1_build_B.log; do sleep 20; done
source scripts/env_hf.sh > /dev/null
$PY -u numword/src/build_Btwin_matched.py > numword/logs/build_Btwin.log 2>&1
echo "$(date) twins rebuilt"
