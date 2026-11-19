#!/bin/bash
# routing cells (claim-based), then the grid worker (claim-based); runs on whichever GPU host calls it
source ${WSP_ROOT:-$PWD}/scripts/env_vllm.sh
GM=${GM:-0.45} bash scripts/run_routing_cells.sh
source ${WSP_ROOT:-$PWD}/scripts/env_vllm.sh
while [ ! -f results/msweep/DIRS_READY ]; do sleep 30; done
$PY src/math_sweep.py worker --grid all --gpu_mem ${GM:-0.45} 2>&1 | grep -E "^\[|generated|Error|error:|Traceback"
echo WORKER_DONE
