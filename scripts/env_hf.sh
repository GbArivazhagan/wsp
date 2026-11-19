# Source from anywhere: sets the repo root P, HF_HOME, PY (transformers/peft venv), then cds to the repo root.
export P=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export WSP_ROOT=$P
[ -f "$P/scripts/env_local.sh" ] && source "$P/scripts/env_local.sh"   # machine-specific overrides (not in git)
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export MPLCONFIGDIR=/tmp/mpl_$USER TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export PY=$P/venvs/hf/bin/python
cd $P
