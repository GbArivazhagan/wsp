#!/bin/bash
# Regenerate all summary tables of the edit-rank study into rank_study/results/tables.txt
source ${WSP_ROOT:-$PWD}/scripts/env_hf.sh > /dev/null
cd $P
{
echo "## TEST (200 GSM8K problems x 4 samples; paired vs unedited model)"; $PY rank_study/src/analyze.py test
echo; echo "## TEST gains vs the pooled 8-sample base (seed-0 + seed-1 base runs; requested by the main track)"; $PY rank_study/src/analyze.py test8
echo; echo "## Seed-pooled contrast R1-single (lr 2e-3, seeds 0+1) vs R16-all-linear (seeds 0+1)"; $PY rank_study/src/analyze.py contrast_seeds
echo; echo "## DEV (100 problems x 4, max 2048 tokens)"; $PY rank_study/src/analyze.py dev
echo; echo "## Pooled DEV+TEST (secondary; DEV was used for selection of r1d15x10)"; $PY rank_study/src/analyze.py pooled
echo; echo "## Joint bootstrap contrasts (R1-single vs high-rank)"
for b in r16all r64d15 r64d15x10 r1alldown r8d15; do [ -f rank_study/results/test_$b.jsonl ] && $PY rank_study/src/analyze.py contrast r1d15x10 $b | tr -d '\n '; echo; done
echo; echo "## Decision rule"; $PY rank_study/src/analyze.py decide r1d15x10
echo; echo "## Held-out NLL of the unedited model's DEV solutions (per token)"
for f in rank_study/results/nll/*.json; do $PY -c "import json;d=json.load(open('$f'));print('$f'.split('/')[-1][:-5], 'correct', round(d['correct']['nll_per_tok'],4), 'incorrect', round(d['incorrect']['nll_per_tok'],4), 'margin', round(d['margin'],4))"; done
echo; echo "## Cosines of learned output directions (layer 15)"; $PY rank_study/src/analyze.py cos
} 2>&1 | grep -v "Warning\|warn(" > rank_study/results/tables.txt
echo written rank_study/results/tables.txt
