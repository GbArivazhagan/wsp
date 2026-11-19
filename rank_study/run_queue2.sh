#!/bin/bash
# Shared job queue for the edit-rank study (several machines can run this on the shared /scratch).
# queue.txt lines: name|needs(comma list or -)|command   ; state in qstate/: <name>.lock (claimed) .done .fail
# Stops starting jobs after 22:35 (hard stop 22:50) or when qstate/STOP exists.
source ${WSP_ROOT:-$PWD}/scripts/env_vllm.sh
RS=$P/rank_study; Q=$RS/queue.txt; S=$RS/qstate; mkdir -p $S $RS/logs
GPU_MEM=${GPU_MEM:-0.85}; export GPU_MEM
echo $$ > $S/runner_$(hostname).pid
while true; do
  [ -e $S/STOP ] && { echo "$(date +%T) STOP"; exit 0; }
  hm=$((10#$(date +%H)*60 + 10#$(date +%M)))
  (( hm >= 22*60+35 )) && { echo "$(date +%T) past 22:35, exit"; exit 0; }
  picked=""
  # TRAIN_FIRST=1: first look only at training jobs (tr_*), then at the rest. SKIP_TRAIN=1: never run tr_* jobs.
  for pass in 1 2; do
    [[ -n "$picked" ]] && break
    [[ $pass == 1 && -z "$TRAIN_FIRST" ]] && continue
    while IFS='|' read -r name needs cmd; do
      [[ -z "$name" || "$name" == \#* ]] && continue
      [[ -n "$SKIP_TRAIN" && "$name" == tr_* ]] && continue
      [[ $pass == 1 && "$name" != tr_* ]] && continue
      [[ -e $S/$name.done || -e $S/$name.fail || -e $S/$name.lock ]] && continue
      ok=1; for d in ${needs//,/ }; do [[ "$d" == "-" ]] && continue; [ -e $S/$d.done ] || ok=0; done
      (( ok )) || continue
      mkdir $S/$name.lock 2>/dev/null || continue
      picked=$name; pcmd=$cmd; break
    done < $Q
  done
  if [ -z "$picked" ]; then sleep 15; continue; fi
  echo "$(date +%T) $(hostname) start $picked"
  t0=$(date +%s)
  ( source $P/scripts/env_vllm.sh; eval "$pcmd" ) > $RS/logs/$picked.log 2>&1 < /dev/null
  rc=$?
  if (( rc == 0 )); then touch $S/$picked.done; else touch $S/$picked.fail; fi
  echo "$(date +%T) $(hostname) end $picked rc=$rc $(( $(date +%s) - t0 ))s"
done
