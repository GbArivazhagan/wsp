#!/bin/bash
# numword queue worker: claims jobs from jobs.txt with mkdir locks; runs SLOTS jobs at a time on this host.
# usage: worker.sh SLOTS GM      (GM = vLLM gpu_memory_utilization per job)
cd ${WSP_ROOT:-$PWD}
SLOTS=${1:-1}; export GM=${2:-0.4}; Q=numword/queue; H=$(hostname -s)
mkdir -p $Q/locks $Q/done
source scripts/env_hf.sh > /dev/null; cd ${WSP_ROOT:-$PWD}
declare -A RUN
while true; do
  for n in "${!RUN[@]}"; do kill -0 ${RUN[$n]} 2>/dev/null || unset RUN[$n]; done
  left=0
  while IFS='|' read -r name pre cmd; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    [ -e $Q/done/$name ] && continue
    left=1
    [ -d $Q/locks/$name ] && continue
    (( ${#RUN[@]} >= SLOTS )) && break
    bash -c "$pre" 2>/dev/null || continue
    mkdir $Q/locks/$name 2>/dev/null || continue
    echo "$(date +%T) $H start $name"
    ( bash -c "$cmd" > numword/logs/job_$name.log 2>&1; rc=$?
      if [ $rc -eq 0 ]; then touch $Q/done/$name; else rmdir $Q/locks/$name; fi
      echo "$(date +%T) $H end $name rc=$rc" ) &
    RUN[$name]=$!
    sleep 15
  done < $Q/jobs.txt
  (( left == 0 && ${#RUN[@]} == 0 )) && { echo "$(date +%T) $H queue empty"; exit 0; }
  sleep 30
done
