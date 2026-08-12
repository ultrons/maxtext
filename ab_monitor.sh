#!/bin/bash
# Collect steady-state s/step for all 7 A/B variants concurrently.
CTX=gke_cloud-tpu-multipod-dev_us-central1_bodaborg-super-tpu7x-y6k
VARIANTS="base noqk nodrs nodtag nodcag nockr nospho"
SUM=/mnt/disks/scratch/maxtext-1410-upstream/ab_results.txt

for iter in $(seq 1 60); do
  done_count=0
  : > $SUM
  for v in $VARIANTS; do
    L=$(kubectl --context $CTX get pods -n default -l jobset.sigs.k8s.io/jobset-name=siv-ab-$v --field-selector status.phase=Running --no-headers 2>/dev/null | awk '$1 ~ /slice-job-0-0-/{print $1;exit}')
    if [ -z "$L" ]; then
      # maybe completed - grab any pod
      L=$(kubectl --context $CTX get pods -n default -l jobset.sigs.k8s.io/jobset-name=siv-ab-$v --no-headers 2>/dev/null | awk '$1 ~ /slice-job-0-0-/{print $1;exit}')
    fi
    [ -z "$L" ] && { echo "$v: (no leader yet)" >> $SUM; continue; }
    STEPS=$(kubectl --context $CTX logs $L -n default 2>/dev/null | grep 'completed step' | grep -oE 'step: [0-9]+, seconds: [0-9.]+')
    N=$(echo "$STEPS" | grep -c 'step:')
    FAIL=$(kubectl --context $CTX logs $L -n default 2>/dev/null | grep -oE 'TearDownMesh|STOP_SESSION|RESOURCE_EXHAUSTED|ValueError|missing [0-9]+ required' | head -1)
    if [ -n "$FAIL" ]; then echo "$v: FAILED ($FAIL)" >> $SUM; done_count=$((done_count+1)); continue; fi
    MEAN=$(echo "$STEPS" | sort -u | awk -F'[:,]' '{s=$2+0;sec=$4+0; if(s>=3&&s<=9){sum+=sec;n++}} END{if(n)printf "%.3f",sum/n; else printf "NA"}')
    if [ "$N" -ge 10 ]; then echo "$v: mean(3-9)=$MEAN s/step  [n=$N]" >> $SUM; done_count=$((done_count+1));
    else echo "$v: steps=$N (compiling/running) mean-so-far=$MEAN" >> $SUM; fi
  done
  echo "--- iter $iter ($done_count/7 done) ---"; cat $SUM
  [ "$done_count" -ge 7 ] && { echo "ALL DONE"; break; }
  sleep 40
done
