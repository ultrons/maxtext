#!/bin/bash
CTX=gke_cloud-tpu-multipod-dev_us-central1_bodaborg-super-tpu7x-y6k
VARIANTS="base noqk nodrs nodtag nodcag nockr nospho"
SUM=/mnt/disks/scratch/maxtext-1410-upstream/ab_results.txt
for iter in $(seq 1 50); do
  done_count=0
  : > $SUM
  for v in $VARIANTS; do
    L=$(kubectl --context $CTX get pods -n default -l jobset.sigs.k8s.io/jobset-name=siv-ab-$v --no-headers 2>/dev/null | awk '$1 ~ /slice-job-0-0-/{print $1; exit}')
    if [ -z "$L" ]; then echo "$v: (forming)" >> $SUM; continue; fi
    LG=$(kubectl --context $CTX logs "$L" -n default 2>/dev/null)
    FAIL=$(printf '%s' "$LG" | grep -oE 'TearDownMesh|STOP_SESSION|RESOURCE_EXHAUSTED|ValueError|missing [0-9]+ required|Aborted' | head -1)
    N=$(printf '%s' "$LG" | grep -c 'completed step')
    N=${N:-0}
    M=$(printf '%s' "$LG" | grep 'completed step' | grep -oE 'step: [0-9]+, seconds: [0-9.]+' | sort -u | awk -F'[:,]' '{s=$2+0;sec=$4+0; if(s>=3&&s<=9){sum+=sec;n++}} END{if(n)printf "%.3f",sum/n; else printf "NA"}')
    if [ -n "$FAIL" ]; then echo "$v: FAILED($FAIL)" >> $SUM; done_count=$((done_count+1)); continue; fi
    if [ "$N" -ge 11 ]; then echo "$v: DONE mean(3-9)=$M [n=$N]" >> $SUM; done_count=$((done_count+1)); else echo "$v: running n=$N mean-so-far=$M" >> $SUM; fi
  done
  echo "===== iter $iter : $done_count/7 done ====="; cat $SUM
  if [ "$done_count" -ge 7 ]; then echo "ALL DONE"; break; fi
  sleep 45
done
