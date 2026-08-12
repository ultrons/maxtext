#!/bin/bash
# Monitor siv-chengnuo-8x8x8 formation + step times. TPS/chip = 8192 / step_s (8x8x8, pdbs=1).
CTX=gke_cloud-tpu-multipod-dev_us-central1_bodaborg-super-tpu7x-y6k
NAME=siv-chengnuo-8x8x8
LOG=/tmp/chengnuo.log
: > $LOG
for i in $(seq 1 90); do
  # capture logs (append)
  kubectl --context $CTX logs -l jobset.sigs.k8s.io/jobset-name=$NAME --prefix=true --tail=200 --max-log-requests=20 2>/dev/null \
    | grep -E "completed step|Error|Traceback|error|OOM|NaN" >> $LOG 2>/dev/null
  # pods ready?
  READY=$(kubectl --context $CTX get pods -l jobset.sigs.k8s.io/jobset-name=$NAME --no-headers 2>/dev/null | grep -c Running)
  # parse step times (steps 6..18, skip profiler window 5-7)
  MEAN=$(grep -oE "completed step: [0-9]+, seconds: [0-9.]+" $LOG | awk -F'[:,]' '{s=$2+0; t=$4+0; if(s>=8 && s<=18){sum+=t; n++}} END{if(n>0) printf "%.3f", sum/n; else print "NA"}')
  N=$(grep -oE "completed step: [0-9]+, seconds: [0-9.]+" $LOG | awk -F'[:,]' '{s=$2+0; if(s>=8&&s<=18)n++} END{print n+0}')
  if [ "$MEAN" != "NA" ]; then
    TPS=$(awk "BEGIN{printf \"%.0f\", 8192/$MEAN}")
    echo "iter $i: ready=$READY n=$N mean=${MEAN}s TPS/chip=$TPS"
  else
    echo "iter $i: ready=$READY n=$N mean=NA"
  fi
  # done condition: 10+ clean steps
  if [ "$N" -ge 10 ]; then
    echo ">>> chengnuo-8x8x8 DONE mean(8-18)=${MEAN}s / ${TPS} TPS/chip"
    break
  fi
  sleep 30
done
echo "--- last errors (if any) ---"
grep -iE "Error|Traceback|OOM|NaN" $LOG | tail -5