#!/bin/bash
# $1 = workload name. Dedup steps by number; report steady-state mean(10-19) + loss + errors.
CTX=gke_cloud-tpu-multipod-dev_us-central1_bodaborg-super-tpu7x-y6k
NAME="$1"
RAW=/tmp/${NAME}.raw
for i in $(seq 1 60); do
  kubectl --context $CTX logs -l jobset.sigs.k8s.io/jobset-name=$NAME --tail=400 --max-log-requests=4 2>/dev/null \
    | grep -E "completed step|Error|Traceback|NaN|nan|exceed|RESOURCE_EXHAUSTED" > $RAW.tmp 2>/dev/null
  [ -s $RAW.tmp ] && mv $RAW.tmp $RAW
  READY=$(kubectl --context $CTX get pods -l jobset.sigs.k8s.io/jobset-name=$NAME --no-headers 2>/dev/null | grep -c Running)
  # dedup completed-step lines by step number
  STEPS=$(grep -oE "completed step: [0-9]+, seconds: [0-9.]+" $RAW 2>/dev/null | sort -t: -k2 -n -u)
  MAXSTEP=$(echo "$STEPS" | grep -oE "step: [0-9]+" | grep -oE "[0-9]+" | sort -n | tail -1)
  MEAN=$(echo "$STEPS" | awk -F'[:,]' '{s=$2+0;t=$4+0; if(s>=10&&s<=19){sum+=t;n++}} END{if(n>0)printf "%.3f",sum/n; else print "NA"}')
  N=$(echo "$STEPS" | awk -F'[:,]' '{s=$2+0; if(s>=10&&s<=19)n++} END{print n+0}')
  ERR=$(grep -icE "Traceback|RESOURCE_EXHAUSTED|nan|exceed" $RAW 2>/dev/null)
  if [ "$MEAN" != "NA" ] && [ "$N" != "0" ]; then
    TPS=$(awk "BEGIN{printf \"%.0f\", 8192/$MEAN}")
    echo "iter $i: ready=$READY maxstep=${MAXSTEP:-0} n=$N mean(10-19)=${MEAN}s TPS/chip=$TPS err=$ERR"
  else
    echo "iter $i: ready=$READY maxstep=${MAXSTEP:-0} n=$N mean=NA err=$ERR"
  fi
  if [ "$N" -ge 10 ]; then echo ">>> $NAME DONE mean=${MEAN}s / ${TPS} TPS/chip"; break; fi
  if [ "$ERR" -gt 0 ] 2>/dev/null && [ "${MAXSTEP:-0}" -lt 3 ] 2>/dev/null; then echo ">>> $NAME ERROR pre-step3"; grep -iE "Traceback|RESOURCE_EXHAUSTED|exceed" $RAW | tail -3; break; fi
  sleep 30
done
echo "--- loss trajectory (dedup) ---"
kubectl --context $CTX logs -l jobset.sigs.k8s.io/jobset-name=$NAME --tail=400 --max-log-requests=4 2>/dev/null \
  | grep -oE "completed step: [0-9]+, seconds: [0-9.]+, loss: [0-9.eE+-]+" | sort -t: -k2 -n -u | tail -20
