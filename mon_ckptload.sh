#!/bin/bash
CTX=gke_cloud-tpu-multipod-dev_us-central1_bodaborg-super-tpu7x-y6k
POD=siv-cn-ckptload1-slice-job-0-0
LOG=/tmp/ckptload1.log
for i in $(seq 1 200); do
  ST=$(kubectl --context $CTX get pod ${POD}-* -o jsonpath='{.items[0].status.phase}' 2>/dev/null)
  if [ "$ST" = "Running" ]; then
    kubectl --context $CTX logs -l jobset.sigs.k8s.io/jobset-name=siv-cn-ckptload1 --prefix=false --timestamps=true --tail=-1 -f --max-log-requests=1 2>&1 | tee -a $LOG
    break
  fi
  echo "poll $i: phase=$ST" ; sleep 30
done
