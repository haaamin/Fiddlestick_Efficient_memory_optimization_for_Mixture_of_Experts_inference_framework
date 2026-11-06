#!/bin/bash
# vLLM across VRAM budgets. cpu_offload_gb is searched at each budget for the
# setting that keeps as many weights resident as the budget allows: holding it
# fixed would let the added budget be absorbed by the key-value cache instead,
# which understates what vLLM can do. Produces data/vllm/budget.
set -u
. "$(dirname "$0")/env.sh"
for i in $(seq 1 8640); do [ -f $MARK.done ] && break; sleep 10; done
log "waiting for the preceding sweep"
cd /tmp

# sweep <tag> <model> <model size GiB> <budgets...>
sweep(){ local tag=$1 model=$2 size=$3; shift 3
  for gb in "$@"; do
    local lab=${tag}_${gb}GB
    [ -f $MK/$lab.done ] && { log "skip $lab"; continue; }
    local best=""
    # A smaller offload keeps more weights resident; take the smallest that starts.
    for m in 4 8 14 22 30; do
      local off=$(( size - gb + m ))
      [ $off -ge $size ] && off=$(( size - 1 ))
      [ $off -lt 0 ] && off=0
      log "  probe $lab offload=${off}GB (m=$m)"
      if $NUMA timeout 3600 $V/bin/python $B --model "$model" --budget_gib $gb \
           --cpu_offload_gb $off --smoke > $O/${lab}_probe${m}.log 2>&1; then
        best=$off; log "  -> using offload=${off}GB"; break
      fi
    done
    [ -z "$best" ] && { log "  $lab: no setting starts at this budget"; continue; }
    log ">>> $lab measuring (offload=${best}GB)"
    $NUMA timeout 28800 $V/bin/python $B --model "$model" --budget_gib $gb \
      --cpu_offload_gb $best --n_sample 10 --warmup 2 \
      --out $O/$lab.jsonl > $O/$lab.log 2>&1 && { touch $MK/$lab.done; echo "$best" > $O/$lab.off; }
    log "<<< $lab exit=$?"
    grep -E "^  in=" $O/$lab.log | tail -2
  done
}
sweep x7b mistralai/Mixtral-8x7B-v0.1  87  8 16 24 32 40
sweep c22 mistralai/Mixtral-8x22B-v0.1 262 8 16 24 32 40
log "########## vllm_fair DONE ##########"
touch $MARK.done
