#!/bin/bash
# vLLM 예산축 재측정.
# 앞선 측정은 cpu_offload_gb 를 모델별로 고정해, 예산을 늘려도 늘어난 몫이 전부
# KV 캐시로 가고 가중치는 CPU 에 남아 처리량이 평탄했다(불공정한 설정).
# 예산마다 "가능한 한 많은 가중치를 GPU 에 두는" offload 값을 탐색해서 다시 잰다.
set -u
V=$VLLM_ENV
B=$WORK/vllm_bench.py
O=$WORK/vllm_budget2; mkdir -p $O
MK=$SCRATCH/vfair; mkdir -p $MK
log(){ echo "[$(TZ=Asia/Seoul date '+%F %H:%M:%S')] $*"; }
export HF_HOME=$HF_HOME CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
NUMA="numactl --physcpubind=32-63 --membind=1"
for i in $(seq 1 8640); do [ -f $SCRATCH/budall.done ] && break; sleep 10; done
log "선행 스윕 종료 확인"
cd /tmp

# sweep <태그> <모델> <모델크기GiB> <예산들...>
sweep(){ local tag=$1 model=$2 size=$3; shift 3
  for gb in "$@"; do
    local lab=${tag}_${gb}GB
    [ -f $MK/$lab.done ] && { log "skip $lab"; continue; }
    local best=""
    # m 이 작을수록 가중치를 많이 상주시킨다. 뜨는 것 중 가장 공격적인 값을 고른다.
    for m in 4 8 14 22 30; do
      local off=$(( size - gb + m ))
      [ $off -ge $size ] && off=$(( size - 1 ))
      [ $off -lt 0 ] && off=0
      log "  probe $lab offload=${off}GB (m=$m)"
      if $NUMA timeout 3600 $V/bin/python $B --model "$model" --budget_gib $gb \
           --cpu_offload_gb $off --smoke > $O/${lab}_probe${m}.log 2>&1; then
        best=$off; log "  -> 채택 offload=${off}GB"; break
      fi
    done
    [ -z "$best" ] && { log "  $lab: 띄울 수 있는 설정 없음"; continue; }
    log ">>> $lab 본측정 (offload=${best}GB)"
    $NUMA timeout 28800 $V/bin/python $B --model "$model" --budget_gib $gb \
      --cpu_offload_gb $best --n_sample 10 --warmup 2 \
      --out $O/$lab.jsonl > $O/$lab.log 2>&1 && { touch $MK/$lab.done; echo "$best" > $O/$lab.off; }
    log "<<< $lab exit=$?"
    grep -E "^  in=" $O/$lab.log | tail -2
  done
}
sweep x7b mistralai/Mixtral-8x7B-v0.1  87  8 16 24 32 40
sweep phi $SCRATCH/phimoe   78  8 16 24 32 40
sweep c22 mistralai/Mixtral-8x22B-v0.1 262 8 16 24 32 40
log "########## vllm_fair DONE ##########"
touch $SCRATCH/vfair.done
