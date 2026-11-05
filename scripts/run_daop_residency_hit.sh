#!/bin/bash
# DAOP expert cache hit rate on the calibration corpus (ShareGPT) and the
# held-out corpus (C4), plus its GPU occupancy at a 16 GB budget. Runs after
# run_daop_budget.sh. Produces data/daop/daop_orig_hit.
set -u
D=$DAOP; FS=$FS
log(){ echo "[$(TZ=Asia/Seoul date '+%F %H:%M:%S')] $*"; }
export HF_HOME=$HF_HOME CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
NUMA="numactl --physcpubind=32-63 --membind=1"
MK=$SCRATCH/dorig2; mkdir -p $MK
OM=$WORK/daop_orig_mem; mkdir -p $OM
OH=$WORK/daop_orig_hit; mkdir -p $OH
pg(){ $FS/venv/bin/python -c "print(f'{$1*1024/81079*0.95:.4f}')"; }
ck(){ $FS/venv/bin/python -c "
import json; d=json.load(open('$FS/calib/$1.json'))
print(d['cpu_ms_effective'], d['h2d_pageable_ms'])"; }
X=mistralai/Mixtral-8x7B-v0.1; C=mistralai/Mixtral-8x22B-v0.1

# Wait for the budget sweep to finish (up to four hours).
for i in $(seq 1 480); do [ -f $SCRATCH/dorig.done ] && break; sleep 30; done
[ -f $SCRATCH/dorig.done ] || { log "ABORT: the budget sweep has not finished"; exit 1; }

n=$(grep -c "for config in range(1 << self.n_expert)" $D/model/modeling_daop.py)
[ "$n" = "2" ] || { log "ABORT: found $n brute-force blocks, expected 2"; exit 1; }
log "brute-force partition verified"

cd $D; export PYTHONPATH=$D

log "########## DAOP GPU occupancy at 16 GB ##########"
dact(){ local lab=$1 model=$2 cal=$3
  [ -f $MK/m_$lab.done ] && { log "skip m_$lab"; return 0; }
  read LC LG < <(ck $cal)
  log ">>> mem $lab"
  FS_LATENCY_CPU=$LC FS_LATENCY_GPU=$LG DAOP_INPUTS=16,64 DAOP_OUTPUTS=2,16 \
  $NUMA timeout 7200 ./venv/bin/python DAOP/daop_infer.py \
    --model "$model" --dataset_name sharegpt --proportion_gpu $(pg 16) \
    --num_samples 2 > $OM/$lab.log 2>&1 && touch $MK/m_$lab.done
  grep -oE "PEAK_MIB [0-9]+|CUDA_USED_MIB [0-9]+|Number of experts on GPU: [0-9]+/[0-9]+" $OM/$lab.log | tail -3 | tr '\n' ' '; echo
}
dact x7b "$X" d34_32
dact c22 "$C" d34_22b_32

log "########## DAOP hit rate (ShareGPT / C4) ##########"
hit(){ local lab=$1 model=$2 cal=$3 ds=$4
  [ -f $MK/h_$lab.done ] && { log "skip h_$lab"; return 0; }
  read LC LG < <(ck $cal)
  log ">>> hit $lab"
  FS_LATENCY_CPU=$LC FS_LATENCY_GPU=$LG DAOP_INPUTS=16,32,64 DAOP_OUTPUTS=2,4,8,16 \
  $NUMA timeout 28800 ./venv/bin/python DAOP/daop_infer.py \
    --model "$model" --dataset_name $ds --proportion_gpu $(pg 16) \
    --num_samples 10 > $OH/$lab.log 2>&1 && touch $MK/h_$lab.done
  grep -oE "HIT_RATE [0-9.]+|Number of experts on GPU: [0-9]+/[0-9]+" $OH/$lab.log | tail -3 | tr '\n' ' '; echo
}
for ds in sharegpt c4; do
  hit x7b_$ds "$X" d34_32     $ds
  hit c22_$ds "$C" d34_22b_32 $ds
done
log "########## daop_orig2 DONE ##########"
touch $SCRATCH/dorig2.done
