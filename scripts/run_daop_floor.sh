#!/bin/bash
# DAOP 4차: 최소 상주량(expert 캐시 0개) 재측정, 원본 전수탐색 코드.
set -u
D=$DAOP; FS=$FS
log(){ echo "[$(TZ=Asia/Seoul date '+%F %H:%M:%S')] $*"; }
export HF_HOME=$HF_HOME CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
NUMA="numactl --physcpubind=32-63 --membind=1"
MK=$SCRATCH/dorig4; mkdir -p $MK
O=$WORK/daop_orig_floor; mkdir -p $O
ck(){ $FS/venv/bin/python -c "
import json; d=json.load(open('$FS/calib/$1.json'))
print(d['cpu_ms_effective'], d['h2d_pageable_ms'])"; }
n=$(grep -c "for config in range(1 << self.n_expert)" $D/model/modeling_daop.py)
[ "$n" = "2" ] || { log "ABORT: 전수탐색 $n 개"; exit 1; }
cd $D; export PYTHONPATH=$D
res(){ local lab=$1 model=$2 cal=$3
  [ -f $MK/$lab.done ] && { log "skip $lab"; return; }
  read LC LG < <(ck $cal)
  log ">>> 상주량 $lab"
  FS_LATENCY_CPU=$LC FS_LATENCY_GPU=$LG DAOP_INPUTS=16 DAOP_OUTPUTS=2 \
  $NUMA timeout 7200 ./venv/bin/python DAOP/daop_infer.py \
    --model "$model" --dataset_name sharegpt --proportion_gpu 0.0001 \
    --num_samples 1 > $O/$lab.log 2>&1 && touch $MK/$lab.done
  grep -oE 'PEAK_MIB [0-9]+|CUDA_USED_MIB [0-9]+|Number of experts on GPU: [0-9]+/[0-9]+' $O/$lab.log | tail -3 | tr '\n' ' '; echo
}
res x7b mistralai/Mixtral-8x7B-v0.1  d34_32
res c22 mistralai/Mixtral-8x22B-v0.1 d34_22b_32
log "########## daop_orig4 DONE ##########"
touch $SCRATCH/dorig4.done
