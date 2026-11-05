#!/bin/bash
# DAOP 예산축 재측정 — 2^n 전수탐색(원본) 복원 상태에서 재실행.
# argmin 치환을 되돌렸으므로 Fig3(16GB)·Fig6(예산 스윕)이 모두 원본 코드 한 배치로 통일된다.
# Phi-3.5-MoE 는 논문에서 제외되어 돌리지 않는다.
set -u
FS=$FS
D=$DAOP
log(){ echo "[$(TZ=Asia/Seoul date '+%F %H:%M:%S')] $*"; }
export HF_HOME=$HF_HOME CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
NUMA="numactl --physcpubind=32-63 --membind=1"
MK=$SCRATCH/dorig; mkdir -p $MK
OD=$WORK/daop_orig; mkdir -p $OD
pg(){ $FS/venv/bin/python -c "print(f'{$1*1024/81079*0.95:.4f}')"; }
ck(){ $FS/venv/bin/python -c "
import json; d=json.load(open('$FS/calib/$1.json'))
print(d['cpu_ms_effective'], d['h2d_pageable_ms'])"; }
X=mistralai/Mixtral-8x7B-v0.1; C=mistralai/Mixtral-8x22B-v0.1

# 되돌리기 검증: 전수탐색이 실제로 복원됐는지 확인하고, 아니면 중단
n=$(grep -c "for config in range(1 << self.n_expert)" $D/model/modeling_daop.py)
[ "$n" = "2" ] || { log "ABORT: 전수탐색 블록 $n 개 (2 이어야 함)"; exit 1; }
log "전수탐색 복원 확인 OK"

cd $D; export PYTHONPATH=$D
dsweep(){ local tag=$1 model=$2 cal=$3; shift 3
  read LC LG < <(ck $cal)
  for gb in "$@"; do
    local lab=${tag}_${gb}GB
    [ -f $MK/$lab.done ] && { log "skip $lab"; continue; }
    log ">>> DAOP $lab"
    FS_LATENCY_CPU=$LC FS_LATENCY_GPU=$LG DAOP_INPUTS=16,32,64 DAOP_OUTPUTS=2,4,8,16 \
    $NUMA timeout 28800 ./venv/bin/python DAOP/daop_infer.py \
      --model "$model" --dataset_name sharegpt --proportion_gpu $(pg $gb) \
      --num_samples 10 > $OD/$lab.log 2>&1 && touch $MK/$lab.done
    log "<<< $lab exit=$?  $(grep -oE 'PEAK_MIB [0-9]+' $OD/$lab.log | tail -1)"
  done
}
dsweep x7b "$X" d34_32     8 16 24 32 40
dsweep c22 "$C" d34_22b_32 16 24 32 40
log "########## daop_orig DONE ##########"
touch $SCRATCH/dorig.done
