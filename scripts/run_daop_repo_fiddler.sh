#!/bin/bash
# DAOP 3차: DAOP 저장소에 동봉된 Fiddler 구현을 통일 조건(16GB, 12셀, n=10)에서 측정.
# DAOP 의 절대 처리량을 우리 Fiddler 와 직접 비교할 때, 두 저장소의 Fiddler 구현
# 성능 차이가 섞이는지 확인하기 위한 것이다. 원본 전수탐색 코드 상태에서 실행한다.
set -u
D=$DAOP; FS=$FS
log(){ echo "[$(TZ=Asia/Seoul date '+%F %H:%M:%S')] $*"; }
export HF_HOME=$HF_HOME CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
NUMA="numactl --physcpubind=32-63 --membind=1"
MK=$SCRATCH/dorig3; mkdir -p $MK
O=$WORK/daop_orig_fdrepo; mkdir -p $O
pg(){ $FS/venv/bin/python -c "print(f'{16*1024/81079*0.95:.4f}')"; }
ck(){ $FS/venv/bin/python -c "
import json; d=json.load(open('$FS/calib/$1.json'))
print(d['cpu_ms_effective'], d['h2d_pageable_ms'])"; }
X=mistralai/Mixtral-8x7B-v0.1; C=mistralai/Mixtral-8x22B-v0.1

for i in $(seq 1 960); do [ -f $SCRATCH/dorig2.done ] && break; sleep 30; done
[ -f $SCRATCH/dorig2.done ] || { log "ABORT: 2차 미완료"; exit 1; }
n=$(grep -c "for config in range(1 << self.n_expert)" $D/model/modeling_fiddler.py)
[ "$n" = "2" ] || { log "ABORT: modeling_fiddler 전수탐색 $n 개"; exit 1; }
log "3차 시작 (DAOP 저장소 Fiddler)"

cd $D; export PYTHONPATH=$D
run(){ local lab=$1 model=$2 cal=$3
  [ -f $MK/$lab.done ] && { log "skip $lab"; return 0; }
  read LC LG < <(ck $cal)
  log ">>> $lab"
  FS_LATENCY_CPU=$LC FS_LATENCY_GPU=$LG DAOP_INPUTS=16,32,64 DAOP_OUTPUTS=2,4,8,16 \
  $NUMA timeout 28800 ./venv/bin/python fiddler/fiddler_infer.py \
    --model "$model" --dataset_name sharegpt --cpu_offload 1 \
    --proportion_gpu $(pg) --num_samples 10 > $O/$lab.log 2>&1 && touch $MK/$lab.done
  log "<<< $lab exit=$?"
}
run x7b "$X" d34_32
run c22 "$C" d34_22b_32
log "########## daop_orig3 DONE ##########"
touch $SCRATCH/dorig3.done
