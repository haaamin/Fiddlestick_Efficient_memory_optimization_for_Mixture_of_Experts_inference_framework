#!/bin/bash
# 8x7B 예산 스윕 재측정: 8/16/24/32/40GB 를 하나의 배치로.
# 기존 Fig6 의 8x7B 곡선은 16~40GB(uni2, 10-22)와 8GB(v3, 10-26)가 섞여 있었다.
# uni2_x7b_8GB 의 Fiddlestick 이 0.431 tok/s 로 오염되어 그 점만 재측정했던 탓이다.
set -u
FS=$FS
DS=$FS/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json
log(){ echo "[$(TZ=Asia/Seoul date '+%F %H:%M:%S')] $*"; }
export CUDA_VISIBLE_DEVICES=0 HF_HOME=$HF_HOME
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 FS_NUM_THREADS=32
NUMA="numactl --physcpubind=32-63 --membind=1"
MK=$SCRATCH/v5x7b; mkdir -p $MK
cd $FS
pct(){ ./venv/bin/python -c "print(f'{$1*1024/81079:.4f}')"; }
consts(){ ./venv/bin/python -c "
import json; d=json.load(open('calib/d34_32.json'))
print(d['cpu_ms_effective'], d['h2d_pageable_ms'])"; }
read LC LG < <(consts)
X=mistralai/Mixtral-8x7B-v0.1
for g in 8 16 24 32 40; do
  D=$FS/results/v5_x7b_${g}GB; mkdir -p $D
  for s in fiddler fiddlestick; do
    mk=v5_x7b_${g}GB_$s
    [ -f $MK/$mk.done ] && { log "skip $mk"; continue; }
    log ">>> $mk (load1m=$(cut -d' ' -f1 /proc/loadavg))"
    FS_MEM_PERCENT=$(pct $g) FS_LATENCY_CPU=$LC FS_LATENCY_GPU=$LG \
    FS_POPULARITY_JSON=$FS/popularity_8x7b.json \
    $NUMA timeout 28800 ./venv/bin/python benchmarks/run_bench.py \
      --system $s --model "$X" --beam_widths 1 --n_sample 10 \
      --input_tokens 16,32,64 --output_tokens 2,4,8,16 \
      --dataset $DS --out $D/$s.jsonl > $D/$s.log 2>&1 && touch $MK/$mk.done
    log "<<< $mk exit=$?"
  done
done
log "########## v5 x7b budget DONE ##########"
touch $SCRATCH/v5x7b.done
