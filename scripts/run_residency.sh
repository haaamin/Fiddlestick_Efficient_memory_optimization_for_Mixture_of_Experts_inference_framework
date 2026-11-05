#!/bin/bash
# 메모리 측정 기준 통일: 모든 시스템을 cudaMemGetInfo(total-free) 기준으로 다시 잰다.
# 기존 init_peak_mib(torch 할당분만)와 vLLM 보고값(컨텍스트 포함)은 축이 달랐다.
set -u
FS=$FS; D=$DAOP
DS=$FS/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json
log(){ echo "[$(TZ=Asia/Seoul date '+%F %H:%M:%S')] $*"; }
export HF_HOME=$HF_HOME CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 FS_NUM_THREADS=32
NUMA="numactl --physcpubind=32-63 --membind=1"
MK=$SCRATCH/memu; mkdir -p $MK
O=$FS/results/mem_unified; mkdir -p $O
OD=$WORK/daop_memu; mkdir -p $OD
ck(){ $FS/venv/bin/python -c "
import json; d=json.load(open('$FS/calib/$1.json'))
print(d['cpu_ms_effective'], d['h2d_pageable_ms'])"; }
pg(){ $FS/venv/bin/python -c "print(f'{16*1024/81079*0.95:.4f}')"; }

log "########## 우리 시스템 (expert 0개 강제) ##########"
ours(){ local lab=$1 sys=$2 model=$3 cal=$4 pop=$5
  [ -f $MK/$lab.done ] && { log "skip $lab"; return 0; }
  read LC LG < <(ck $cal); cd $FS
  log ">>> $lab"
  FS_FORCE_EXPERTS=0 FS_MEM_PERCENT=0.5052 FS_LATENCY_CPU=$LC FS_LATENCY_GPU=$LG \
  FS_POPULARITY_JSON=$FS/$pop \
  $NUMA timeout 7200 ./venv/bin/python benchmarks/run_bench.py \
    --system $sys --model "$model" --beam_widths 1 --n_sample 1 \
    --input_tokens 16 --output_tokens 2 --dataset $DS \
    --out $O/$lab.jsonl > $O/$lab.log 2>&1 && touch $MK/$lab.done
  ./venv/bin/python -c "
import json
for ln in open('$O/$lab.jsonl'):
    r=json.loads(ln)
    if r.get('record')=='meta':
        print(f\"    nexp={r.get('n_expert_on_gpu')}  torch_peak={r.get('init_peak_mib')} MiB  cuda_used={r.get('cuda_used_mib')} MiB\"); break" 2>/dev/null
}
X=mistralai/Mixtral-8x7B-v0.1; C=mistralai/Mixtral-8x22B-v0.1; P=$SCRATCH/phimoe
ours x7b_fd fiddler         "$X" d34_32     popularity_8x7b.json
ours x7b_fs fiddlestick     "$X" d34_32     popularity_8x7b.json
ours phi_fd fiddler_phi     "$P" d34_phi_32 popularity_phimoe.json
ours phi_fs fiddlestick_phi "$P" d34_phi_32 popularity_phimoe.json
ours c22_fd fiddler         "$C" d34_22b_32 popularity_8x22b.json
ours c22_fs fiddlestick     "$C" d34_22b_32 popularity_8x22b.json

log "########## DAOP (캐시 0개) ##########"
cd $D; export PYTHONPATH=$D
dres(){ local lab=$1 model=$2 cal=$3
  [ -f $MK/d_$lab.done ] && { log "skip d_$lab"; return 0; }
  read LC LG < <(ck $cal)
  log ">>> DAOP $lab"
  FS_LATENCY_CPU=$LC FS_LATENCY_GPU=$LG DAOP_INPUTS=16 DAOP_OUTPUTS=2 \
  $NUMA timeout 7200 ./venv/bin/python DAOP/daop_infer.py \
    --model "$model" --dataset_name sharegpt --proportion_gpu 0.0001 \
    --num_samples 1 > $OD/$lab.log 2>&1 && touch $MK/d_$lab.done
  grep -oE "PEAK_MIB [0-9]+|CUDA_USED_MIB [0-9]+" $OD/$lab.log | tail -2 | tr '\n' ' '; echo
}
dres x7b "$X" d34_32
dres phi $PHI_MODEL d34_phi_32
dres c22 "$C" d34_22b_32
log "########## 통일 측정 DONE ##########"
touch $SCRATCH/memu.done
