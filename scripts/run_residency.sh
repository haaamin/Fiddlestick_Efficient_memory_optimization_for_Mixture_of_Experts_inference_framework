#!/bin/bash
# Minimum GPU residency for every system on one basis. The expert cache is
# forced to zero and the GPU memory the process occupies is read from
# cudaMemGetInfo as total minus free, which includes the CUDA context and
# allocations that bypass the PyTorch allocator. Produces data/ours/residency.
set -u
. "$(dirname "$0")/env.sh"
MK=$MARK; mkdir -p $MK
# build_data.py reads these two directories by name; keep them in step.
O=$OUT/ours/residency; mkdir -p $O
OD=$OUT/daop/daop_orig_floor; mkdir -p $OD
pg(){ "$PY" -c "print(f'{16*1024/81079*0.95:.4f}')"; }

log "########## Fiddler and Fiddlestick, cache forced to zero ##########"
ours(){ local lab=$1 sys=$2 model=$3 cal=$4 pop=$5
  [ -f $MK/$lab.done ] && { log "skip $lab"; return 0; }
  read LC LG < <(consts $cal); cd $ROOT
  log ">>> $lab"
  FS_FORCE_EXPERTS=0 FS_MEM_PERCENT=0.5052 FS_LATENCY_CPU=$LC FS_LATENCY_GPU=$LG \
  FS_POPULARITY_JSON=$pop \
  $NUMA timeout 7200 "$PY" "$BENCH" \
    --system $sys --model "$model" --beam_widths 1 --n_sample 1 \
    --input_tokens 16 --output_tokens 2 --dataset $DS \
    --out $O/$lab.jsonl > $O/$lab.log 2>&1 && touch $MK/$lab.done
  "$PY" -c "
import json
for ln in open('$O/$lab.jsonl'):
    r=json.loads(ln)
    if r.get('record')=='meta':
        print(f\"    nexp={r.get('n_expert_on_gpu')}  torch_peak={r.get('init_peak_mib')} MiB  cuda_used={r.get('cuda_used_mib')} MiB\"); break"
}
X=mistralai/Mixtral-8x7B-v0.1; C=mistralai/Mixtral-8x22B-v0.1; P=$MARK
ours x7b_fd fiddler         "$X" d34_32     $POP/popularity_8x7b.json
ours x7b_fs fiddlestick     "$X" d34_32     $POP/popularity_8x7b.json
ours c22_fd fiddler         "$C" d34_22b_32 $POP/popularity_8x22b.json
ours c22_fs fiddlestick     "$C" d34_22b_32 $POP/popularity_8x22b.json

log "########## DAOP, cache forced to zero ##########"
cd $D; export PYTHONPATH=$D
dres(){ local lab=$1 model=$2 cal=$3
  [ -f $MK/d_$lab.done ] && { log "skip d_$lab"; return 0; }
  read LC LG < <(consts $cal)
  log ">>> DAOP $lab"
  FS_LATENCY_CPU=$LC FS_LATENCY_GPU=$LG DAOP_INPUTS=16 DAOP_OUTPUTS=2 \
  $NUMA timeout 7200 "$PY" DAOP/daop_infer.py \
    --model "$model" --dataset_name sharegpt --proportion_gpu 0.0001 \
    --num_samples 1 > $OD/$lab.log 2>&1 && touch $MK/d_$lab.done
  grep -oE "PEAK_MIB [0-9]+|CUDA_USED_MIB [0-9]+" $OD/$lab.log | tail -2 | tr '\n' ' '; echo
}
dres x7b "$X" d34_32
dres c22 "$C" d34_22b_32
log "########## done ##########"
touch $MARK.done
