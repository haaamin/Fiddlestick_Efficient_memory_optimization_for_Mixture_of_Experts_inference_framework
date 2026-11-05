#!/bin/bash
# DAOP minimum GPU residency: proportion_gpu is set low enough that no expert
# is cached, leaving only the non-expert weights resident, and the GPU memory
# the process occupies is recorded. Produces data/daop/daop_orig_floor.
set -u
. "$(dirname "$0")/env.sh"
export HF_HOME=$HF_HOME CUDA_VISIBLE_DEVICES=0
MK=$MARK; mkdir -p $MK
O=$OUT/daop/daop_orig_floor; mkdir -p $O
n=$(grep -c "for config in range(1 << self.n_expert)" $D/model/modeling_daop.py)
[ "$n" = "2" ] || { log "ABORT: found $n brute-force blocks, expected 2"; exit 1; }
cd $D; export PYTHONPATH=$D
res(){ local lab=$1 model=$2 cal=$3
  [ -f $MK/$lab.done ] && { log "skip $lab"; return; }
  read LC LG < <(ck $cal)
  log ">>> residency $lab"
  FS_LATENCY_CPU=$LC FS_LATENCY_GPU=$LG DAOP_INPUTS=16 DAOP_OUTPUTS=2 \
  $NUMA timeout 7200 "$PY" DAOP/daop_infer.py \
    --model "$model" --dataset_name sharegpt --proportion_gpu 0.0001 \
    --num_samples 1 > $O/$lab.log 2>&1 && touch $MK/$lab.done
  grep -oE 'PEAK_MIB [0-9]+|CUDA_USED_MIB [0-9]+|Number of experts on GPU: [0-9]+/[0-9]+' $O/$lab.log | tail -3 | tr '\n' ' '; echo
}
res x7b mistralai/Mixtral-8x7B-v0.1  d34_32
res c22 mistralai/Mixtral-8x22B-v0.1 d34_22b_32
log "########## daop_orig4 DONE ##########"
touch $MARK.done
