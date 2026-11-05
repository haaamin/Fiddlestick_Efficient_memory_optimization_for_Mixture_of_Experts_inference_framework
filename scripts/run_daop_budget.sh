#!/bin/bash
# DAOP across VRAM budgets, 8 to 40 GB. DAOP runs its published algorithm; the
# patch in patches/ adds only instrumentation. The script refuses to start
# unless DAOP's own brute-force expert partition is present in both model
# files, so that an accidentally modified checkout cannot be measured.
# Produces data/daop/daop_orig.
set -u
. "$(dirname "$0")/env.sh"
export HF_HOME=$HF_HOME CUDA_VISIBLE_DEVICES=0
MK=$MARK; mkdir -p $MK
OD=$OUT/daop/daop_orig; mkdir -p $OD
pg(){ "$PY" -c "print(f'{$1*1024/81079*0.95:.4f}')"; }
X=mistralai/Mixtral-8x7B-v0.1; C=mistralai/Mixtral-8x22B-v0.1

# Refuse to run unless DAOP's own brute-force expert partition is present.
n=$(grep -c "for config in range(1 << self.n_expert)" $D/model/modeling_daop.py)
[ "$n" = "2" ] || { log "ABORT: found $n brute-force blocks, expected 2"; exit 1; }
log "brute-force partition verified"

cd $D; export PYTHONPATH=$D
dsweep(){ local tag=$1 model=$2 cal=$3; shift 3
  read LC LG < <(ck $cal)
  for gb in "$@"; do
    local lab=${tag}_${gb}GB
    [ -f $MK/$lab.done ] && { log "skip $lab"; continue; }
    log ">>> DAOP $lab"
    FS_LATENCY_CPU=$LC FS_LATENCY_GPU=$LG DAOP_INPUTS=16,32,64 DAOP_OUTPUTS=2,4,8,16 \
    $NUMA timeout 28800 "$PY" DAOP/daop_infer.py \
      --model "$model" --dataset_name sharegpt --proportion_gpu $(pg $gb) \
      --num_samples 10 > $OD/$lab.log 2>&1 && touch $MK/$lab.done
    log "<<< $lab exit=$?  $(grep -oE 'PEAK_MIB [0-9]+' $OD/$lab.log | tail -1)"
  done
}
dsweep x7b "$X" d34_32     8 16 24 32 40
dsweep c22 "$C" d34_22b_32 16 24 32 40
log "########## daop_orig DONE ##########"
touch $MARK.done
