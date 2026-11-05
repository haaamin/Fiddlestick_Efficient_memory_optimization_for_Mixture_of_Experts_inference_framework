#!/bin/bash
# The Fiddler implementation bundled in the DAOP repository, measured under the
# same conditions as everything else (16 GB, twelve cells, ten samples). DAOP
# is built on this copy, so its throughput is the baseline against which DAOP's
# own contribution should be read. Produces data/daop/daop_orig_fdrepo.
set -u
. "$(dirname "$0")/env.sh"
export HF_HOME=$HF_HOME CUDA_VISIBLE_DEVICES=0
MK=$MARK; mkdir -p $MK
O=$OUT/daop/daop_orig_fdrepo; mkdir -p $O
pg(){ "$PY" -c "print(f'{16*1024/81079*0.95:.4f}')"; }
X=mistralai/Mixtral-8x7B-v0.1; C=mistralai/Mixtral-8x22B-v0.1

for i in $(seq 1 960); do [ -f $MARK.done ] && break; sleep 30; done
[ -f $MARK.done ] || { log "ABORT: the hit-rate run has not finished"; exit 1; }
n=$(grep -c "for config in range(1 << self.n_expert)" $D/model/modeling_fiddler.py)
[ "$n" = "2" ] || { log "ABORT: modeling_fiddler has $n brute-force blocks, expected 2"; exit 1; }
log "measuring the Fiddler bundled with DAOP"

cd $D; export PYTHONPATH=$D
run(){ local lab=$1 model=$2 cal=$3
  [ -f $MK/$lab.done ] && { log "skip $lab"; return 0; }
  read LC LG < <(ck $cal)
  log ">>> $lab"
  FS_LATENCY_CPU=$LC FS_LATENCY_GPU=$LG DAOP_INPUTS=16,32,64 DAOP_OUTPUTS=2,4,8,16 \
  $NUMA timeout 28800 "$PY" fiddler/fiddler_infer.py \
    --model "$model" --dataset_name sharegpt --cpu_offload 1 \
    --proportion_gpu $(pg) --num_samples 10 > $O/$lab.log 2>&1 && touch $MK/$lab.done
  log "<<< $lab exit=$?"
}
run x7b "$X" d34_32
run c22 "$C" d34_22b_32
log "########## daop_orig3 DONE ##########"
touch $MARK.done
