#!/bin/bash
# Held-out evaluation: the expert placement is computed from ShareGPT and held
# fixed while the evaluation corpus is changed to C4. Conditions match the rest
# of the evaluation: 16 GB, 32 pinned cores, greedy decoding, ten samples,
# inputs 16/32/64 crossed with outputs 2/4/8/16.
# Produces data/ours/{c4_b1_n10,c4_c22}.
set -u
FS=$FS
D=$DAOP
KT=$WORK/ktransformers; V=$KT/venv
SP=$V/lib/python3.12/site-packages/ktransformers
C4=$FS/benchmarks/c4_sharegptfmt.json
log(){ echo "[$(TZ=Asia/Seoul date '+%F %H:%M:%S')] $*"; }
export CUDA_VISIBLE_DEVICES=0 HF_HOME=$HF_HOME
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
LC=$($FS/venv/bin/python -c "import json;print(json.load(open('$FS/calib/d34_32.json'))['fiddler']['latency_cpu'])")
LG=$($FS/venv/bin/python -c "import json;print(json.load(open('$FS/calib/d34_32.json'))['fiddler']['latency_gpu'])")
NUMA="numactl --physcpubind=32-63 --membind=1"

# ---------- one run of the harness ----------
ours(){ local dir=$1 sys=$2 ds=$3 pop=$4
  mkdir -p $dir
  [ -f $dir/$sys.done ] && { log "skip $dir/$sys"; return; }
  log ">>> $dir / $sys  (pop=$(basename $pop))"
  cd $FS
  FS_NUM_THREADS=32 FS_MEM_PERCENT=0.2021 \
  FS_LATENCY_CPU=$LC FS_LATENCY_GPU=$LG FS_POPULARITY_JSON=$pop \
  $NUMA timeout 7200 ./venv/bin/python benchmarks/run_bench.py \
    --system $sys --beam_widths 1 --n_sample 10 \
    --input_tokens 16,32,64 --output_tokens 2,4,8,16 \
    --dataset $ds --out $dir/$sys.jsonl > $dir/$sys.log 2>&1 && touch $dir/$sys.done
  log "<<< $dir/$sys exit=$?"
}

log "########## held-out corpus (C4) ##########"
O1=$FS/results/c4_b1_n10
ours $O1 fiddlestick $C4 $FS/popularity_8x7b.json
ours $O1 fiddler     $C4 $FS/popularity_8x7b.json

log ">>> C4 / DAOP"
OD=$WORK/daop_c4; mkdir -p $OD
cd $D; export PYTHONPATH=$D
PG=$($FS/venv/bin/python -c "print(f'{16*1024/81079*0.95:.4f}')")
if [ ! -f $OD/A_daop.done ]; then
  FS_LATENCY_CPU=$LC FS_LATENCY_GPU=$LG DAOP_INPUTS=16,32,64 DAOP_OUTPUTS=2,4,8,16 \
  $NUMA timeout 7200 ./venv/bin/python DAOP/daop_infer.py \
    --model mistralai/Mixtral-8x7B-v0.1 --dataset_name c4 \
    --proportion_gpu $PG --num_samples 10 > $OD/A_daop.log 2>&1 && touch $OD/A_daop.done
fi
log "<<< C4/DAOP exit=$?"

log ">>> C4 / KTransformers"
OK=$WORK/kt_results
if [ ! -f $OK/kt_c4.done ]; then
  cd /tmp
  LD_LIBRARY_PATH=$KT/cuda124/lib:${LD_LIBRARY_PATH:-} \
  $NUMA timeout 7200 $V/bin/python $WORK/kt_bench.py \
    --rule $SP/optimize/optimize_rules/Mixtral.yaml --attn sdpa --cpu_infer 32 \
    --input_tokens 16,32,64 --output_tokens 2,4,8,16 --n_sample 10 --warmup 2 \
    --dataset $C4 --out $OK/kt_c4.jsonl > $OK/kt_c4.log 2>&1 && touch $OK/kt_c4.done
fi
log "<<< C4/KT exit=$?"

log "########## ranking transfer: evaluate on ShareGPT with a ranking built elsewhere ##########"
SG=$FS/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json
for TAG in piqa c4; do
  ours $FS/results/rank_$TAG fiddlestick $SG $FS/popularity_8x7b_$TAG.json
  ours $FS/results/rank_$TAG fiddler     $SG $FS/popularity_8x7b_$TAG.json
done

log "########## done ##########"
touch $SCRATCH/heldout.done
