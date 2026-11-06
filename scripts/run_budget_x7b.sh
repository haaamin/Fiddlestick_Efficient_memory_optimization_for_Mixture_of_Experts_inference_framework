#!/bin/bash
# VRAM budget sweep for Mixtral-8x7B: 8, 16, 24, 32 and 40 GB for Fiddler and
# Fiddlestick, all five measured in one pass. Produces data/ours/v5_x7b_*GB.
set -u
. "$(dirname "$0")/env.sh"
MK=$MARK; mkdir -p $MK
read LC LG < <(consts d34_32)
X=mistralai/Mixtral-8x7B-v0.1
for g in 8 16 24 32 40; do
  D=$OUT/ours/v5_x7b_${g}GB; mkdir -p $D
  for s in fiddler fiddlestick; do
    mk=v5_x7b_${g}GB_$s
    [ -f $MK/$mk.done ] && { log "skip $mk"; continue; }
    log ">>> $mk (load1m=$(cut -d' ' -f1 /proc/loadavg))"
    FS_MEM_PERCENT=$(pct $g) FS_LATENCY_CPU=$LC FS_LATENCY_GPU=$LG \
    FS_POPULARITY_JSON=$POP/popularity_8x7b.json \
    $NUMA timeout 28800 "$PY" "$BENCH" \
      --system $s --model "$X" --beam_widths 1 --n_sample 10 \
      --input_tokens 16,32,64 --output_tokens 2,4,8,16 \
      --dataset $DS --out $D/$s.jsonl > $D/$s.log 2>&1 && touch $MK/$mk.done
    log "<<< $mk exit=$?"
  done
done
log "########## v5 x7b budget DONE ##########"
touch $MARK.done
