#!/bin/bash
# Beam width sweep: widths 1, 2, 4 and 8 for Fiddler and Fiddlestick at a
# 16 GB budget. All four widths are measured in one pass so that the trend is
# internally consistent. Produces data/ours/v4_beam_{x7b,c22}.
set -u
. "$(dirname "$0")/env.sh"
MK=$MARK; mkdir -p $MK

run(){ local dir=$1 lab=$2 sys=$3 model=$4 ck=$5 pop=$6 beams=$7 mp=$8
  local mk=$(echo "${dir##*/}_$lab" | tr '/' '_')
  [ -f $MK/$mk.done ] && { log "skip $mk"; return 0; }
  mkdir -p $dir
  read LC LG < <(consts $ck)
  log ">>> $mk (load1m=$(cut -d' ' -f1 /proc/loadavg))"
  FS_MEM_PERCENT=$mp FS_LATENCY_CPU=$LC FS_LATENCY_GPU=$LG FS_POPULARITY_JSON=$POP/$pop \
  $NUMA timeout 43200 "$PY" "$BENCH" \
    --system $sys --model "$model" --beam_widths $beams --n_sample 10 \
    --input_tokens 16,32,64 --output_tokens 2,4,8,16 \
    --dataset $DS --out $dir/$lab.jsonl > $dir/$lab.log 2>&1 && touch $MK/$mk.done
  log "<<< $mk exit=$?"
}
X=mistralai/Mixtral-8x7B-v0.1; C=mistralai/Mixtral-8x22B-v0.1
B=1,2,4,8

log "########## 8x7B beam $B ##########"
D=$OUT/ours/v4_beam_x7b
run $D fiddler     fiddler     "$X" d34_32     $POP/popularity_8x7b.json  $B $(pct 16)
run $D fiddlestick fiddlestick "$X" d34_32     $POP/popularity_8x7b.json  $B $(pct 16)

log "########## 8x22B beam $B ##########"
D=$OUT/ours/v4_beam_c22
run $D fiddler     fiddler     "$C" d34_22b_32 $POP/popularity_8x22b.json $B $(pct 16)
run $D fiddlestick fiddlestick "$C" d34_22b_32 $POP/popularity_8x22b.json $B $(pct 16)

log "########## beam4 DONE ##########"
touch $MARK.done
