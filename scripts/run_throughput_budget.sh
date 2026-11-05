#!/bin/bash
# Throughput at a 16 GB budget and across budgets for Fiddler and Fiddlestick.
# The two systems are always measured as a pair in the same session so that
# machine conditions cannot differ between them.
# Produces data/ours/{final_b1_n10,v3_c22_*GB}.
set -u
. "$(dirname "$0")/env.sh"
MK=$MARK; mkdir -p $MK

# run <dir> <label> <system> <model> <calib> <popularity> <beams> <mem%>
run(){ local dir=$1 lab=$2 sys=$3 model=$4 ck=$5 pop=$6 beams=$7 mp=$8
  local mk=$(echo "${dir##*/}_$lab" | tr '/' '_')
  [ -f $MK/$mk.done ] && { log "skip $mk"; return 0; }
  mkdir -p $dir
  read LC LG < <(consts $ck)
  log ">>> $mk (load1m=$(cut -d' ' -f1 /proc/loadavg))"
  FS_MEM_PERCENT=$mp FS_LATENCY_CPU=$LC FS_LATENCY_GPU=$LG FS_POPULARITY_JSON=$POP/$pop \
  $NUMA timeout 28800 "$PY" "$BENCH" \
    --system $sys --model "$model" --beam_widths $beams --n_sample 10 \
    --input_tokens 16,32,64 --output_tokens 2,4,8,16 \
    --dataset $DS --out $dir/$lab.jsonl > $dir/$lab.log 2>&1 && touch $MK/$mk.done
  log "<<< $mk exit=$?"
  [ -f $dir/$lab.jsonl ] && "$PY" -c "
import json,collections,statistics as st
a=collections.defaultdict(list)
for ln in open('$dir/$lab.jsonl'):
    r=json.loads(ln)
    if r.get('record')=='sample': a[(r['input_token'],r['output_token'])].append(r['prefill_time'])
v=[st.mean(x) for x in a.values()]
print(f'    mean prefill {st.mean(v):.3f}s (slowest cell {max(v):.3f}s)')"
}
X=mistralai/Mixtral-8x7B-v0.1; C=mistralai/Mixtral-8x22B-v0.1

log "########## Mixtral-8x7B ##########"
D=$OUT/ours/v3_x7b_8GB;  run $D fiddler fiddler "$X" d34_32 $POP/popularity_8x7b.json 1 $(pct 8)
                           run $D fiddlestick fiddlestick "$X" d34_32 $POP/popularity_8x7b.json 1 $(pct 8)
D=$OUT/ours/v3_beam_x7b; run $D fiddler fiddler "$X" d34_32 $POP/popularity_8x7b.json 1,2,8 $(pct 16)
                           run $D fiddlestick fiddlestick "$X" d34_32 $POP/popularity_8x7b.json 1,2,8 $(pct 16)

log "########## 8x22B ##########"
for g in 8 16 24 32 40; do
  D=$OUT/ours/v3_c22_${g}GB
  run $D fiddler     fiddler     "$C" d34_22b_32 $POP/popularity_8x22b.json 1 $(pct $g)
  run $D fiddlestick fiddlestick "$C" d34_22b_32 $POP/popularity_8x22b.json 1 $(pct $g)
done
D=$OUT/ours/v3_beam_c22
run $D fiddler     fiddler     "$C" d34_22b_32 $POP/popularity_8x22b.json 1,2,8 $(pct 16)
run $D fiddlestick fiddlestick "$C" d34_22b_32 $POP/popularity_8x22b.json 1,2,8 $(pct 16)

log "########## redo2 DONE ##########"
touch $MARK.done
