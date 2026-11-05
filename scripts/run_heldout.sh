#!/bin/bash
# Held-out evaluation: the expert placement is computed from ShareGPT and held
# fixed while the evaluation corpus is changed to C4. Conditions match the rest
# of the evaluation: a 16 GB budget, 32 pinned cores, greedy decoding, ten
# samples, inputs 16/32/64 crossed with outputs 2/4/8/16.
# Produces data/ours/c4_b1_n10 and data/ours/c4_c22.
#
# C4 must be prepared in the same jsonl shape as the ShareGPT file; point C4 at
# it. DAOP is not run here: the hit rate it reports is not comparable to ours,
# which is why it does not appear in the held-out figure. See the README.
set -u
. "$(dirname "$0")/env.sh"
C4="${C4:-$ROOT/benchmarks/c4_prompts.json}"
X=mistralai/Mixtral-8x7B-v0.1
C=mistralai/Mixtral-8x22B-v0.1

ours(){ local dir=$1 sys=$2 model=$3 calib=$4 pop=$5
  mkdir -p "$dir"
  [ -f "$MARK/$(basename $dir)_$sys.done" ] && { log "skip $(basename $dir)/$sys"; return; }
  read LC LG < <(consts "$calib")
  log ">>> $(basename $dir) / $sys"
  FS_MEM_PERCENT=$(pct 16) FS_LATENCY_CPU=$LC FS_LATENCY_GPU=$LG FS_POPULARITY_JSON="$pop" \
  $NUMA timeout 28800 "$PY" "$BENCH" \
    --system "$sys" --model "$model" --beam_widths 1 --n_sample 10 \
    --input_tokens 16,32,64 --output_tokens 2,4,8,16 \
    --dataset "$C4" --out "$dir/$sys.jsonl" > "$dir/$sys.log" 2>&1 \
    && touch "$MARK/$(basename $dir)_$sys.done"
  log "<<< $(basename $dir)/$sys exit=$?"
}

log "########## held-out corpus (C4) ##########"
ours "$OUT/ours/c4_b1_n10" fiddlestick "$X" d34_32     "$POP/popularity_8x7b.json"
ours "$OUT/ours/c4_b1_n10" fiddler     "$X" d34_32     "$POP/popularity_8x7b.json"
ours "$OUT/ours/c4_c22"    fiddlestick "$C" d34_22b_32 "$POP/popularity_8x22b.json"
ours "$OUT/ours/c4_c22"    fiddler     "$C" d34_22b_32 "$POP/popularity_8x22b.json"
log "########## done ##########"
