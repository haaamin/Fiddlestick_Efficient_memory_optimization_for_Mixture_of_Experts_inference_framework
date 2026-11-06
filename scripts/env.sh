#!/bin/bash
# Paths and environment shared by the run scripts. Source this first:
#     . "$(dirname "$0")/env.sh"
#
# Override anything here from the outside; nothing is hard-coded to one machine.
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CALIB="${CALIB:-$ROOT/calibration/latency}"      # per-model latency constants
POP="${POP:-$ROOT/calibration/popularity}"       # expert popularity rankings
OUT="${OUT:-$ROOT/data}"                         # where results are written
MARK="${MARK:-$ROOT/.done}"                      # completion markers, so reruns skip
DS="${DS:-$ROOT/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json}"
PY="${PY:-python}"                               # interpreter with the deps installed
BENCH="${BENCH:-$ROOT/benchmarks/run_bench.py}"

# CPU pinning. Set NUMA="" to run without it.
NUMA="${NUMA:-numactl --physcpubind=32-63 --membind=1}"

# HF_HOME is passed through when the caller has one; it is not required.
[ -n "${HF_HOME:-}" ] && export HF_HOME
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-32}"
export FS_NUM_THREADS="${FS_NUM_THREADS:-32}"

mkdir -p "$MARK"
log(){ echo "[$(date '+%F %H:%M:%S')] $*"; }

# Fraction of an 80 GB card that a budget in GB corresponds to.
pct(){ "$PY" -c "print(f'{$1*1024/81079:.4f}')"; }
# The two offload latency constants measured for a model.
consts(){ "$PY" -c "
import json; d = json.load(open('$CALIB/$1.json'))
print(d['cpu_ms_effective'], d['h2d_pageable_ms'])"; }
