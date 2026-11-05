# Fiddlestick

Code and measurement data for *Fiddlestick: Efficient memory optimization for
Mixture of Experts inference framework*.

Fiddlestick serves a Mixture-of-Experts model on a single GPU by streaming the
non-expert layers from host memory instead of keeping them resident, and
spending the memory this frees on a larger expert cache.

The data and scripts of the first submission are kept under `legacy/`; see
`legacy/NOTE.md`. They do not feed any current figure.

## Layout

```
src/fiddlestick/        the system described in the paper
src/fiddler_baseline/   the Fiddler implementation it is compared against
benchmarks/run_bench.py the harness that produced every measurement in data/ours
calibration/            per-model latency constants and expert popularity rankings
patches/                instrumentation applied to DAOP to record the same metrics
scripts/                the run scripts, one per experiment
data/                   raw measurements behind the figures
figures/                build_data.py -> figures/data/paper_data.json -> make_figures.py
```

## Reproducing the figures

```
python figures/build_data.py      # data/ -> figures/data/paper_data.json
python figures/make_figures.py    # -> figures/out/Fig3.png ... Fig9.png
```

No GPU is needed for this; it reads the archived measurements. The figures it
writes are byte-identical to the ones in the paper.

## Which data belongs to which figure

| Figure | What it shows | Data |
|---|---|---|
| Fig 3 | Throughput at a 16 GB budget | `data/ours/{final_b1_n10,v3_c22_16GB}`, `data/daop/daop_orig/*_16GB.log`, `data/vllm/budget/*_16GB.jsonl` |
| Fig 4 | Prefill and per-token decode latency | same runs as Fig 3 (per-sample `prefill_time` / `decode_time`) |
| Fig 5 | Minimum GPU residency, expert cache forced to zero | `data/ours/residency`, `data/daop/daop_orig_floor` |
| Fig 6 | Throughput against the VRAM budget | `data/ours/{v5_x7b_*GB,v3_c22_*GB}`, `data/daop/daop_orig`, `data/vllm/budget` |
| Fig 7 | Throughput gain against beam width | `data/ours/v4_beam_{x7b,c22}` |
| Fig 8 | Expert cache hit rate on the calibration corpus | `data/ours/{final_b1_n10,v3_c22_16GB}` |
| Fig 9 | Hit rate on a held-out corpus | `data/ours/{c4_b1_n10,c4_c22}` |

Every run in one figure comes from a single measurement session, so no curve
mixes batches taken on different days.

## Why DAOP is absent from Fig 8 and Fig 9

DAOP appears in every figure where the comparison is well posed, but not in the
two hit-rate figures. During decode it predicts the placement for the next
layer from the preceding layer's gate, pre-computes the CPU-side results, and
substitutes a resident expert when both of the top two are on the CPU. A hit
rate defined as whether the requested expert was resident therefore does not
mean the same thing for that design as for Fiddler and Fiddlestick, which share
a cache mechanism. `data/daop/daop_orig_hit/` keeps the logs that were recorded
while investigating this; their HIT_RATE lines count the prefill phase only and
should not be read as comparable to the native numbers.

## Measurement conditions

One NVIDIA H100 PCIe (80 GB) on PCIe Gen 5 x16, host AMD EPYC 9355 (32-core,
2 socket). 32 physical cores are pinned to the NUMA node that hosts the GPU and
the same pinning is applied to every system; SMT is not used. Models are
Mixtral-8x7B and Mixtral-8x22B in bfloat16. The workload is 12 cells, input
lengths {16, 32, 64} crossed with output lengths {2, 4, 8, 16}, 10 samples each,
greedy decoding unless the beam width is stated.

Throughput comparisons are aggregated as the mean of per-cell ratios rather than
the ratio of means. Memory is reported as the GPU memory the process occupies,
measured with `cudaMemGetInfo` as total minus free, which includes the CUDA
context and allocations that bypass the PyTorch caching allocator.

## Compared systems

**Fiddler** — run from `src/fiddler_baseline`, the implementation this work
extends.

**DAOP** — `https://github.com/ecolab-nus/DAOP` at commit `08fef33`.
`patches/daop_instrumentation.patch` adds the workload grid, beam width, peak
memory and hit-rate counters so that DAOP reports the same quantities as the
harness in `benchmarks/`. It changes no part of the placement algorithm: DAOP's
own brute-force expert partition is left exactly as published.

**vLLM** — 0.32.0, `cpu_offload_gb` tuned at each budget so that as many weights
as possible stay resident, since leaving it fixed lets the extra budget be
absorbed by the key-value cache.

KTransformers is not included. It executes the entire model in 4-bit, so placing
it beside the bfloat16 systems would not compare like with like.

## Environments

The three systems have conflicting dependencies and were each run in their own
environment: Fiddler and Fiddlestick on PyTorch 2.2.2 (CUDA 11.8) with
transformers 4.36.2; DAOP on PyTorch 2.2.2 (CUDA 12.1) with transformers 4.44.0;
vLLM 0.32.0 on PyTorch 2.13.0 (CUDA 13.0) with transformers 5.19.0. Python is
3.12.7, except 3.12.3 for vLLM.

## License

Apache License 2.0; see `LICENSE`. `src/fiddler_baseline` derives from Fiddler
and `patches/` applies to DAOP; see `NOTICE` for their licenses and copyright.
