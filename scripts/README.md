# Run scripts

One script per experiment. Each writes into `data/` under the directory named
in the figure table of the top-level README, so that `figures/build_data.py`
picks the results up without further configuration.

All of them assume:

- `FS` points at a checkout of this repository with the model weights available
  to `transformers` offline,
- 32 physical cores pinned to the NUMA node hosting the GPU
  (`numactl --physcpubind=... --membind=...`),
- the latency constants in `calibration/latency/` and the popularity rankings in
  `calibration/popularity/`, both passed through environment variables.

| Script | Produces | Figure |
|---|---|---|
| `run_throughput_budget.sh` | `data/ours/{final_b1_n10,v3_c22_*GB}` | Fig 3, 4, 6, 8 |
| `run_budget_x7b.sh` | `data/ours/v5_x7b_*GB` | Fig 6 |
| `run_beam.sh` | `data/ours/v4_beam_{x7b,c22}` | Fig 7 |
| `run_residency.sh` | `data/ours/residency` | Fig 5 |
| `run_heldout.sh` | `data/ours/{c4_b1_n10,c4_c22}` | Fig 9 |
| `run_daop_budget.sh` | `data/daop/daop_orig` | Fig 3, 6 |
| `run_daop_residency_hit.sh` | `data/daop/daop_orig_hit` | Fig 8, 9 |
| `run_daop_floor.sh` | `data/daop/daop_orig_floor` | Fig 5 |
| `run_daop_repo_fiddler.sh` | `data/daop/daop_orig_fdrepo` | text only |
| `run_vllm.sh`, `vllm_bench.py` | `data/vllm/*` | Fig 3, 6 |

A note on batching: every curve in a figure is measured in one pass of one
script, because an earlier version of Fig 6 mixed a budget point re-measured on
a different day with the rest of the sweep and the two were not comparable.
