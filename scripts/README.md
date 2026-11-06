# Run scripts

One script per experiment. Each writes into `data/` under the directory named
in the figure table of the top-level README, so that `figures/build_data.py`
picks the results up without further configuration.

Every script sources `env.sh`, which resolves paths relative to this
repository and can be overridden from the environment:

Checked by running each script with the measurement command stubbed out, so
that the paths, the calibration lookups and the per-budget arithmetic are
exercised without a GPU. `bash -n` alone does not catch an undefined function
or a variable that resolves to the wrong path.

| Variable | Default | What it is |
|---|---|---|
| `PY` | `python` | interpreter with the dependencies installed |
| `ROOT` | the repository | everything else is relative to this |
| `CALIB` | `calibration/latency` | per-model offload latency constants |
| `POP` | `calibration/popularity` | expert popularity rankings |
| `OUT` | `data` | where results are written |
| `DS` | `benchmarks/ShareGPT_V3_...json` | the evaluation corpus |
| `NUMA` | `numactl --physcpubind=32-63 --membind=1` | CPU pinning; set empty to disable |

The DAOP scripts additionally need `D` pointing at a DAOP checkout with
`patches/` applied, and the vLLM script needs an interpreter with vLLM 0.32.0.

| Script | Produces | Figure |
|---|---|---|
| `run_throughput_budget.sh` | `data/ours/{final_b1_n10,v3_c22_*GB}` | Fig 3, 4, 6, 8 |
| `run_budget_x7b.sh` | `data/ours/v5_x7b_*GB` | Fig 6 |
| `run_beam.sh` | `data/ours/v4_beam_{x7b,c22}` | Fig 5 |
| `run_residency.sh` | `data/ours/residency` | Fig 7 |
| `run_heldout.sh` | `data/ours/{c4_b1_n10,c4_c22}` | Fig 9 |
| `run_daop_budget.sh` | `data/daop/daop_orig` | Fig 3, 6 |
| `run_daop_floor.sh` | `data/daop/daop_orig_floor` | Fig 7 |
| `run_daop_repo_fiddler.sh` | `data/daop/daop_orig_fdrepo` | text only |
| `run_daop_residency_hit.sh` | `data/daop/daop_orig_hit` | not used in any figure |
| `run_vllm.sh`, `vllm_bench.py` | `data/vllm/*` | Fig 3, 6 |

A note on batching: every curve in a figure is measured in one pass of one
script, because an earlier version of Fig 6 mixed a budget point re-measured on
a different day with the rest of the sweep and the two were not comparable.
