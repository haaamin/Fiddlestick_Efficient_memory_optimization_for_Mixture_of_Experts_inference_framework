# DAOP instrumentation

`daop_instrumentation.patch` applies to `https://github.com/ecolab-nus/DAOP`
at commit `08fef33`:

```
git clone https://github.com/ecolab-nus/DAOP && cd DAOP
git checkout 08fef33
cp /path/to/patches/_phimoe_compat.py model/
git apply /path/to/patches/daop_instrumentation.patch
```

`_phimoe_compat.py` has to be copied in because the patch redirects DAOP's
PhiMoE import to it. DAOP imports those symbols from
`transformers.models.phimoe`, which does not exist in transformers 4.44, the
version it runs on, and the Mixtral classes sit in the same files, so without
this the import fails before any Mixtral work can start. The module re-exports
the transformers symbols and stubs the PhiMoE classes, which is sufficient for
the Mixtral evaluation in the paper. Set `DAOP_PHIMOE_DIR` to a PhiMoE model
directory if you want the real classes instead.

## What it changes

- the input and output token grid, and the beam width, become configurable by
  environment variable, so that DAOP can be run on the twelve-cell workload the
  paper uses rather than its own single condition;
- the two offload latency constants become overridable, so that DAOP is given
  the same measured values as Fiddler and Fiddlestick instead of its built-in
  defaults;
- peak GPU memory, process GPU occupancy and the expert cache hit rate are
  reported, so that DAOP reports the same quantities as `benchmarks/run_bench.py`;
- the expert popularity ranking can be read from the same file the other
  systems use.

## What it does not change

The placement algorithm. DAOP's brute-force search over expert placements is
left exactly as published, as is its prediction mechanism and its cache policy.
The scripts in `scripts/` that run DAOP refuse to start unless both model files
still contain that search, so a modified checkout cannot be measured by
accident.

## Not needed to reproduce the figures

`figures/build_data.py` reads the recorded logs under `data/daop/`, so the
figures regenerate without DAOP being installed. The patch is needed only to
repeat the measurements.
