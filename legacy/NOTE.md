# Original submission (July 2026)

These are the data and plotting scripts for the three result figures of the
first submission, which evaluated Mixtral-8x7B against Fiddler alone. They are
kept for reference.

The revision replaced this evaluation: it covers two models, compares against
three systems, sweeps the VRAM budget and validates the expert cache on a
held-out corpus. Nothing here feeds the current figures. Use the top-level
`figures/` pipeline instead.

| Here | Became |
|---|---|
| `fig3/` | Fig 3, now four systems on two models |
| `fig4/` | Fig 4, now prefill and per-token decode separately |
| `fig5/` | Fig 5, re-measured on the `cudaMemGetInfo` basis, plus Fig 6 |
