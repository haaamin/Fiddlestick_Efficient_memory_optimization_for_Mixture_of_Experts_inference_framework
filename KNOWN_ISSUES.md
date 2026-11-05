# Known issues

## Weight copy and compute are not ordered between CUDA streams

`src/fiddlestick/mixtral.py` copies the current layer's non-expert weights on
one stream and then reads those buffers from the compute stream without an
event between them:

```python
with torch.cuda.stream(streams0):
    self.get_current_layer(...)          # writes current_* buffers
    self.bring_block_to_gpu(i_layer + 1, ...)
with torch.cuda.stream(streams2):
    inps = self.current_input_layernorm(inps)   # reads current_*
```

`torch.cuda.synchronize()` runs at the end of each layer iteration, so layer
boundaries are ordered, but within a layer the two streams are not.

**This affects results.** Adding `streams2.wait_stream(streams0)` before the
compute block changes 17 of 120 generated continuations on Mixtral-8x7B at a
16 GB budget, with the expert placement and peak memory identical in both runs
(42 cached experts, 15140 MiB). Synchronisation cannot change arithmetic, so
the unsynchronised version was reading weights that had not finished copying.

Three independent runs of the unsynchronised code agree with each other on all
120 continuations, so the effect is reproducible rather than random: the timing
resolves the same way every time on this machine. It should not be assumed to
hold on other hardware.

**Cost of the coarse fix.** `wait_stream` also waits for the prefetch of the
next layer and so removes the double buffering this work is about: throughput
on Mixtral-8x7B at 16 GB falls from 4.05 to 3.57 tokens/s, 10.9%. The narrower
fix records an event straight after `get_current_layer` and waits only on that,
leaving the prefetch overlapped.

The data in `data/` and the figures in `figures/` were produced with the code as
published here, without either fix. We record the issue rather than silently
changing the code, so that the measurements and the source stay in
correspondence.
