# Known issues

## Ordering between the weight copy and the compute that reads it

The current layer's non-expert weights are copied on one CUDA stream and read
from another, and the published measurements were taken without an event
between them. `torch.cuda.synchronize()` at the end of each layer iteration
orders layer boundaries but not the two streams within a layer.

We checked whether this affected the results. Recording an event right after
`get_current_layer` and waiting on it from the compute stream produces
**identical output on all 120 generated continuations** at a 16 GB budget on
Mixtral-8x7B, with throughput unchanged (4.0496 against 4.0501 tokens/s, a
0.1% difference that is within run-to-run noise). The dependency was therefore
already satisfied in practice, and the numbers in the paper are unaffected.

The released source carries that event, so the ordering no longer depends on
timing. The data under `data/` was collected before it was added; the check
above is the evidence that the two correspond.

A coarser fix, `streams2.wait_stream(streams0)`, is **not** what to use. It
also waits for the prefetch of the next layer, which removes the double
buffering this work is about: throughput drops to 3.5676 tokens/s, 10.9%, below
the Fiddler baseline. It also changes 17 of the 120 continuations, which we
attribute to the altered interleaving between the CPU expert threads and the
GPU experts changing the order in which partial results are summed, rather than
to any value being wrong.

## bring_block_to_gpu ignores its stream argument

`bring_block_to_gpu(i, stream)` accepts a stream and does not use it; the
copies run on whichever stream is current. Passing `streams1` therefore has no
effect and the prefetch runs on `streams0` alongside `get_current_layer`. This
is harmless, and in fact necessary: the two share the `copy_*` buffers, so
running them on one stream is what keeps the prefetch from overwriting a buffer
that `get_current_layer` has not finished reading.
