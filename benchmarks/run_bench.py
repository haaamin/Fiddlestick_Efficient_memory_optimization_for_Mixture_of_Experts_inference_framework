"""Fiddlestick / Fiddler benchmark driver (H100 re-run).

Keeps the original measurement loop from latency.py / latency_fiddler.py
(n_sample=10, random.seed(0), same text-selection rule) so numbers stay
comparable with the H200 results reported in the paper. Adds:
  - --system to pick which implementation to import
  - env-driven HW constraints (FS_MEM_PERCENT, FS_NUM_THREADS)
  - per-sample JSONL output so the raw data can be published
"""

import argparse
import json
import os
import random
import sys
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser.add_argument(
        "--system",
        type=str,
        required=True,
        choices=["fiddler", "fiddlestick", "fiddler_phi", "fiddlestick_phi"],
        help="fiddler -> ../src, fiddlestick -> ../src_hamin",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="Model path. default `mistralai/Mixtral-8x7B-v0.1`.",
    )
    parser.add_argument(
        "--cpu-offload",
        type=int,
        default=1,
        choices=[0, 1],
        help="0: exeute at GPU (baseline), 1: offload to CPU.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for inference.")
    parser.add_argument("--beam_width", type=int, default=4, help="Beam search number.")
    parser.add_argument(
        "--beam_widths",
        type=str,
        default=None,
        help="Comma-separated beam widths to sweep in a single process "
        "(avoids reloading the 94GB model per width). Overrides --beam_width.",
    )
    parser.add_argument("--n_sample", type=int, default=10)
    parser.add_argument("--input_tokens", type=str, default="16,32,64")
    parser.add_argument("--output_tokens", type=str, default="2,4,8,16")
    parser.add_argument("--dataset", type=str, default="./ShareGPT_V3_unfiltered_cleaned_split.json")
    parser.add_argument("--out", type=str, required=True, help="JSONL output path")

    args = parser.parse_args()

    # resolve relative to this file, not the cwd: the sweep driver runs from the
    # repo root while the original latency.py ran from benchmarks/
    _HERE = os.path.dirname(os.path.abspath(__file__))
    # Mixtral lives in src/ and src_hamin/.
    _SRC = {"fiddler": "src", "fiddlestick": "src_hamin",
            "fiddler_phi": "src_phi", "fiddlestick_phi": "src_hamin_phi"}
    sys.path.append(os.path.join(_HERE, "..", _SRC[args.system]))
    from fiddler import FiddlerMixtral  # noqa: E402

    import torch  # noqa: E402

    with open(args.dataset, "r") as f:
        data = json.load(f)

    texts = []
    for d in data:
        if len(d["conversations"]) == 0:
            continue
        # the input of the first round
        texts.append(" ".join(d["conversations"][0]["value"].split()))

    random.seed(0)
    random.shuffle(texts)

    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(0)
    model = FiddlerMixtral(args)
    init_peak = torch.cuda.max_memory_allocated(0)
    # cuda_used_mib is the GPU memory the process occupies, total minus free.
    # It includes the CUDA context and allocations that bypass the PyTorch
    # allocator, putting it on the same basis as what vLLM reports.
    _tot_free, _tot_all = torch.cuda.mem_get_info(0)
    cuda_used = _tot_all - _tot_free

    meta = {
        "record": "meta",
        "system": args.system,
        "beam_width": args.beam_width,
        "mem_percent": float(os.environ.get("FS_MEM_PERCENT", "0.05")),
        "num_threads": int(os.environ.get("FS_NUM_THREADS", "2")),
        "kv_reserve_gib": float(os.environ.get("FS_KV_RESERVE", "0")),
        "pin_baseline": os.environ.get("FS_PIN_BASELINE", "0"),
        "latency_cpu": float(os.environ.get("FS_LATENCY_CPU", "7")),
        "latency_gpu": float(os.environ.get("FS_LATENCY_GPU", "70")),
        "n_expert_on_gpu": int(model.expert_loc.sum()),
        "n_expert_total": int(model.n_layer * model.n_expert),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory // (1024 ** 2),
        "init_peak_mib": init_peak // (1024 ** 2),
        "cuda_used_mib": cuda_used // (1024 ** 2),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    print(json.dumps(meta, ensure_ascii=False), flush=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    input_tokens = [int(x) for x in args.input_tokens.split(",")]
    output_tokens = [int(x) for x in args.output_tokens.split(",")]
    beam_widths = (
        [int(x) for x in args.beam_widths.split(",")]
        if args.beam_widths
        else [args.beam_width]
    )
    n_sample = args.n_sample

    for beam_width in beam_widths:
        # beam_width is only read inside generate(); expert placement is fixed at
        # __init__, so sweeping it in-process does not perturb the cache.
        model.beam_width = beam_width
        for input_token in input_tokens:
            for output_token in output_tokens:
                idx_text = 0
                prefill_time_sum, decode_time_sum, hit_rate_sum = 0, 0, 0
                torch.cuda.reset_peak_memory_stats(0)
                t_wall = time.time()
                oom = None
                for i_sample in range(n_sample):
                    while True:
                        text = texts[idx_text]
                        idx_text += 1
                        if len(text.split()) >= input_token:
                            # enough input length
                            break
                    try:
                        prefill_time, decode_time, hit_rate = model.generate(
                            [text], output_token=output_token, input_token=input_token
                        )
                    except torch.cuda.OutOfMemoryError as e:
                        # long sequences can exhaust the KV-cache headroom left after
                        # expert caching. Record the cell and move on instead of
                        # losing the whole run.
                        oom = str(e).split("\n")[0]
                        torch.cuda.empty_cache()
                        break
                    prefill_time_sum += prefill_time
                    decode_time_sum += decode_time
                    hit_rate_sum += hit_rate
                    with open(args.out, "a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "record": "sample",
                                    "system": args.system,
                                    "beam_width": beam_width,
                                    "input_token": input_token,
                                    "output_token": output_token,
                                    "i_sample": i_sample,
                                    "prefill_time": prefill_time,
                                    "decode_time": decode_time,
                                    "hit_rate": hit_rate,
                                }
                            )
                            + "\n"
                        )
                if oom is not None:
                    agg = {
                        "record": "agg", "system": args.system,
                        "beam_width": beam_width, "input_token": input_token,
                        "output_token": output_token, "n_sample": 0,
                        "oom": oom, "peak_mib": torch.cuda.max_memory_allocated(0) // (1024 ** 2),
                        "wall_s": round(time.time() - t_wall, 1),
                    }
                    print(json.dumps(agg), flush=True)
                    with open(args.out, "a") as f:
                        f.write(json.dumps(agg) + "\n")
                    continue
                agg = {
                    "record": "agg",
                    "system": args.system,
                    "beam_width": beam_width,
                    "input_token": input_token,
                    "output_token": output_token,
                    "n_sample": n_sample,
                    "prefill_time": prefill_time_sum / n_sample,
                    "decode_time": decode_time_sum / n_sample,
                    "hit_rate": hit_rate_sum / n_sample,
                    "throughput": output_token * n_sample / (prefill_time_sum + decode_time_sum),
                    "peak_mib": torch.cuda.max_memory_allocated(0) // (1024 ** 2),
                    "wall_s": round(time.time() - t_wall, 1),
                }
                print(json.dumps(agg), flush=True)
                with open(args.out, "a") as f:
                    f.write(json.dumps(agg) + "\n")
