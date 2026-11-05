# -*- coding: utf-8 -*-
"""Measure vLLM under the same conditions as the other systems.
   The budget is imposed with gpu_memory_utilization and weight offloading
   with cpu_offload_gb. vLLM V1 spawns its workers, so everything executable
lives inside main()."""
import argparse, json, os, time, random, subprocess, threading


class GpuPeak:
    """Poll nvidia-smi for the peak GPU usage. vLLM V1 runs in a worker process,
       so torch.cuda instrumentation in the parent reports zero."""

    def __init__(self, hz=10.0):
        self.period, self.peak, self._stop = 1.0 / hz, 0, threading.Event()
        self._t = None

    def _loop(self):
        while not self._stop.wait(self.period):
            try:
                o = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"], text=True, timeout=5)
                self.peak = max(self.peak, int(o.strip().split("\n")[0]))
            except Exception:
                pass

    def __enter__(self):
        self.peak = 0; self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True); self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Mixtral-8x7B-v0.1")
    ap.add_argument("--dataset", default="$FS/benchmarks/"
                                        "ShareGPT_V3_unfiltered_cleaned_split.json")
    ap.add_argument("--budget_gib", type=float, default=16.0)
    ap.add_argument("--cpu_offload_gb", type=int, default=80)
    ap.add_argument("--input_tokens", default="16,32,64")
    ap.add_argument("--output_tokens", default="2,4,8,16")
    ap.add_argument("--n_sample", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max_model_len", type=int, default=256)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="$WORK/vllm_results/vllm.jsonl")
    return ap.parse_args()


def main():
    a = parse()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    import torch
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    util = a.budget_gib / total_gib
    print(f"[cfg] GPU {total_gib:.1f} GiB -> budget {a.budget_gib} GiB "
          f"(gpu_memory_utilization={util:.4f}), cpu_offload_gb={a.cpu_offload_gb}", flush=True)

    t0 = time.time()
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=util,
              cpu_offload_gb=a.cpu_offload_gb, enforce_eager=True,
              max_model_len=a.max_model_len, disable_log_stats=True)
    load_s = time.time() - t0
    print(f"[load] {load_s:.1f}s", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model, use_fast=True)
    with open(a.dataset) as f:
        data = json.load(f)
    texts = [" ".join(d["conversations"][0]["value"].split())
             for d in data if d["conversations"]]
    random.seed(0); random.shuffle(texts)   # same sample order as run_bench.py

    def prompts(nin, k):
        out, i = [], 0
        while len(out) < k and i < len(texts):
            if len(texts[i].split()) >= nin:
                out.append(tok.decode(tok(texts[i]).input_ids[:nin],
                                      skip_special_tokens=True))
            i += 1
        return out

    if a.smoke:
        o = llm.generate(prompts(16, 1), SamplingParams(temperature=0, max_tokens=8))
        print("[smoke] OK:", repr(o[0].outputs[0].text[:60]), flush=True)
        return

    with open(a.out, "w") as fout:
        fout.write(json.dumps({"record": "meta", "system": "vllm",
                               "budget_gib": a.budget_gib,
                               "cpu_offload_gb": a.cpu_offload_gb,
                               "load_s": load_s,
                               "max_model_len": a.max_model_len}) + "\n")
        for nin in [int(x) for x in a.input_tokens.split(",")]:
            for nout in [int(x) for x in a.output_tokens.split(",")]:
                sp = SamplingParams(temperature=0, max_tokens=nout,
                                    min_tokens=nout, ignore_eos=True)
                for p in prompts(nin, a.warmup):    # warm-up, not measured
                    llm.generate([p], sp, use_tqdm=False)
                ps = prompts(nin, a.n_sample)
                with GpuPeak() as gp:
                    t = time.time()
                    for p in ps:
                        llm.generate([p], sp, use_tqdm=False)  # one at a time, no continuous batching
                    wall = time.time() - t
                rec = {"record": "agg", "system": "vllm", "input_token": nin,
                       "output_token": nout, "n_sample": len(ps),
                       "throughput": len(ps) * nout / wall, "wall_s": wall,
                       "peak_mib": gp.peak}
                fout.write(json.dumps(rec) + "\n"); fout.flush()
                print(f"  in={nin:3d} out={nout:3d}  {rec['throughput']:6.2f} tok/s  "
                      f"peak={rec['peak_mib']:.0f} MiB", flush=True)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
