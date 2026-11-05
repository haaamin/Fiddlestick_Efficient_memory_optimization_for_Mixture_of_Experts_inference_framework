# -*- coding: utf-8 -*-
"""Build the figure input (figures/data/paper_data.json) from data/.

Which measurements belong to which figure is listed in the table in
README.md. Running this script and then make_figures.py regenerates Fig 3 to
Fig 9 of the paper.
"""
import collections, glob, json, os, re, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
MODELS = ("x7b", "c22")
N_EXPERT = {"x7b": 256, "c22": 448}
BUDGETS = (8, 16, 24, 32, 40)
OURS = {                      # per model: the 16 GB run, the held-out run, the beam sweep
    "x7b": {"sharegpt": "final_b1_n10", "c4": "c4_b1_n10", "beam": "v4_beam_x7b"},
    "c22": {"sharegpt": "v3_c22_16GB",  "c4": "c4_c22",    "beam": "v4_beam_c22"},
}
BUDGET_DIR = {"x7b": "v5_x7b_{g}GB", "c22": "v3_c22_{g}GB"}


def _samples(path, beam=1):
    """Collect per-cell throughput, latency and hit rate from a run_bench.py jsonl."""
    if not os.path.exists(path):
        return None
    tp = collections.defaultdict(list)
    pf = collections.defaultdict(list)
    dc = collections.defaultdict(list)
    hit = collections.defaultdict(list)
    meta = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("record") == "meta":
            meta = r
        if r.get("record") != "sample" or r.get("beam_width") != beam:
            continue
        k = "%sx%s" % (r["input_token"], r["output_token"])
        tp[k].append(r["output_token"] / (r["prefill_time"] + r["decode_time"]))
        pf[k].append(r["prefill_time"])
        # The first token comes out of the prefill pass, so the decode stage
        # performs N-1 forward steps for an output of N tokens.
        dc[k].append(r["decode_time"] / (r["output_token"] - 1))
        if "hit_rate" in r:
            hit[k].append(r["hit_rate"])
    if not tp:
        return None
    cells = {k: st.mean(v) for k, v in tp.items()}
    out = {"cells": cells, "mean": st.mean(list(cells.values())),
           "prefill": st.mean([st.mean(v) for v in pf.values()]),
           "decode_tok": st.mean([st.mean(v) for v in dc.values()]),
           "peak": meta.get("init_peak_mib"), "nexp": meta.get("n_expert_on_gpu")}
    if hit:
        out["hit"] = st.mean([x for v in hit.values() for x in v])
    return out


def _meta(path):
    """Read the meta record. Residency uses cuda_used_mib, the cudaMemGetInfo basis."""
    if not os.path.exists(path):
        return None
    for line in open(path):
        r = json.loads(line)
        if r.get("record") == "meta":
            return r
    return None


def _daop_log(path):
    """DAOP only writes to stdout, so its numbers are parsed out of the log."""
    if not os.path.exists(path):
        return None
    cells, cur, peak, cuda, hit, nexp = {}, None, None, None, None, None
    for line in open(path, errors="replace"):
        m = re.search(r"input_token: (\d+)", line)
        if m:
            cur = int(m.group(1))
        m = re.search(r"output_token:\s*(\d+),\s*([0-9.]+)\s*token/s", line)
        if m:
            cells["%dx%s" % (cur, m.group(1))] = float(m.group(2))
        m = re.search(r"PEAK_MIB (\d+)", line)
        if m:
            peak = int(m.group(1))
        m = re.search(r"CUDA_USED_MIB (\d+)", line)
        if m:
            cuda = int(m.group(1))
        m = re.search(r"HIT_RATE ([\d.]+)", line)
        if m:
            hit = float(m.group(1))
        m = re.search(r"Number of experts on GPU: (\d+)/(\d+)", line)
        if m:
            nexp = int(m.group(1))
    if not cells:
        return {"peak": peak, "cuda": cuda, "hit": hit, "nexp": nexp}
    return {"cells": cells, "mean": st.mean(list(cells.values())), "peak": peak,
            "cuda": cuda, "hit": hit, "nexp": nexp}


def _vllm(path):
    if not os.path.exists(path):
        return None
    cells, peak = {}, 0
    for line in open(path):
        r = json.loads(line)
        if r.get("record") == "agg":
            cells["%sx%s" % (r["input_token"], r["output_token"])] = r["throughput"]
            peak = max(peak, r.get("peak_mib") or 0)
    return {"cells": cells, "mean": st.mean(list(cells.values())), "peak": peak} if cells else None


def main():
    out = {"throughput": {}, "latency": {}, "beam": {}, "budget": {},
           "residency": {}, "hit": {}, "heldout": {}, "daop_repo_fiddler": {}}

    for m in MODELS:
        d = OURS[m]
        fd = _samples("%s/ours/%s/fiddler.jsonl" % (D, d["sharegpt"]))
        fs = _samples("%s/ours/%s/fiddlestick.jsonl" % (D, d["sharegpt"]))
        out["throughput"][m] = {
            "Fiddlestick": fs, "Fiddler": fd,
            "DAOP": _daop_log("%s/daop/daop_orig/%s_16GB.log" % (D, m)),
            "vLLM": _vllm("%s/vllm/budget/%s_16GB.jsonl" % (D, m))}
        out["latency"][m] = {k: {"prefill": v["prefill"], "decode_tok": v["decode_tok"]}
                             for k, v in (("Fiddler", fd), ("Fiddlestick", fs))}
        for b in (1, 2, 4, 8):
            e = {k: _samples("%s/ours/%s/%s.jsonl" % (D, d["beam"], s), b)
                 for k, s in (("Fiddler", "fiddler"), ("Fiddlestick", "fiddlestick"))}
            if all(e.values()):
                out["beam"]["%s|%d" % (m, b)] = e
        for g in BUDGETS:
            dd = BUDGET_DIR[m].format(g=g)
            e = {}
            for k, s in (("Fiddler", "fiddler"), ("Fiddlestick", "fiddlestick")):
                v = _samples("%s/ours/%s/%s.jsonl" % (D, dd, s))
                if v:
                    e[k] = v
            v = _daop_log("%s/daop/daop_orig/%s_%dGB.log" % (D, m, g))
            if v and v.get("cells"):
                e["DAOP"] = v
            v = _vllm("%s/vllm/budget/%s_%dGB.jsonl" % (D, m, g))
            if v:
                e["vLLM"] = v
            if e:
                out["budget"]["%s|%d" % (m, g)] = e
        # Residency floor: GPU memory occupied with the expert cache forced to zero.
        r = {}
        for k, s in (("Fiddler", "fd"), ("Fiddlestick", "fs")):
            meta = _meta("%s/ours/residency/%s_%s.jsonl" % (D, m, s))
            if meta:
                assert meta.get("n_expert_on_gpu") == 0, "expected a run with an empty cache"
                r[k] = meta["cuda_used_mib"]
        v = _daop_log("%s/daop/daop_orig_floor/%s.log" % (D, m))
        if v:
            r["DAOP"] = v["cuda"] or v["peak"]
        out["residency"][m] = r
        # Hit rate on the calibration corpus and on the held-out corpus.
        out["hit"][m] = {}
        for corpus, key in (("ShareGPT", "sharegpt"), ("C4", "c4")):
            e = {}
            for k, s in (("Fiddlestick", "fiddlestick"), ("Fiddler", "fiddler")):
                v = _samples("%s/ours/%s/%s.jsonl" % (D, d[key], s))
                if v and v.get("hit") is not None:
                    e[k] = {"hit": v["hit"], "nexp": v["nexp"],
                            "coverage": v["nexp"] / N_EXPERT[m]}
            # DAOP is deliberately absent here. During decode it predicts the
            # placement for the next layer from the preceding layer's gate,
            # pre-computes the CPU side and substitutes a resident expert when
            # both of the top two are on the CPU, so "was the requested expert
            # resident" does not mean the same thing for it. See README.
            out["hit"][m][corpus] = e
        out["heldout"][m] = {k: _samples("%s/ours/%s/%s.jsonl" % (D, d["c4"], s))
                             for k, s in (("Fiddler", "fiddler"), ("Fiddlestick", "fiddlestick"))}
        v = _daop_log("%s/daop/daop_orig_fdrepo/%s.log" % (D, m))
        if v:
            out["daop_repo_fiddler"][m] = v

    os.makedirs(os.path.join(ROOT, "figures", "data"), exist_ok=True)
    p = os.path.join(ROOT, "figures", "data", "paper_data.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)
    for m in MODELS:
        print(" ", m, {k: (round(v["mean"], 3) if v and v.get("mean") else None)
                       for k, v in out["throughput"][m].items()})
        print("   residency(MiB)", out["residency"][m])


if __name__ == "__main__":
    main()
