# -*- coding: utf-8 -*-
"""data/ 의 원자료에서 그림 입력(figures/data/*.json)을 만든다.

그림과 원자료의 대응은 README.md 의 표를 따른다. 이 스크립트와 make_figures.py
두 개만 돌리면 논문의 Fig3~Fig9 가 그대로 재생성된다.
"""
import collections, glob, json, os, re, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
MODELS = ("x7b", "c22")
N_EXPERT = {"x7b": 256, "c22": 448}
BUDGETS = (8, 16, 24, 32, 40)
OURS = {                      # 모델별 (16GB 기준 실행, held-out 실행) 디렉토리
    "x7b": {"sharegpt": "final_b1_n10", "c4": "c4_b1_n10", "beam": "v4_beam_x7b"},
    "c22": {"sharegpt": "v3_c22_16GB",  "c4": "c4_c22",    "beam": "v4_beam_c22"},
}
BUDGET_DIR = {"x7b": "v5_x7b_{g}GB", "c22": "v3_c22_{g}GB"}


def _samples(path, beam=1):
    """run_bench.py 가 남긴 jsonl 에서 셀별 처리량·지연·적중률을 모은다."""
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
        dc[k].append(r["decode_time"] / r["output_token"])
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
    """meta 레코드만 읽는다. 상주량은 cudaMemGetInfo 기준(cuda_used_mib)을 쓴다."""
    if not os.path.exists(path):
        return None
    for line in open(path):
        r = json.loads(line)
        if r.get("record") == "meta":
            return r
    return None


def _daop_log(path):
    """DAOP 는 stdout 로그만 남기므로 거기서 긁는다."""
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
        # 상주 하한: expert 캐시를 0 으로 강제한 실행의 프로세스 GPU 점유량
        r = {}
        for k, s in (("Fiddler", "fd"), ("Fiddlestick", "fs")):
            meta = _meta("%s/ours/residency/%s_%s.jsonl" % (D, m, s))
            if meta:
                assert meta.get("n_expert_on_gpu") == 0, "캐시가 0 인 실행이어야 한다"
                r[k] = meta["cuda_used_mib"]
        v = _daop_log("%s/daop/daop_orig_floor/%s.log" % (D, m))
        if v:
            r["DAOP"] = v["cuda"] or v["peak"]
        out["residency"][m] = r
        # 적중률: 보정 코퍼스와 held-out 코퍼스
        out["hit"][m] = {}
        for corpus, key in (("ShareGPT", "sharegpt"), ("C4", "c4")):
            e = {}
            for k, s in (("Fiddlestick", "fiddlestick"), ("Fiddler", "fiddler")):
                v = _samples("%s/ours/%s/%s.jsonl" % (D, d[key], s))
                if v and v.get("hit") is not None:
                    e[k] = {"hit": v["hit"], "nexp": v["nexp"],
                            "coverage": v["nexp"] / N_EXPERT[m]}
            v = _daop_log("%s/daop/daop_orig_hit/%s_%s.log" % (D, m, key))
            if v and v.get("hit") is not None:
                e["DAOP"] = {"hit": v["hit"], "nexp": v["nexp"],
                             "coverage": v["nexp"] / N_EXPERT[m]}
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
