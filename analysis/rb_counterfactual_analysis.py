"""Effect of editing the same queries, per system, with paired statistics.

For each edit variant, on the queries it applies to:
  delta(system) = metric(edited) - metric(original), same queries, same gold poem
  DiD(system)   = delta(system) - delta(Qwen3-8B)
with a paired bootstrap over queries (10^4, seed 0) for the DiD, and an exact McNemar test for
edited vs original within each system (R@5).

Conditions:
  inj_*       one verbatim half-verse of the target appended to each non-quoting query
  hand_para   hand-written rewrites of the quoted wording (79 of 80 queries; one is a quotation
              from start to end). Mean query length is preserved (10.9 -> 11.3 tokens), so wording
              is not confounded with query length.
  para_span, para_span_v2   LLM rewrites, restricted to the rewrites a human rater judged valid
              (human_checks/paraphrase_check/answers.csv): strict = same information need 2 and no
              remembered wording left; lenient = same need >= 1 and wording left <= 1. Without
              ratings the output is marked `unvalidated`.
  del_span    deletion of the quoted tokens (control; it also shortens the queries)

Writes results/derived/rb_counterfactual.csv
"""
import glob
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DERIVED, ROOT  # noqa: E402
from eacl_metrics import NICE, SYSTEMS, mcnemar  # noqa: E402

RUNS = os.path.join(ROOT, "analysis", "runs_rebuttal")
ANS = os.path.join(ROOT, "human_checks", "paraphrase_check")
KEY = os.path.join(ANS, "key.csv")
REF = "qwen3_8b"
NBOOT = 10000
METRICS = {"R@5": lambda r: (r <= 5).astype(float), "R@1": lambda r: (r == 1).astype(float),
           "MRR": lambda r: 1.0 / r}


def load(system, variant):
    fn = os.path.join(RUNS, f"{system}__fannorflop__verses__{variant}.json.gz")
    if not os.path.exists(fn):
        return None
    with gzip.open(fn, "rt") as f:
        return {r["qid"]: r["rank_of_gold"] for r in json.load(f)}


def validated_qids(variant):
    """qid sets for the strict and lenient validation filters, or None if not rated yet."""
    a_fn = os.path.join(ANS, "answers.csv")
    if not (os.path.exists(a_fn) and os.path.exists(KEY)):
        return None
    a = pd.read_csv(a_fn)
    a.columns = [c.strip().lstrip("\ufeff") for c in a.columns]
    a = a.merge(pd.read_csv(KEY), on="id")
    a = a[a.variant == variant]
    strict = set(a[(a.same_need == 2) & (a.wording_left == 0)].qid)
    lenient = set(a[(a.same_need >= 1) & (a.wording_left <= 1)].qid)
    return {"strict": strict, "lenient": lenient}


def analyse(variant, qids, label):
    runs = {s: (load(s, "orig"), load(s, variant)) for s in SYSTEMS}
    runs = {s: v for s, v in runs.items() if v[0] and v[1]}
    qids = sorted(set(qids) & set.intersection(*[set(v[1]) for v in runs.values()]))
    if not qids or REF not in runs:
        return []
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(qids), size=(NBOOT, len(qids)))
    out = []
    for mname, f in METRICS.items():
        def d(s):
            o = f(np.array([runs[s][0][q] for q in qids], float))
            e = f(np.array([runs[s][1][q] for q in qids], float))
            return o, e, (e - o)[idx].mean(1)
        ro, re_, rb = d(REF)
        for s in runs:
            o, e, b = d(s)
            dd = b - rb
            rec = {"variant": variant, "subset": label, "n": len(qids), "metric": mname,
                   "system": NICE[s], "orig": o.mean(), "edited": e.mean(),
                   "delta": e.mean() - o.mean(),
                   "DiD_vs_ref": (e.mean() - o.mean()) - (re_.mean() - ro.mean()),
                   "DiD_lo": np.percentile(dd, 2.5), "DiD_hi": np.percentile(dd, 97.5),
                   "DiD_p": np.nan if s == REF else min(1.0, 2 * min(np.mean(dd <= 0), np.mean(dd >= 0)))}
            if mname == "R@5":
                _, _, rec["mcnemar_p"] = mcnemar(o.astype(bool), e.astype(bool))
            out.append(rec)
    return out


def main():
    variants = sorted({os.path.basename(p).split("__")[-1][:-8]
                       for p in glob.glob(os.path.join(RUNS, "*.json.gz"))} - {"orig"})
    rows = []
    for v in variants:
        ed = load("bm25", v)
        if ed is None or any(load(s, v) is None for s in SYSTEMS):
            print(f"skipping {v}: not run for every system yet")
            continue
        allq = list(ed)
        if v.startswith("para_span"):
            val = validated_qids(v)
            if val is None:
                rows += analyse(v, allq, "unvalidated")
            else:
                rows += analyse(v, val["strict"], "validated_strict")
                rows += analyse(v, val["lenient"], "validated_lenient")
        else:
            rows += analyse(v, allq, "all")
    df = pd.DataFrame(rows).round(4)
    df.to_csv(os.path.join(DERIVED, "rb_counterfactual.csv"), index=False)
    pd.set_option("display.width", 220)
    for (v, sub), g in df[df.metric == "R@5"].groupby(["variant", "subset"]):
        print(f"\n== {v} / {sub} (n={g.n.iloc[0]}) R@5")
        print(g[["system", "orig", "edited", "delta", "DiD_vs_ref", "DiD_lo", "DiD_hi", "DiD_p",
                 "mcnemar_p"]].sort_values("delta").to_string(index=False))


if __name__ == "__main__":
    main()
