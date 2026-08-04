"""Per-query lexical features for the query-abstraction spectrum (E2 / §3 / §5).

For each gold query we measure, against each indexing mode of its target poem:
  containment C(Q,D) = |tok(Q) & tok(D)| / |tok(Q)|      (the paper's bridge measure)
  jaccard    J(Q,D)                                       (length-robust check)
  lcn        longest verbatim n-gram shared with the doc   (the ToT / quotation measure)

`lcn` against the *verses* is the variable that separates the two query sets by design:
FannOrFlop queries simulate a half-remembered poem and often quote it; Diwan queries were
deliberately written as abstract descriptions. `has_verbatim3 = lcn >= 3` defines the subset
used for the controlled test in §5.

Writes results/derived/query_features.csv
"""
import ast
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, DERIVED, ar_tokens  # noqa: E402
from unified_run import load_queries, load_units  # noqa: E402

MODES = {"diwan": ["verses", "expl_ai"], "fannorflop": ["verses", "expl_ai", "expl_ff"]}


def containment(q, d):
    q, d = set(q), set(d)
    return len(q & d) / len(q) if q else 0.0


def jaccard(q, d):
    q, d = set(q), set(d)
    return len(q & d) / len(q | d) if (q or d) else 0.0


def longest_common_ngram(qt, dt, cap=12):
    """Length of the longest contiguous token n-gram of the query occurring in the doc."""
    best = 0
    for n in range(1, min(cap, len(qt)) + 1):
        dn = {tuple(dt[i:i + n]) for i in range(len(dt) - n + 1)}
        if any(tuple(qt[i:i + n]) in dn for i in range(len(qt) - n + 1)):
            best = n
        else:
            break
    return best


def main():
    rows = []
    for corpus in ("diwan", "fannorflop"):
        queries = load_queries(corpus, "gold")
        docs = {m: load_units(corpus, m) for m in MODES[corpus]}
        for r in queries.itertuples():
            qt = ar_tokens(r.prompt)
            rec = {"corpus": corpus, "qid": r.qid, "prompt": r.prompt,
                   "gold_pid": r.gold_pid, "q_len": len(qt)}
            for m in MODES[corpus]:
                u = docs[m].get(r.gold_pid)
                if not u:
                    rec[f"C_{m}"] = np.nan
                    rec[f"J_{m}"] = np.nan
                    rec[f"lcn_{m}"] = np.nan
                    rec[f"dlen_{m}"] = np.nan
                    continue
                dt = ar_tokens(" ".join(u))
                rec[f"C_{m}"] = containment(qt, dt)
                rec[f"J_{m}"] = jaccard(qt, dt)
                rec[f"lcn_{m}"] = longest_common_ngram(qt, dt)
                rec[f"dlen_{m}"] = len(dt)
            rows.append(rec)
    df = pd.DataFrame(rows)
    df["has_verbatim3"] = df["lcn_verses"] >= 3
    df["has_verbatim4"] = df["lcn_verses"] >= 4
    fn = os.path.join(DERIVED, "query_features.csv")
    df.to_csv(fn, index=False)
    print("wrote", fn, df.shape)

    print("\n=== Query-abstraction spectrum (gold queries) ===")
    agg = df.groupby("corpus").agg(
        n=("qid", "size"),
        q_len=("q_len", "mean"),
        verbatim3=("has_verbatim3", "mean"),
        verbatim4=("has_verbatim4", "mean"),
        lcn=("lcn_verses", "mean"),
        C_verses=("C_verses", "mean"),
        C_expl_ai=("C_expl_ai", "mean"),
    ).round(3)
    print(agg.to_string())
    ff = df[df.corpus == "fannorflop"]
    print(f"\nFannOrFlop C_expl_ff mean = {ff['C_expl_ff'].mean():.3f}  "
          f"(doc len: verses {ff['dlen_verses'].mean():.0f}, "
          f"expl_ai {ff['dlen_expl_ai'].mean():.0f}, expl_ff {ff['dlen_expl_ff'].mean():.0f} tok)")
    print("\nFannOrFlop split by verbatim quotation:")
    print(ff.groupby("has_verbatim3")[["C_verses", "C_expl_ai", "C_expl_ff", "q_len"]]
          .agg(["mean", "size"]).round(3).to_string())


if __name__ == "__main__":
    main()
