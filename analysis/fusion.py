"""Fusion and hybrid retrieval, evaluated on Track A from the unified rank dumps.

Two combinations:
  FUSE(verses, expl)   reciprocal-rank fusion of a model's verse index and its explanation
                       index: does indexing both beat choosing between them?
  HYBRID(BM25, dense)  reciprocal-rank fusion of BM25 and a dense model, both verse mode, with
                       and without an explanation index.

RRF score(d) = sum_r 1/(k + rank_r(d)), k=60 (Cormack et al. 2009).

Reported per corpus and, on Fann or Flop, split by whether the query quotes the poem verbatim,
to show which recipe is safe at every point of the query-abstraction spectrum.

Writes results/derived/eacl_fusion.csv
"""
import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DERIVED, ROOT  # noqa: E402
from eacl_metrics import NICE, boot_ci, mcnemar, metrics_from_ranks, stars  # noqa: E402

RUNS = os.path.join(ROOT, "analysis", "runs")
DENSE = ["gte", "jina", "silma", "labse", "muffakir", "qwen3", "qwen3_8b", "bgem3", "e5"]
K_RRF = 60


def load(system, corpus, mode):
    fn = os.path.join(RUNS, f"{system}__{corpus}__{mode}__gold.json.gz")
    if not os.path.exists(fn):
        return None
    with gzip.open(fn, "rt", encoding="utf-8") as f:
        return {r["qid"]: r for r in json.load(f)}


def rrf_rank_of_gold(runs, qid, gold):
    """Rank of the gold poem after RRF over several ranked lists."""
    score = defaultdict(float)
    for run in runs:
        for rank, pid in enumerate(run[qid]["top_pids"], start=1):
            score[str(pid)] += 1.0 / (K_RRF + rank)
    if gold not in score:
        return None  # gold outside every list's top-100
    order = sorted(score.items(), key=lambda kv: -kv[1])
    return next(i for i, (p, _) in enumerate(order, 1) if p == gold)


def evaluate(runs, qids, golds):
    ranks, miss = [], 0
    for q in qids:
        r = rrf_rank_of_gold(runs, q, golds[q])
        if r is None:
            miss += 1
            ranks.append(1e6)  # counts as a failure at every cutoff
        else:
            ranks.append(r)
    return np.array(ranks, float), miss


def main():
    qf = pd.read_csv(os.path.join(DERIVED, "query_features.csv"))
    vmap = {(r.corpus, r.qid): bool(r.has_verbatim3) for r in qf.itertuples()}
    rows = []

    for corpus in ("diwan", "fannorflop"):
        expl_modes = ["expl_ai"] + (["expl_ff"] if corpus == "fannorflop" else [])
        bm = load("bm25", corpus, "verses")
        for model in DENSE:
            v = load(model, corpus, "verses")
            if v is None:
                continue
            qids = sorted(v.keys())
            golds = {q: v[q]["gold_pid"] for q in qids}
            variants = {"verses": [v]}
            for em in expl_modes:
                e = load(model, corpus, em)
                if e is not None:
                    variants[em] = [e]
                    variants[f"fuse_verses+{em}"] = [v, e]
            if bm is not None:
                variants["hybrid_bm25+dense"] = [v, bm]
                for em in expl_modes:
                    e = load(model, corpus, em)
                    if e is not None:
                        variants[f"hybrid_bm25+dense+{em}"] = [v, bm, e]

            base_ranks = None
            for name, runs in variants.items():
                for subset in (["all", "quote", "noquote"] if corpus == "fannorflop" else ["all"]):
                    if subset == "quote":
                        ids = [q for q in qids if vmap.get((corpus, q), False)]
                    elif subset == "noquote":
                        ids = [q for q in qids if not vmap.get((corpus, q), False)]
                    else:
                        ids = qids
                    ranks, miss = evaluate(runs, ids, golds)
                    rec = {"corpus": corpus, "model": NICE.get(model, model), "variant": name,
                           "subset": subset, "n": len(ids), "gold_outside_top100": miss}
                    rec.update({k: round(float(x), 4) for k, x in metrics_from_ranks(ranks).items()})
                    lo, hi = boot_ci(ranks, "R@5")
                    rec["R@5_lo"], rec["R@5_hi"] = round(lo, 4), round(hi, 4)
                    if name == "verses" and subset == "all":
                        base_ranks = ranks
                    if base_ranks is not None and subset == "all" and name != "verses":
                        _, _, p = mcnemar(base_ranks <= 5, ranks <= 5)
                        rec["p_vs_verses"], rec["sig"] = p, stars(p)
                    rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DERIVED, "eacl_fusion.csv"), index=False)
    print("wrote eacl_fusion.csv", df.shape)

    print("\n=== R@5 by variant (mean over models) ===")
    for corpus in ("diwan", "fannorflop"):
        d = df[(df.corpus == corpus)]
        for subset in (["all"] if corpus == "diwan" else ["all", "quote", "noquote"]):
            p = (d[d.subset == subset].pivot_table(index="variant", values="R@5", aggfunc="mean")
                 .sort_values("R@5", ascending=False))
            print(f"\n-- {corpus} / {subset} --")
            print(p.round(3).to_string())

    print("\n=== Best variant per model (FannOrFlop, all) ===")
    d = df[(df.corpus == "fannorflop") & (df.subset == "all")]
    print(d.loc[d.groupby("model")["R@5"].idxmax()][["model", "variant", "R@5"]]
          .sort_values("R@5", ascending=False).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
