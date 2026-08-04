"""Track A metrics from the unified rank dumps: R@1/@5, MRR, nDCG@10 with bootstrap 95% CIs,
paired exact McNemar between indexing modes, and the verbatim-quotation stratification (E2).

Everything is derived from analysis/runs/*__gold.json.gz, so no table can mix pipelines.

Writes:
  results/derived/eacl_trackA.csv        one row per (system, corpus, mode) [+ subset]
  results/derived/eacl_mcnemar.csv       paired mode contrasts
"""
import glob
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DERIVED, ROOT  # noqa: E402

RUNS = os.path.join(ROOT, "analysis", "runs")
SYSTEMS = ["gte", "jina", "silma", "labse", "muffakir", "qwen3", "qwen3_8b", "bgem3", "e5", "bm25"]
NICE = {"gte": "gte", "jina": "jina-v3", "silma": "silma-sts", "labse": "LaBSE",
        "muffakir": "Muffakir-v2", "qwen3": "Qwen3-0.6B", "qwen3_8b": "Qwen3-8B",
        "bgem3": "BGE-M3", "e5": "mE5-large", "bm25": "BM25"}
RNG = np.random.default_rng(0)
NBOOT = 10000


def load_run(system, corpus, mode, qset="gold"):
    fn = os.path.join(RUNS, f"{system}__{corpus}__{mode}__{qset}.json.gz")
    if not os.path.exists(fn):
        return None
    with gzip.open(fn, "rt", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


def metrics_from_ranks(ranks):
    r = np.asarray(ranks, dtype=float)
    return {
        "R@1": np.mean(r == 1), "R@5": np.mean(r <= 5),
        "MRR": np.mean(1.0 / r),
        "nDCG@10": np.mean(np.where(r <= 10, 1.0 / np.log2(r + 1), 0.0)),
    }


def boot_ci(ranks, key, nboot=NBOOT):
    r = np.asarray(ranks, dtype=float)
    n = len(r)
    idx = RNG.integers(0, n, size=(nboot, n))
    samp = r[idx]
    if key == "R@1":
        stat = (samp == 1).mean(1)
    elif key == "R@5":
        stat = (samp <= 5).mean(1)
    elif key == "MRR":
        stat = (1.0 / samp).mean(1)
    else:
        stat = np.where(samp <= 10, 1.0 / np.log2(samp + 1), 0.0).mean(1)
    return float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))


def mcnemar(a, b):
    """a, b: boolean success vectors, paired. Exact binomial on discordant pairs."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    b01 = int(np.sum(~a & b))   # improved
    b10 = int(np.sum(a & ~b))   # worsened
    n = b01 + b10
    p = binomtest(min(b01, b10), n, 0.5).pvalue if n else 1.0
    return b01, b10, p


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else ""


def main():
    qf = pd.read_csv(os.path.join(DERIVED, "query_features.csv"))
    vmap = {(r.corpus, r.qid): bool(r.has_verbatim3) for r in qf.itertuples()}

    modes = {"diwan": ["verses", "expl_ai"],
             "fannorflop": ["verses", "expl_ai", "expl_ff"]}
    rows, mc_rows = [], []

    for corpus, mlist in modes.items():
        for system in SYSTEMS:
            runs = {m: load_run(system, corpus, m) for m in mlist}
            runs = {m: d for m, d in runs.items() if d is not None}
            if "verses" not in runs:
                continue
            for m, d in runs.items():
                d = d.dropna(subset=["rank_of_gold"])
                subsets = {"all": d}
                if corpus == "fannorflop":
                    q = d["qid"].map(lambda x: vmap.get((corpus, x), False))
                    subsets["quote"] = d[q]
                    subsets["noquote"] = d[~q]
                for sname, sd in subsets.items():
                    if len(sd) < 10:
                        continue
                    r = sd["rank_of_gold"].values
                    rec = {"system": NICE[system], "corpus": corpus, "mode": m,
                           "subset": sname, "n": len(r)}
                    mm = metrics_from_ranks(r)
                    for k, v in mm.items():
                        rec[k] = round(float(v), 4)
                        lo, hi = boot_ci(r, k)
                        rec[k + "_lo"], rec[k + "_hi"] = round(lo, 4), round(hi, 4)
                    rows.append(rec)

            # paired mode contrasts vs verses, on shared queries
            base = runs["verses"].dropna(subset=["rank_of_gold"]).set_index("qid")
            for m, d in runs.items():
                if m == "verses":
                    continue
                d = d.dropna(subset=["rank_of_gold"]).set_index("qid")
                shared = base.index.intersection(d.index)
                for sname in (["all", "quote", "noquote"] if corpus == "fannorflop" else ["all"]):
                    if sname == "quote":
                        ids = [q for q in shared if vmap.get((corpus, q), False)]
                    elif sname == "noquote":
                        ids = [q for q in shared if not vmap.get((corpus, q), False)]
                    else:
                        ids = list(shared)
                    if len(ids) < 10:
                        continue
                    a = base.loc[ids, "rank_of_gold"].values <= 5
                    b = d.loc[ids, "rank_of_gold"].values <= 5
                    b01, b10, p = mcnemar(a, b)
                    mc_rows.append({"system": NICE[system], "corpus": corpus,
                                    "contrast": f"{m}_vs_verses", "subset": sname,
                                    "n": len(ids), "R@5_verses": round(a.mean(), 4),
                                    "R@5_mode": round(b.mean(), 4),
                                    "delta": round(b.mean() - a.mean(), 4),
                                    "improved": b01, "worsened": b10,
                                    "p": p, "sig": stars(p)})

    A = pd.DataFrame(rows)
    M = pd.DataFrame(mc_rows)
    A.to_csv(os.path.join(DERIVED, "eacl_trackA.csv"), index=False)
    M.to_csv(os.path.join(DERIVED, "eacl_mcnemar.csv"), index=False)
    print("wrote eacl_trackA.csv", A.shape, "| eacl_mcnemar.csv", M.shape)

    print("\n=== Track A, verse mode, all queries (R@5 [95% CI]) ===")
    v = A[(A["mode"] == "verses") & (A.subset == "all")].sort_values(
        ["corpus", "R@5"], ascending=[True, False])
    for c in ("diwan", "fannorflop"):
        print(f"\n-- {c} --")
        for r in v[v.corpus == c].itertuples():
            print(f"  {r.system:12s} R@1={getattr(r,'_6'):.3f}  "
                  f"R@5={r._9:.3f} [{r._10:.3f},{r._11:.3f}]  n={r.n}")

    print("\n=== E2 CONTROLLED TEST: FannOrFlop verse-mode R@5 by quotation subset ===")
    ff = A[(A.corpus == "fannorflop") & (A["mode"] == "verses")]
    p = ff.pivot_table(index="system", columns="subset", values="R@5")
    p["n_quote"] = ff[ff.subset == "quote"].set_index("system")["n"]
    p["n_noquote"] = ff[ff.subset == "noquote"].set_index("system")["n"]
    print(p.round(3).to_string())
    print("\n(Diwan verse-mode R@5 for reference:)")
    print(A[(A.corpus == "diwan") & (A["mode"] == "verses")]
          .set_index("system")["R@5"].round(3).to_string())


if __name__ == "__main__":
    main()
