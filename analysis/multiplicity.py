"""Holm-Bonferroni correction within each family of tests the paper reports.

Families are defined by what a reader compares side by side:

  A  mode contrasts in Table 2: within each (corpus, contrast), across the 10 systems
  B  the quotation-slope interaction of section 5.1: 9 systems against the reference encoder
  C  Track B: BM25 against each of the 9 dense encoders, within each corpus

Writes results/derived/eacl_multiplicity.csv and prints anything whose significance label changes.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DERIVED  # noqa: E402
from eacl_metrics import stars  # noqa: E402


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (n - i) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def main():
    rows, changed = [], []

    # ---- family A: Table 2 mode contrasts
    M = pd.read_csv(os.path.join(DERIVED, "eacl_mcnemar.csv"))
    M = M[M.subset == "all"].copy()
    for (corpus, contrast), g in M.groupby(["corpus", "contrast"]):
        adj = holm(g["p"].values)
        for (_, r), pa in zip(g.iterrows(), adj):
            before, after = stars(r["p"]), stars(pa)
            rows.append({"family": f"A: {corpus}/{contrast}", "item": r["system"],
                         "p": r["p"], "p_holm": pa, "stars": before, "stars_holm": after,
                         "k": len(g)})
            if before != after:
                changed.append((f"A {corpus}/{contrast}", r["system"], r["p"], pa,
                                before, after))

    # ---- family B: the §5 interaction
    I = os.path.join(DERIVED, "eacl_interaction.csv")
    if os.path.exists(I):
        df = pd.read_csv(I).dropna(subset=["DiD_p"])
        adj = holm(df["DiD_p"].values)
        for (_, r), pa in zip(df.iterrows(), adj):
            before, after = stars(r["DiD_p"]), stars(pa)
            rows.append({"family": "B: quotation-slope DiD", "item": r["system"],
                         "p": r["DiD_p"], "p_holm": pa, "stars": before,
                         "stars_holm": after, "k": len(df)})
            if before != after:
                changed.append(("B DiD", r["system"], r["DiD_p"], pa, before, after))

    # ---- family C: Track B, BM25 vs each dense encoder
    J = os.path.join(DERIVED, "trackB_rejudged.csv")
    if os.path.exists(J):
        from scipy.stats import wilcoxon
        NICE = {"gte": "gte", "jina": "jina-v3", "silma": "silma-sts", "labse": "LaBSE",
                "muffakir": "Muffakir-v2", "qwen3": "Qwen3-0.6B", "qwen3_8b": "Qwen3-8B",
                "bgem3": "BGE-M3", "e5": "mE5-large", "bm25": "BM25"}
        B = pd.read_csv(J)
        B["sys"] = B.system.map(NICE)
        for corpus, g in B.groupby("corpus"):
            piv = g.pivot_table(index="qid", columns="sys", values="overall_quality")
            names, ps = [], []
            for s in piv.columns:
                if s == "BM25":
                    continue
                d = piv[["BM25", s]].dropna()
                if len(d) < 10 or (d["BM25"] == d[s]).all():
                    continue
                names.append(s)
                ps.append(wilcoxon(d["BM25"], d[s]).pvalue)
            adj = holm(ps)
            for s, p, pa in zip(names, ps, adj):
                before, after = stars(p), stars(pa)
                rows.append({"family": f"C: TrackB {corpus}", "item": f"BM25 vs {s}",
                             "p": p, "p_holm": pa, "stars": before, "stars_holm": after,
                             "k": len(ps)})
                if before != after:
                    changed.append((f"C {corpus}", f"BM25 vs {s}", p, pa, before, after))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DERIVED, "eacl_multiplicity.csv"), index=False)

    n_sig_before = int((out["p"] < .05).sum())
    n_sig_after = int((out["p_holm"] < .05).sum())
    print(f"{len(out)} tests in {out.family.nunique()} families")
    print(f"significant at .05 before Holm: {n_sig_before}   after: {n_sig_after}")
    if changed:
        print("\nsignificance label changed under Holm:")
        for fam, item, p, pa, b, a in changed:
            print(f"  {fam:26s} {item:16s} p={p:.4g} -> {pa:.4g}   {b or 'ns'} -> {a or 'ns'}")
    else:
        print("\nno test changes its significance label under Holm.")
    lost = out[(out["p"] < .05) & (out["p_holm"] >= .05)]
    print(f"\ntests that lose significance entirely: {len(lost)}")
    if len(lost):
        print(lost[["family", "item", "p", "p_holm"]].to_string(index=False))
    print(f"\nwrote {DERIVED}/eacl_multiplicity.csv")


if __name__ == "__main__":
    main()
