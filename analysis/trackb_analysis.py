"""E3 - Track B (thematic retrieval) and the cross-track dissociation.

Inputs
  results/derived/trackB_rejudged.csv   judge scores on the UNIFIED pipeline's retrievals
  results/derived/eacl_trackA.csv       known-item metrics from the same pipeline
  results/derived/human_annotation_cells.csv   blind human relevance, 20 queries x 3 systems

Outputs
  results/derived/eacl_trackB.csv       per-system judge means, complete-case, with CIs
  results/derived/eacl_dissociation.csv per-corpus Track A vs Track B ranks + Kendall tau

Handling of judge failures: a query is used for a comparison only if every system being
compared was scored on it (complete case). The number of retained queries is reported; raw
column means over unequal support are never compared.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DERIVED  # noqa: E402
from eacl_metrics import NICE  # noqa: E402

_RNG = np.random.default_rng(0)


def boot_ci(x, nboot=10000):
    """Bootstrap 95% CI of a mean (judge scores are means, not rank-derived rates)."""
    x = np.asarray(x, float)
    s = x[_RNG.integers(0, len(x), size=(nboot, len(x)))].mean(1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))

METRICS = ["semantic_relevance", "thematic_coherence", "metaphor_understanding",
           "semantic_coverage", "direct_textual_relevance", "precision", "overall_quality"]


def main():
    B = pd.read_csv(os.path.join(DERIVED, "trackB_rejudged.csv"))
    A = pd.read_csv(os.path.join(DERIVED, "eacl_trackA.csv"))
    B["sys"] = B["system"].map(lambda s: NICE.get(s, s))

    print("judge coverage (rows scored / 200 queries):")
    cov = B.groupby(["corpus", "sys"])["overall_quality"].agg(["count", "mean"]).round(2)
    print(cov.to_string(), "\n")

    rows, diss = [], []
    for corpus, g in B.groupby("corpus"):
        piv = g.pivot_table(index="qid", columns="sys", values="overall_quality")
        systems = list(piv.columns)
        cc = piv.dropna()
        print(f"=== {corpus}: {len(systems)} systems, complete-case n={len(cc)}/"
              f"{piv.shape[0]} queries ===")

        for s in systems:
            rec = {"corpus": corpus, "system": s, "n_complete": len(cc)}
            for m in METRICS:
                mp = g.pivot_table(index="qid", columns="sys", values=m).dropna()
                rec[m] = round(float(mp[s].mean()), 3) if s in mp else np.nan
            lo, hi = boot_ci(cc[s].values)
            rec["overall_lo"], rec["overall_hi"] = round(lo, 3), round(hi, 3)
            rows.append(rec)

        # paired Wilcoxon: BM25 vs every dense system
        if "BM25" in cc:
            print("  paired Wilcoxon, BM25 vs dense (overall_quality):")
            for s in systems:
                if s == "BM25":
                    continue
                try:
                    st, p = wilcoxon(cc["BM25"], cc[s], zero_method="wilcox")
                except ValueError:
                    continue
                print(f"    BM25 vs {s:12s} Δ={cc['BM25'].mean()-cc[s].mean():+.3f} p={p:.2e}"
                      f" {'*' if p < .05 else ''}")

        # dissociation: Track A R@5 (verse mode) vs Track B overall_quality
        a = A[(A.corpus == corpus) & (A["mode"] == "verses") & (A.subset == "all")]
        a = a.set_index("system")["R@5"]
        common = [s for s in systems if s in a.index]
        if len(common) >= 4:
            ta = a.loc[common].values
            tb = cc[common].mean().values
            tau, p = kendalltau(ta, tb)
            rho, prho = spearmanr(ta, tb)
            print(f"  Track A vs Track B over {len(common)} systems: "
                  f"Kendall tau={tau:+.3f} (p={p:.3f}), Spearman rho={rho:+.3f}\n")
            ra = pd.Series(ta, index=common).rank(ascending=False)
            rb = pd.Series(tb, index=common).rank(ascending=False)
            for s in common:
                diss.append({"corpus": corpus, "system": s, "trackA_R@5": round(a[s], 3),
                             "trackB_overall": round(float(cc[s].mean()), 3),
                             "rankA": int(ra[s]), "rankB": int(rb[s]),
                             "rank_shift": int(rb[s] - ra[s]),
                             "kendall_tau": round(tau, 3), "tau_p": round(p, 4)})

    T = pd.DataFrame(rows)
    D = pd.DataFrame(diss)
    T.to_csv(os.path.join(DERIVED, "eacl_trackB.csv"), index=False)
    D.to_csv(os.path.join(DERIVED, "eacl_dissociation.csv"), index=False)

    print("=== DISSOCIATION TABLE ===")
    for c, g in D.groupby("corpus"):
        print(f"\n-- {c} (Kendall tau={g.kendall_tau.iloc[0]:+.3f}, p={g.tau_p.iloc[0]}) --")
        print(g.sort_values("rankA")[["system", "trackA_R@5", "rankA",
                                      "trackB_overall", "rankB", "rank_shift"]]
              .to_string(index=False))

    # ---- human vs judge on the annotated subset ----
    hp = os.path.join(DERIVED, "human_annotation_cells.csv")
    if os.path.exists(hp):
        H = pd.read_csv(hp)
        H["sys"] = H["system"].map({"bm25": "BM25", "jina": "jina-v3", "hybrid": "hybrid"})
        m = H.merge(B[["corpus", "qid", "sys", "overall_quality", "semantic_relevance"]],
                    on=["corpus", "qid", "sys"], how="inner")
        if len(m) >= 10:
            r1, p1 = spearmanr(m["score"], m["overall_quality"])
            r2, p2 = spearmanr(m["score"], m["semantic_relevance"])
            print(f"\n=== HUMAN vs JUDGE (n={len(m)} system-query cells) ===")
            print(f"  human graded relevance vs judge overall_quality:    "
                  f"Spearman rho={r1:+.3f} p={p1:.2e}")
            print(f"  human graded relevance vs judge semantic_relevance: "
                  f"Spearman rho={r2:+.3f} p={p2:.2e}")
            print("\n  system means side by side:")
            print(m.groupby("sys")[["score", "overall_quality", "semantic_relevance"]]
                  .mean().round(3).to_string())
            m.to_csv(os.path.join(DERIVED, "human_vs_judge.csv"), index=False)


if __name__ == "__main__":
    main()
