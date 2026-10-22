"""Agreement between the two thematic-relevance annotators, and whether the second annotator
reproduces the first one's results.

Rater 1 = annotation/annotations.csv. Rater 2 = annotation/annotations_rater2.csv, or the file
given on the command line. Both graded the same 126 items with the same guidelines, blind to
system and to each other.

Reports
  item level   exact agreement, Cohen's kappa (unweighted, linear, quadratic), confusion matrix,
               Spearman rho, each rater's grade distribution
  cell level   per (query, system) mean of the top-3 grades, as in annotation_analysis.py:
               system means per rater, rater-vs-rater rho, paired Wilcoxon BM25 vs jina-v3
  vs judge     Spearman rho with the primary judge's overall quality on the cells both cover,
               for rater 1, rater 2 and their mean

Usage:  python3 analysis/rb_iaa.py [annotation/annotations_rater2.csv]
Writes results/derived/rb_iaa_items.csv, rb_iaa_cells.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotation_analysis import load  # noqa: E402
from common import DERIVED, ROOT  # noqa: E402


def cells(df, prov, col):
    m = prov.merge(df[["item_id", "corpus", "qid", col]], on="item_id")
    c = m.groupby(["corpus", "qid", "system"]).agg(
        score=(col, "mean"), n=(col, "size"), n_ann=(col, lambda s: s.notna().sum())).reset_index()
    c = c[c.n_ann == c.n]
    ok = c.groupby(["corpus", "qid"]).system.transform("size") == prov.system.nunique()
    return c[ok]


def main():
    r2_fn = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "annotation", "annotations_rater2.csv")
    df, prov = load()
    r2 = pd.read_csv(r2_fn)
    r2["relevance"] = pd.to_numeric(r2["relevance"], errors="coerce")
    df = df.rename(columns={"relevance": "r1"}).merge(
        r2[["item_id", "relevance"]].rename(columns={"relevance": "r2"}), on="item_id", how="left")
    both = df.dropna(subset=["r1", "r2"])
    print(f"items: {len(df)}; graded by rater 2: {df.r2.notna().sum()}; by both: {len(both)}")
    a, b = both.r1.astype(int), both.r2.astype(int)
    print(f"exact agreement {np.mean(a == b):.3f}; within one point {np.mean(abs(a - b) <= 1):.3f}")
    for w in (None, "linear", "quadratic"):
        print(f"Cohen's kappa ({w or 'unweighted'}): {cohen_kappa_score(a, b, weights=w):.3f}")
    print(f"Spearman rho: {spearmanr(a, b)[0]:.3f}")
    print("confusion (rows rater 1, cols rater 2):")
    print(pd.crosstab(a, b))
    print("means:", round(a.mean(), 3), round(b.mean(), 3), "| distributions:",
          a.value_counts().sort_index().to_dict(), b.value_counts().sort_index().to_dict())
    for corp, g in both.groupby("corpus"):
        print(f"  {corp}: n={len(g)} quadratic kappa "
              f"{cohen_kappa_score(g.r1.astype(int), g.r2.astype(int), weights='quadratic'):.3f}")
    both.to_csv(os.path.join(DERIVED, "rb_iaa_items.csv"), index=False)

    df["rmean"] = df[["r1", "r2"]].mean(axis=1, skipna=False)
    C = None
    for col in ("r1", "r2", "rmean"):
        c = cells(df, prov, col).rename(columns={"score": col})[["corpus", "qid", "system", col]]
        C = c if C is None else C.merge(c, on=["corpus", "qid", "system"])
    print(f"\ncells complete for both raters: {len(C)} "
          f"({C.groupby(['corpus', 'qid']).ngroups} queries)")
    print("system means (0-2):")
    print(C.groupby(["corpus", "system"])[["r1", "r2", "rmean"]].mean().round(3))
    print(C.groupby("system")[["r1", "r2", "rmean"]].mean().round(3))
    print(f"cell-level rho rater1 vs rater2: {spearmanr(C.r1, C.r2)[0]:.3f}")
    piv = C.pivot_table(index=["corpus", "qid"], columns="system", values=["r1", "r2", "rmean"])
    for col in ("r1", "r2", "rmean"):
        d = piv[col]["bm25"] - piv[col]["jina"]
        p = wilcoxon(piv[col]["bm25"], piv[col]["jina"])[1] if (d != 0).any() else np.nan
        print(f"  {col}: BM25 - jina-v3 = {d.mean():+.3f}, paired Wilcoxon p = {p:.4f} "
              f"(n={len(d)} queries)")

    B = pd.read_csv(os.path.join(DERIVED, "trackB_rejudged.csv"))
    m = C.merge(B[["corpus", "qid", "system", "overall_quality"]], on=["corpus", "qid", "system"])
    print(f"\nvs primary judge overall quality ({len(m)} cells):")
    for col in ("r1", "r2", "rmean"):
        r, p = spearmanr(m[col], m.overall_quality)
        print(f"  {col}: rho = {r:+.3f}, p = {p:.2g}")
    C.to_csv(os.path.join(DERIVED, "rb_iaa_cells.csv"), index=False)


if __name__ == "__main__":
    main()
