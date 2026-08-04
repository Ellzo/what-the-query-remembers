"""E4 - analyse the blinded human relevance annotation.

Per-system score = for each query, the mean graded relevance (0/1/2) of that system's top-3
retrieved poems; then averaged over queries. Because the annotation pool is deduplicated, one
item can serve several systems; provenance comes from annotation/key.csv, which the annotator
never saw.

Reports per-system means with bootstrap CIs over queries, and paired Wilcoxon tests between
systems on the per-query scores. Handles partial annotation (only complete queries are used).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DERIVED, ROOT  # noqa: E402

ANN = os.path.join(ROOT, "annotation")
RNG = np.random.default_rng(0)


def load():
    task = pd.read_csv(os.path.join(ANN, "task.csv"))
    key = pd.read_csv(os.path.join(ANN, "key.csv"))
    ann = pd.read_csv(os.path.join(ANN, "annotations.csv"))
    ann["relevance"] = pd.to_numeric(ann["relevance"], errors="coerce")
    df = task.drop(columns=["relevance", "note"]).merge(
        ann[["item_id", "relevance", "note"]], on="item_id", how="left")
    # expand provenance: one row per (item, system, rank)
    rows = []
    for r in key.itertuples():
        if not isinstance(r.systems, str):
            continue
        for tok in r.systems.split(";"):
            s, rk = tok.split("@")
            rows.append({"item_id": r.item_id, "system": s, "rank": int(rk)})
    prov = pd.DataFrame(rows)
    return df, prov


def boot_ci(x, n=10000):
    x = np.asarray(x, float)
    s = x[RNG.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main():
    df, prov = load()
    m = prov.merge(df[["item_id", "corpus", "qid", "relevance"]], on="item_id")

    # a (query, system) cell is usable only if all 3 of its items are annotated
    cell = m.groupby(["corpus", "qid", "system"]).agg(
        score=("relevance", "mean"), n=("relevance", "size"),
        n_ann=("relevance", lambda s: s.notna().sum())).reset_index()
    cell["complete"] = cell["n_ann"] == cell["n"]
    # keep queries where EVERY system is complete, so comparisons are paired
    ok = cell.groupby(["corpus", "qid"])["complete"].transform("all")
    cell = cell[ok & cell["complete"]]

    total_q = df.groupby(["corpus", "qid"]).ngroups
    used_q = cell.groupby(["corpus", "qid"]).ngroups
    print(f"annotated items: {df['relevance'].notna().sum()}/{len(df)}   "
          f"complete queries: {used_q}/{total_q}")
    if used_q == 0:
        return
    print(f"corpora covered: {sorted(cell.corpus.unique())}\n")

    print("=== Mean graded relevance of top-3 (0-2), by system ===")
    for scope in ["ALL"] + sorted(cell.corpus.unique()):
        sub = cell if scope == "ALL" else cell[cell.corpus == scope]
        nq = sub.groupby(["corpus", "qid"]).ngroups
        if scope != "ALL" and nq < 3:
            continue
        print(f"\n-- {scope} (n={nq} queries) --")
        for s, g in sorted(sub.groupby("system"), key=lambda kv: -kv[1]["score"].mean()):
            lo, hi = boot_ci(g["score"].values)
            print(f"  {s:8s} {g['score'].mean():.3f} [{lo:.3f},{hi:.3f}]")

    print("\n=== Paired Wilcoxon on per-query scores (all corpora pooled) ===")
    piv = cell.pivot_table(index=["corpus", "qid"], columns="system", values="score")
    piv = piv.dropna()
    systems = sorted(piv.columns)
    for i, a in enumerate(systems):
        for b in systems[i + 1:]:
            d = piv[a] - piv[b]
            if (d != 0).sum() == 0:
                print(f"  {a} vs {b}: identical")
                continue
            try:
                st, p = wilcoxon(piv[a], piv[b], zero_method="wilcox")
            except ValueError:
                continue
            print(f"  {a:8s} vs {b:8s}  Δ={d.mean():+.3f}  W={st:.0f}  p={p:.4f} "
                  f"{'*' if p < .05 else ''}  (n={len(piv)})")

    out = os.path.join(DERIVED, "human_annotation_cells.csv")
    cell.to_csv(out, index=False)
    print(f"\nwrote {out}")

    # distribution sanity
    print("\nrelevance distribution:",
          df["relevance"].value_counts(dropna=True).sort_index().to_dict())


if __name__ == "__main__":
    main()
