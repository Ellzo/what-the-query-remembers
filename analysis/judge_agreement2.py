"""Agreement between the two Track B judges, and whether the second reproduces the finding.

The Track B claim that matters is not any system's absolute score but the ordering, and above all
that BM25 is last. This script asks whether an open-weights judge, run locally, reproduces it.

Reports:
  * cell-level Spearman between judge 1 (Gemini 3.5 Flash) and judge 2 (Gemma-3-27B-it), per aspect
  * system-level means under each judge, and Kendall tau between the two system rankings per corpus
  * BM25's rank under judge 2, and paired Wilcoxon of BM25 against every dense encoder
  * judge 2 against the 126 blind human judgments, on the cells both cover

Writes results/derived/eacl_judge2.csv (system-level) and prints the agreement statistics.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DERIVED, METRICS  # noqa: E402

NICE = {"gte": "gte", "jina": "jina-v3", "silma": "silma-sts", "labse": "LaBSE",
        "muffakir": "Muffakir-v2", "qwen3": "Qwen3-0.6B", "qwen3_8b": "Qwen3-8B",
        "bgem3": "BGE-M3", "e5": "mE5-large", "bm25": "BM25"}
J1 = os.path.join(DERIVED, "trackB_rejudged.csv")
J2 = os.path.join(DERIVED, "trackB_judge2.csv")


def main():
    if not os.path.exists(J2):
        print(f"missing {J2} - run analysis/judge2.py first")
        return
    a = pd.read_csv(J1)
    b = pd.read_csv(J2)
    b = b[b.get("parsed", True).astype(bool)] if "parsed" in b.columns else b
    for d in (a, b):
        d["sys"] = d.system.map(lambda s: NICE.get(s, s))

    key = ["sys", "corpus", "qid"]
    m = a[key + METRICS].merge(b[key + METRICS], on=key, suffixes=("_j1", "_j2"))
    print(f"paired cells: {len(m)}  (judge2 rows: {len(b)}, parse-failed excluded)\n")

    print("cell-level Spearman between judges, per aspect:")
    for mt in METRICS:
        x, y = m[f"{mt}_j1"], m[f"{mt}_j2"]
        ok = x.notna() & y.notna()
        r, p = spearmanr(x[ok], y[ok])
        print(f"  {mt:28s} rho={r:+.3f}  p={p:.2g}  n={int(ok.sum())}")

    rows = []
    print("\nsystem-level overall_quality:")
    for corpus in sorted(m.corpus.unique()):
        g = m[m.corpus == corpus]
        s1 = g.groupby("sys")["overall_quality_j1"].mean()
        s2 = g.groupby("sys")["overall_quality_j2"].mean()
        common = s1.index.intersection(s2.index)
        t, p = kendalltau(s1[common], s2[common])
        r1 = s1.rank(ascending=False)
        r2 = s2.rank(ascending=False)
        print(f"\n  {corpus}: Kendall tau between judges = {t:+.3f} (p={p:.3f})")
        print(f"  {'system':14s} {'judge1':>7s} {'judge2':>7s} {'rank1':>6s} {'rank2':>6s}")
        for s in s2.sort_values(ascending=False).index:
            print(f"  {s:14s} {s1.get(s, np.nan):7.2f} {s2[s]:7.2f} "
                  f"{r1.get(s, np.nan):6.0f} {r2[s]:6.0f}")
            rows.append({"corpus": corpus, "system": s, "judge1_overall": round(s1.get(s, np.nan), 3),
                         "judge2_overall": round(s2[s], 3), "judge1_rank": int(r1.get(s, 0)),
                         "judge2_rank": int(r2[s]), "kendall_tau": round(t, 3),
                         "tau_p": round(p, 4)})

        # does judge 2 also put BM25 last, and significantly so?
        piv = g.pivot_table(index="qid", columns="sys", values="overall_quality_j2")
        if "BM25" in piv.columns:
            print(f"  BM25 rank under judge2: {int(r2['BM25'])} of {len(r2)}")
            for s in piv.columns:
                if s == "BM25":
                    continue
                d = piv[["BM25", s]].dropna()
                if len(d) < 10 or (d["BM25"] == d[s]).all():
                    continue
                w, pw = wilcoxon(d["BM25"], d[s])
                print(f"    BM25 vs {s:14s} delta={d['BM25'].mean()-d[s].mean():+.2f} p={pw:.2g}")

    pd.DataFrame(rows).to_csv(os.path.join(DERIVED, "eacl_judge2.csv"), index=False)
    print(f"\nwrote {DERIVED}/eacl_judge2.csv")

    # judge 2 against the blind human judgments, on the cells both cover
    hp = os.path.join(DERIVED, "human_annotation_cells.csv")
    if os.path.exists(hp):
        H = pd.read_csv(hp)
        H["sys"] = H.system.map({"bm25": "BM25", "jina": "jina-v3", "hybrid": "hybrid"})
        hb = H.merge(b[["sys", "corpus", "qid", "overall_quality"]],
                     on=["sys", "corpus", "qid"], how="inner")
        if len(hb) > 3:
            r, p = spearmanr(hb["score"], hb["overall_quality"])
            print(f"judge2 vs blind human relevance: rho={r:+.3f} p={p:.2g} n={len(hb)} cells")


if __name__ == "__main__":
    main()
