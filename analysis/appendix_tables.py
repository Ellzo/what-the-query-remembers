"""Appendix tables, written to paper/tab_app_*.tex.

  tab_app_threshold.tex  the quotation split repeated at n-gram thresholds 2..5
  tab_app_trackA.tex     full Track A: R@1/R@5/MRR/nDCG@10 with bootstrap 95% CIs
  tab_app_trackb.tex     Track B broken out by all seven judge aspects
  tab_app_judge2*.tex    the second judge: system means and per-aspect agreement
  tab_app_metrics.tex    the quotation-slope test on R@1, R@5, MRR and nDCG@10
  tab_app_judges.tex     Track B overall quality under every judge and rubric

All read results/derived/*.csv, so nothing here is hand-entered.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DERIVED, METRICS, ROOT  # noqa: E402

PAPER = os.path.join(ROOT, "paper")
ORDER = ["Qwen3-8B", "BGE-M3", "mE5-large", "jina-v3", "silma-sts", "gte",
         "Qwen3-0.6B", "LaBSE", "Muffakir-v2", "BM25"]
MODE_NAME = {"verses": "verses", "expl_ai": "AI expl.", "expl_ff": "FoF expl."}


def w(name, body):
    with open(os.path.join(PAPER, name), "w", encoding="utf-8") as f:
        f.write(body + "\n")
    print("wrote", name)


def t_threshold():
    S = pd.read_csv(os.path.join(DERIVED, "eacl_threshold_sweep.csv"))
    thrs = sorted(S.threshold.unique())
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{$\\geq{t}$}}" for t in thrs)
    cmid = "".join(f"\\cmidrule(lr){{{2*i+2}-{2*i+3}}}" for i in range(len(thrs)))
    n = S[S.system == "BM25"].set_index("threshold")
    sub = " & ".join(f"\\scriptsize {int(n.loc[t,'n_quote'])} & \\scriptsize {int(n.loc[t,'n_noquote'])}"
                     for t in thrs)
    b = ["\\begin{tabular}{l" + "cc" * len(thrs) + "}", "\\toprule",
         f"& {head} \\\\", cmid,
         "System & " + " & ".join(["qt. & no-qt."] * len(thrs)) + " \\\\",
         f"& {sub} \\\\", "\\midrule"]
    for s in ORDER:
        if s == "BM25":
            b.append("\\midrule")
        g = S[S.system == s].set_index("threshold")
        if g.empty:
            continue
        nm = "\\textbf{BM25}" if s == "BM25" else s
        cells = " & ".join(f"{g.loc[t,'R@5_quote']:.3f} & {g.loc[t,'R@5_noquote']:.3f}"
                           for t in thrs)
        b.append(f"{nm} & {cells} \\\\")
    b += ["\\bottomrule", "\\end{tabular}"]
    w("tab_app_threshold.tex", "\n".join(b))


def t_tracka():
    A = pd.read_csv(os.path.join(DERIVED, "eacl_trackA.csv"))
    A = A[A.subset == "all"]
    b = ["\\begin{tabular}{ll cccc}", "\\toprule",
         "System & Mode & R@1 & R@5 & MRR & nDCG@10 \\\\", "\\midrule"]
    for corpus, label in (("fannorflop", "\\textit{Fann or Flop}"), ("diwan", "\\textit{Diwan}")):
        g = A[A.corpus == corpus]
        if g.empty:
            continue
        b.append(f"\\multicolumn{{6}}{{l}}{{{label}}} \\\\")
        for s in ORDER:
            for m in ("verses", "expl_ai", "expl_ff"):
                r = g[(g.system == s) & (g["mode"] == m)]
                if r.empty:
                    continue
                r = r.iloc[0]
                cells = " & ".join(
                    f"{r[k]:.3f} \\scriptsize [{r[k+'_lo']:.3f},{r[k+'_hi']:.3f}]"
                    for k in ("R@1", "R@5", "MRR", "nDCG@10"))
                nm = "\\textbf{BM25}" if s == "BM25" else s
                b.append(f"{nm} & {MODE_NAME[m]} & {cells} \\\\")
        b.append("\\midrule")
    b[-1] = "\\bottomrule"
    b.append("\\end{tabular}")
    w("tab_app_trackA.tex", "\n".join(b))


def t_trackb_aspects():
    f = os.path.join(DERIVED, "trackB_rejudged.csv")
    if not os.path.exists(f):
        print("skip tab_app_trackb (no judge output)")
        return
    J = pd.read_csv(f)
    NICE = {"gte": "gte", "jina": "jina-v3", "silma": "silma-sts", "labse": "LaBSE",
            "muffakir": "Muffakir-v2", "qwen3": "Qwen3-0.6B", "qwen3_8b": "Qwen3-8B",
            "bgem3": "BGE-M3", "e5": "mE5-large", "bm25": "BM25"}
    J["sys"] = J.system.map(NICE)
    short = {"semantic_relevance": "sem.\\,rel.", "thematic_coherence": "them.\\,coh.",
             "metaphor_understanding": "metaph.", "semantic_coverage": "coverage",
             "direct_textual_relevance": "lit.\\,match", "precision": "prec.",
             "overall_quality": "overall"}
    b = ["\\begin{tabular}{ll" + "c" * len(METRICS) + "}", "\\toprule",
         "Corpus & System & " + " & ".join(short[m] for m in METRICS) + " \\\\", "\\midrule"]
    for corpus, label in (("fannorflop", "FoF"), ("diwan", "Diwan")):
        g = J[J.corpus == corpus]
        for s in ORDER:
            r = g[g.sys == s]
            if r.empty:
                continue
            nm = "\\textbf{BM25}" if s == "BM25" else s
            cells = " & ".join(f"{r[m].mean():.2f}" for m in METRICS)
            b.append(f"{label} & {nm} & {cells} \\\\")
        b.append("\\midrule")
    b[-1] = "\\bottomrule"
    b.append("\\end{tabular}")
    w("tab_app_trackb.tex", "\n".join(b))
    # the aspect on which a lexical system could be flattered, reported explicitly
    for corpus in ("fannorflop", "diwan"):
        g = J[J.corpus == corpus]
        d = g.groupby("sys")["direct_textual_relevance"].mean().sort_values(ascending=False)
        print(f"  direct_textual_relevance, {corpus}: BM25={d.get('BM25', np.nan):.2f}, "
              f"max={d.iloc[0]:.2f} ({d.index[0]}), rank of BM25 = "
              f"{list(d.index).index('BM25')+1}/{len(d)}")


def t_judge2():
    """Two-judge comparison: system means under each judge, plus per-aspect cell agreement."""
    f = os.path.join(DERIVED, "eacl_judge2.csv")
    j2 = os.path.join(DERIVED, "trackB_judge2.csv")
    if not (os.path.exists(f) and os.path.exists(j2)):
        print("skip tab_app_judge2 (second judge not run)")
        return
    from scipy.stats import spearmanr
    J = pd.read_csv(f)
    b = pd.read_csv(j2)
    a = pd.read_csv(os.path.join(DERIVED, "trackB_rejudged.csv"))
    NICE = {"gte": "gte", "jina": "jina-v3", "silma": "silma-sts", "labse": "LaBSE",
            "muffakir": "Muffakir-v2", "qwen3": "Qwen3-0.6B", "qwen3_8b": "Qwen3-8B",
            "bgem3": "BGE-M3", "e5": "mE5-large", "bm25": "BM25"}
    for d in (a, b):
        d["sys"] = d.system.map(NICE)

    b_rows = ["\\begin{tabular}{l cc c cc}", "\\toprule",
              "& \\multicolumn{2}{c}{\\textit{FannOrFlop}} & & \\multicolumn{2}{c}{\\textit{Diwan}} \\\\",
              "\\cmidrule(lr){2-3}\\cmidrule(lr){5-6}",
              "System & judge 1 & judge 2 & & judge 1 & judge 2 \\\\", "\\midrule"]
    for s in ORDER:
        if s == "BM25":
            b_rows.append("\\midrule")
        cells = []
        for c in ("fannorflop", "diwan"):
            r = J[(J.corpus == c) & (J.system == s)]
            cells += ([f"{r.judge1_overall.iloc[0]:.2f}", f"{r.judge2_overall.iloc[0]:.2f}"]
                      if not r.empty else ["n/a", "n/a"])
        nm = "\\textbf{BM25}" if s == "BM25" else s
        b_rows.append(f"{nm} & {cells[0]} & {cells[1]} & & {cells[2]} & {cells[3]} \\\\")
    b_rows += ["\\bottomrule", "\\end{tabular}"]
    w("tab_app_judge2.tex", "\n".join(b_rows))

    key = ["sys", "corpus", "qid"]
    m = a[key + METRICS].merge(b[key + METRICS], on=key, suffixes=("_1", "_2"))
    short = {"semantic_relevance": "semantic relevance", "thematic_coherence": "thematic coherence",
             "metaphor_understanding": "metaphor understanding", "semantic_coverage": "coverage",
             "direct_textual_relevance": "literal match", "precision": "precision",
             "overall_quality": "overall quality"}
    rows = ["\\begin{tabular}{lc}", "\\toprule", "Aspect & cell-level $\\rho$ \\\\", "\\midrule"]
    for mt in METRICS:
        x, y = m[f"{mt}_1"], m[f"{mt}_2"]
        ok = x.notna() & y.notna()
        r, _ = spearmanr(x[ok], y[ok])
        rows.append(f"{short[mt]} & {r:+.2f} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    w("tab_app_judge2_rho.tex", "\n".join(rows))


def t_metrics():
    """Quotation slope on four metrics, BM25 against Qwen3-8B, Holm over the nine comparisons."""
    S = pd.read_csv(os.path.join(DERIVED, "rb_slope_metrics.csv")).set_index(["metric", "system"])
    b = ["\\begin{tabular}{l cc c c cc}", "\\toprule",
         "& \\multicolumn{2}{c}{slope $\\Delta$} & & & & \\\\", "\\cmidrule(lr){2-3}",
         "Metric & BM25 & Qwen3-8B & difference & 95\\% CI & $p$ & $p_{\\mathrm{Holm}}$ \\\\",
         "\\midrule"]
    for m in ("R@1", "R@5", "MRR", "nDCG@10"):
        r, q = S.loc[(m, "BM25")], S.loc[(m, "Qwen3-8B")]
        b.append(f"{m} & {r.delta:.3f} & {q.delta:.3f} & ${r.DiD_vs_ref:+.3f}$ & "
                 f"[{r.DiD_lo:.3f}, {r.DiD_hi:.3f}] & {r.p:.3f} & {r.p_holm:.3f} \\\\")
    b += ["\\bottomrule", "\\end{tabular}"]
    w("tab_app_metrics.tex", "\n".join(b))


def t_judges():
    """Track B overall quality under all three judges and both rubrics, on the shared 100 queries."""
    J = pd.read_csv(os.path.join(DERIVED, "rb_judge_agreement.csv"))
    NICE = {"gte": "gte", "jina": "jina-v3", "silma": "silma-sts", "labse": "LaBSE",
            "muffakir": "Muffakir-v2", "qwen3": "Qwen3-0.6B", "qwen3_8b": "Qwen3-8B",
            "bgem3": "BGE-M3", "e5": "mE5-large", "bm25": "BM25"}
    J["sys"] = J.system.map(NICE)
    judges = ["judge1", "judge2", "judge3o", "judge3n"]
    mean = J[J.quantity == "mean_overall"].set_index(["corpus", "judge_or_pair", "sys"]).value
    b = ["\\begin{tabular}{l cccc c cccc}", "\\toprule",
         "& \\multicolumn{4}{c}{\\textit{FannOrFlop}} & & \\multicolumn{4}{c}{\\textit{Diwan}} \\\\",
         "\\cmidrule(lr){2-5}\\cmidrule(lr){7-10}",
         "System & Gem.\\ & Gem-3 & Q-orig & Q-neut & & Gem.\\ & Gem-3 & Q-orig & Q-neut \\\\",
         "\\midrule"]
    for s in ORDER:
        if s == "BM25":
            b.append("\\midrule")
        nm = "\\textbf{BM25}" if s == "BM25" else s
        cells = []
        for c in ("fannorflop", "diwan"):
            cells.append(" & ".join(f"{mean[(c, j, s)]:.2f}" for j in judges))
        b.append(f"{nm} & {cells[0]} & & {cells[1]} \\\\")
    tau = J[J.quantity == "kendall_tau"].set_index(["corpus", "judge_or_pair"]).value
    b.append("\\midrule")
    row = []
    for c in ("fannorflop", "diwan"):
        row.append(" & ".join(["ref."] + [f"{tau[(c, 'judge1-' + j)]:.2f}" for j in judges[1:]]))
    b.append(f"$\\tau$ with Gemini & {row[0]} & & {row[1]} \\\\")
    b += ["\\bottomrule", "\\end{tabular}"]
    w("tab_app_judges.tex", "\n".join(b))


if __name__ == "__main__":
    t_threshold()
    t_tracka()
    t_trackb_aspects()
    t_judge2()
    t_metrics()
    t_judges()
