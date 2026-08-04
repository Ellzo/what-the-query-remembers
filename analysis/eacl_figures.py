"""Figures for the EACL 2027 paper. All values are read from results/derived/*.csv.

F1  two panels: (a) dose-response, verse-mode R@5 by query-verse overlap quartile, both corpora;
                (b) the controlled test, FannOrFlop R@5 on quote vs no-quote queries per system.
F2  the dissociation: Track A R@5 against Track B judge score, one point per system, per corpus,
    with the human-annotated systems marked.
F3  (appendix) per-system within-query rho, lexical vs embedding predictor.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DERIVED, ROOT  # noqa: E402

FIGS = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIGS, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": .25, "grid.linewidth": .5,
                     # matplotlib defaults to Type 3 fonts in PDF output, which the ACL style
                     # guidelines tell authors to avoid; 42 emits TrueType instead.
                     "pdf.fonttype": 42, "ps.fonttype": 42})
C_LEX, C_DENSE, C_ALT = "#B4413C", "#1F6FB4", "#4C9A6A"


def fig1():
    """Figure 1, split into two single-column panels (a) and (b)."""
    D = pd.read_csv(os.path.join(DERIVED, "eacl_dose_response.csv"))
    A = pd.read_csv(os.path.join(DERIVED, "eacl_trackA.csv"))

    # (a) dose-response
    fig, a = plt.subplots(figsize=(3.3, 2.15))
    order = ["Q1 (most figurative)", "Q2", "Q3", "Q4 (most literal)"]
    d = D[D["mode"] == "verses"]
    for corp, col, mk in [("diwan", C_DENSE, "o"), ("fannorflop", C_LEX, "s")]:
        y = [d[(d.corpus == corp) & (d.quartile == q)]["R@5"].mean() for q in order]
        a.plot(range(4), y, marker=mk, color=col, lw=1.8, ms=5,
               label=f"{'Diwan' if corp=='diwan' else 'FannOrFlop'} (verses)")
    da = D[(D["mode"] == "expl_ai") & (D.corpus == "diwan")]
    a.plot(range(4), [da[da.quartile == q]["R@5"].mean() for q in order],
           marker="^", color=C_ALT, lw=1.8, ms=5, ls="--", label="Diwan (AI expl.)")
    a.set_xticks(range(4))
    a.set_xticklabels(["Q1\nfigurative", "Q2", "Q3", "Q4\nliteral"])
    a.set_ylabel("Recall@5")
    a.set_xlabel("query–verse lexical containment quartile")
    a.legend(fontsize=7, frameon=False)
    a.set_ylim(0, 1)
    fig.savefig(os.path.join(FIGS, "fig_mechanism_a.pdf"))
    plt.close(fig)
    print("wrote fig_mechanism_a.pdf")

    # (b) controlled test
    fig, a = plt.subplots(figsize=(3.3, 2.35))
    ff = A[(A.corpus == "fannorflop") & (A["mode"] == "verses")]
    p = ff.pivot_table(index="system", columns="subset", values="R@5").dropna()
    p = p.sort_values("noquote", ascending=True)
    yy = np.arange(len(p))
    a.barh(yy - .2, p["quote"], .38, color=C_LEX, label="query quotes the poem (n=80)")
    a.barh(yy + .2, p["noquote"], .38, color=C_DENSE, label="no verbatim quote (n=164)")
    a.set_yticks(yy)
    a.set_yticklabels(p.index, fontsize=7.5)
    for i, s in enumerate(p.index):
        if s == "BM25":
            a.get_yticklabels()[i].set_fontweight("bold")
    a.set_xlabel("Recall@5")
    a.set_xlim(0, 1.05)
    # the short bars sit at the bottom, so an unframed legend there sits on top of them
    a.legend(fontsize=7, frameon=True, framealpha=.92, edgecolor="none", loc="lower right")
    fig.savefig(os.path.join(FIGS, "fig_mechanism_b.pdf"))
    plt.close(fig)
    print("wrote fig_mechanism_b.pdf")


def fig2():
    """Figure 2, split into one single-column panel per corpus."""
    fa = os.path.join(DERIVED, "eacl_dissociation.csv")
    if not os.path.exists(fa):
        print("skip fig2 (no dissociation file yet)")
        return
    D = pd.read_csv(fa)
    # (a) FannOrFlop, where the dissociation is; (b) Diwan, the control.
    for corp, suffix in [("fannorflop", "a"), ("diwan", "b")]:
        g = D[D.corpus == corp]
        if g.empty:
            continue
        fig, a = plt.subplots(figsize=(3.3, 2.25))
        for r in g.itertuples():
            lex = r.system == "BM25"
            xa = r[g.columns.get_loc("trackA_R@5") + 1]
            a.scatter(xa, r.trackB_overall, s=64 if lex else 34,
                      color=C_LEX if lex else C_DENSE, zorder=3,
                      marker="D" if lex else "o", edgecolor="white", linewidth=.6)
            a.annotate(r.system, (xa, r.trackB_overall),
                       textcoords="offset points", xytext=(5, 3), fontsize=6.5,
                       fontweight="bold" if lex else "normal")
        a.set_xlabel("Track A: known-item Recall@5")
        a.set_ylabel("Track B: thematic judge score (1–5)")
        a.margins(x=.18, y=.14)
        tau = g.kendall_tau.iloc[0]
        a.set_title(f"{'Diwan' if corp=='diwan' else 'FannOrFlop'}  "
                    f"(Kendall $\\tau$={tau:+.2f})", fontsize=9)
        fig.savefig(os.path.join(FIGS, f"fig_dissociation_{suffix}.pdf"))
        plt.close(fig)
        print(f"wrote fig_dissociation_{suffix}.pdf")


def fig3():
    M = pd.read_csv(os.path.join(DERIVED, "eacl_mechanism.csv"))
    g = M[(M.corpus == "diwan") & (M["mode"] == "expl_ai")].sort_values("rho_lexical")
    fig, a = plt.subplots(figsize=(4.4, 2.9))
    yy = np.arange(len(g))
    a.barh(yy - .2, g.rho_lexical, .38, color=C_LEX, label="lexical containment")
    a.barh(yy + .2, g.rho_embedding, .38, color=C_DENSE, label="embedding cosine")
    a.set_yticks(yy)
    a.set_yticklabels(g.system, fontsize=7.5)
    for i, s in enumerate(g.system):
        if s == "BM25":
            a.get_yticklabels()[i].set_fontweight("bold")
    a.set_xlabel(r"within-query Spearman $\rho$ ($\Delta$overlap vs $\Delta$retrieval)")
    a.legend(fontsize=7, frameon=False, loc="lower right")
    fig.savefig(os.path.join(FIGS, "fig_predictors.pdf"))
    print("wrote fig_predictors.pdf")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
