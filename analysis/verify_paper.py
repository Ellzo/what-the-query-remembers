"""Reproduction gate: check every numeric claim in paper/main.tex against results/derived/.

Each CLAIMS entry pairs a number as written in the paper with a callable that recomputes it from
the released artifacts. A claim that cannot be recomputed is a claim that does not belong in the
paper. Run after any pipeline change:

    python3 analysis/verify_paper.py
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DERIVED, ROOT  # noqa: E402

TEX = os.path.join(ROOT, "paper", "main.tex")


def load(name):
    p = os.path.join(DERIVED, name)
    return pd.read_csv(p) if os.path.exists(p) else None


A = load("eacl_trackA.csv")
M = load("eacl_mcnemar.csv")
Q = load("query_features.csv")
ME = load("eacl_mechanism.csv")
F = load("eacl_fusion.csv")
H = load("human_annotation_cells.csv")
D = load("eacl_dissociation.csv")
B = load("eacl_trackB.csv")


def ta(system, corpus, mode="verses", subset="all", metric="R@5"):
    r = A[(A.system == system) & (A.corpus == corpus) & (A["mode"] == mode)
          & (A.subset == subset)]
    return float(r[metric].iloc[0])


def claims():
    c = []
    add = c.append
    # --- abstract / intro / Track A ---
    add(("BM25 FoF verses R@5", 0.746, ta("BM25", "fannorflop")))
    add(("Qwen3-8B FoF verses R@5", 0.717, ta("Qwen3-8B", "fannorflop")))
    add(("BM25 Diwan verses R@5", 0.392, ta("BM25", "diwan")))
    add(("Qwen3-8B Diwan verses R@5", 0.748, ta("Qwen3-8B", "diwan")))
    add(("BM25 rank on Diwan (of 10)", 8,
         int((A[(A.corpus == "diwan") & (A["mode"] == "verses") & (A.subset == "all")]["R@5"]
              > ta("BM25", "diwan")).sum()) + 1))
    # --- controlled test ---
    add(("BM25 quote-subset R@5", 1.000, ta("BM25", "fannorflop", subset="quote")))
    add(("BM25 no-quote R@5", 0.622, ta("BM25", "fannorflop", subset="noquote")))
    add(("Qwen3-8B no-quote R@5", 0.646, ta("Qwen3-8B", "fannorflop", subset="noquote")))
    add(("n quote queries", 80,
         int(A[(A.corpus == "fannorflop") & (A.subset == "quote")]["n"].iloc[0])))
    add(("n no-quote queries", 164,
         int(A[(A.corpus == "fannorflop") & (A.subset == "noquote")]["n"].iloc[0])))
    add(("BM25 quote-vs-noquote swing", 0.378,
         ta("BM25", "fannorflop", subset="quote") - ta("BM25", "fannorflop", subset="noquote")))
    # --- query spectrum ---
    ff, dw = Q[Q.corpus == "fannorflop"], Q[Q.corpus == "diwan"]
    add(("FoF %% verbatim 3-gram", 32.8, 100 * ff.has_verbatim3.mean()))
    add(("Diwan %% verbatim 3-gram", 2.7, 100 * dw.has_verbatim3.mean()))
    add(("FoF C(Q,verse)", 0.368, ff.C_verses.mean()))
    add(("Diwan C(Q,verse)", 0.151, dw.C_verses.mean()))
    add(("n FoF gold queries", 244, len(ff)))
    add(("n Diwan gold queries", 258, len(dw)))
    add(("total gold queries", 502, len(Q)))
    # --- augmentation swing ---
    mm = M[(M.subset == "all")]
    add(("BM25 Diwan expl gain (pts)", 0.368,
         float(mm[(mm.system == "BM25") & (mm.corpus == "diwan")]["delta"].iloc[0])))
    add(("BM25 FoF expl loss (pts)", -0.389,
         float(mm[(mm.system == "BM25") & (mm.corpus == "fannorflop")
                  & (mm.contrast == "expl_ai_vs_verses")]["delta"].iloc[0])))
    add(("BM25 Diwan expl R@5", 0.760, ta("BM25", "diwan", "expl_ai")))
    add(("BM25 FoF expl R@5", 0.357, ta("BM25", "fannorflop", "expl_ai")))
    # --- mechanism ---
    md = ME[(ME.corpus == "diwan") & (ME["mode"] == "expl_ai")]
    add(("BM25 rho_lexical Diwan", 0.609,
         float(md[md.system == "BM25"].rho_lexical.iloc[0])))
    add(("BM25 rho_embedding Diwan", 0.347,
         float(md[md.system == "BM25"].rho_embedding.iloc[0])))
    add(("dense systems where emb > lex (Diwan)", 9,
         int((md[md.system != "BM25"].rho_embedding
              > md[md.system != "BM25"].rho_lexical).sum())))
    # --- fusion ---
    if F is not None:
        q8 = F[F.model == "Qwen3-8B"]
        add(("Qwen3-8B Diwan expl_ai fusion cell", 0.930,
             float(q8[(q8.corpus == "diwan") & (q8.variant == "expl_ai")
                      & (q8.subset == "all")]["R@5"].iloc[0])))
        add(("Qwen3-8B FoF best hybrid", 0.885,
             float(q8[(q8.corpus == "fannorflop")
                      & (q8.variant == "hybrid_bm25+dense+expl_ff")
                      & (q8.subset == "all")]["R@5"].iloc[0])))
    # --- human annotation ---
    if H is not None:
        piv = H.pivot_table(index="system", values="score", aggfunc="mean")
        add(("human jina score", 0.900, float(piv.loc["jina", "score"])))
        add(("human bm25 score", 0.450, float(piv.loc["bm25", "score"])))
        add(("human hybrid score", 0.700, float(piv.loc["hybrid", "score"])))
        add(("annotated queries", 20, H.groupby(["corpus", "qid"]).ngroups))
    # --- Track B + dissociation ---
    if B is not None:
        def tb(sysname, corpus):
            return float(B[(B.system == sysname) & (B.corpus == corpus)]["overall_quality"].iloc[0])
        add(("BM25 Track B FoF", 1.90, tb("BM25", "fannorflop")))
        add(("BM25 Track B Diwan", 1.70, tb("BM25", "diwan")))
        add(("Qwen3-8B Track B FoF", 2.94, tb("Qwen3-8B", "fannorflop")))
        add(("BM25 Track B rank FoF (of 10)", 10,
             int((B[B.corpus == "fannorflop"]["overall_quality"]
                  > tb("BM25", "fannorflop")).sum()) + 1))
        add(("BM25 Track B rank Diwan (of 10)", 10,
             int((B[B.corpus == "diwan"]["overall_quality"] > tb("BM25", "diwan")).sum()) + 1))
    if D is not None:
        ffd = D[D.corpus == "fannorflop"]
        add(("Kendall tau FoF", 0.244, float(ffd.kendall_tau.iloc[0])))
        add(("Kendall tau Diwan", 0.511, float(D[D.corpus == "diwan"].kendall_tau.iloc[0])))
        add(("BM25 rank shift FoF", 9,
             int(ffd[ffd.system == "BM25"].rank_shift.iloc[0])))
        add(("max non-BM25 rank shift FoF", 4,
             int(ffd[ffd.system != "BM25"].rank_shift.abs().max())))
        # §6 now discloses that removing BM25 restores agreement on FannOrFlop
        from scipy.stats import kendalltau
        g = ffd[ffd.system != "BM25"]
        t, p = kendalltau(g["trackA_R@5"], g.trackB_overall)
        add(("Kendall tau FoF excl. BM25", 0.56, round(float(t), 2)))
        add(("its p", 0.045, float(p)))

    # §5 interaction: the claim the section now makes
    I = load("eacl_interaction.csv")
    if I is not None:
        I = I.set_index("system")
        add(("BM25 quotation slope", 0.378, float(I.loc["BM25", "delta"])))
        add(("BM25 DiD vs Qwen3-8B", 0.162, float(I.loc["BM25", "DiD_vs_ref"])))
        add(("its p", 0.005, float(I.loc["BM25", "DiD_p"])))
        add(("BM25 vs Qwen3-8B overall p (ns)", 0.47, float(I.loc["BM25", "p_vs_ref_all"])))
        add(("systems with significant DiD", 1,
             int((I["DiD_p"] < .05).sum())))
        add(("min quotation slope over systems", 0.135, float(I["delta"].min())))
        add(("max quotation slope over systems", 0.378, float(I["delta"].max())))

    S = load("eacl_threshold_sweep.csv")
    if S is not None:
        b = S[S.system == "BM25"]
        add(("BM25 quote R@5 = 1.0 at thresholds 3-5", 3,
             int((b["R@5_quote"] == 1.0).sum())))
        add(("BM25 quote hits at threshold 2", 140,
             int(round(b[b.threshold == 2]["n_quote"].iloc[0]
                       * b[b.threshold == 2]["R@5_quote"].iloc[0]))))
        add(("max BM25 DiD p over thresholds", 0.049, float(b["DiD_p"].max())))
        t2 = S[(S.threshold == 2)].set_index("system")
        add(("thr2 no-quote Qwen3-8B", 0.571, float(t2.loc["Qwen3-8B", "R@5_noquote"])))
        add(("thr2 no-quote BM25", 0.429, float(t2.loc["BM25", "R@5_noquote"])))
        add(("thr2 no-quote p", 0.039, float(b[b.threshold == 2]["p_vs_Qwen3-8B_noquote"].iloc[0])))

    # Appendix: normalisation ablation, now across more than one encoder
    N = load("eacl_norm_ablation.csv")
    if N is not None:
        piv = N.pivot_table(index="system", columns="normalisation", values="R@5")
        for sysname, raw, light, full in (("gte", 0.023, 0.488, 0.469),
                                          ("labse", 0.093, 0.368, 0.333)):
            if sysname in piv.index:
                add((f"norm raw {sysname}", raw, round(float(piv.loc[sysname, "raw"]), 3)))
                add((f"norm light {sysname}", light, round(float(piv.loc[sysname, "light"]), 3)))
                add((f"norm full {sysname}", full, round(float(piv.loc[sysname, "full"]), 3)))
        gaps = (piv["light"] - piv["full"])
        add(("min light-vs-full gap", 0.019, round(float(gaps.min()), 3)))

    # Limitations: the blinded known-item spot check
    SP = load("eacl_spotcheck.csv")
    if SP is not None:
        s = SP.set_index("criterion")
        add(("spotcheck strict rate", 0.34, float(s.loc["strict (score 2)", "rate"])))
        add(("spotcheck lenient rate", 0.72, float(s.loc["lenient (score 2 or 1)", "rate"])))
        add(("spotcheck n failures", 50, int(s.loc["strict (score 2)", "n"])))
        add(("spotcheck CI lo (pct)", 22, round(100 * float(s.loc["strict (score 2)", "lo"]))))
        add(("spotcheck CI hi (pct)", 48, round(100 * float(s.loc["strict (score 2)", "hi"]))))
        add(("spotcheck diwan rate", 0.40, float(s.loc["strict, diwan", "rate"])))
        add(("spotcheck fannorflop rate", 0.28, float(s.loc["strict, fannorflop", "rate"])))
        add(("corrected R@5 FoF", 0.813,
             float(s.loc["strict (score 2)", "corrected_R@5_fannorflop"])))
        add(("corrected R@5 Diwan", 0.834,
             float(s.loc["strict (score 2)", "corrected_R@5_diwan"])))
        add(("DiD after correction", 0.11,
             round(float(s.loc["DiD, register-specific", "rate"]), 2)))

    # §7 / Appendix: the open-weights second judge
    J2 = load("eacl_judge2.csv")
    B2 = load("trackB_judge2.csv")
    if J2 is not None and B2 is not None:
        for corpus, tau in (("diwan", 0.85), ("fannorflop", 0.81)):
            g = J2[J2.corpus == corpus]
            add((f"judge1-judge2 tau {corpus}", tau, round(float(g.kendall_tau.iloc[0]), 2)))
            add((f"BM25 judge2 rank {corpus}", 9,
                 int(g[g.system == "BM25"].judge2_rank.iloc[0])))
        add(("judge2 cells", 2000, len(B2)))
        add(("judge2 parse failures", 0, int((~B2.parsed.astype(bool)).sum())))
        from scipy.stats import spearmanr as _sp
        A1 = load("trackB_rejudged.csv")
        nice = {"gte": "gte", "jina": "jina-v3", "silma": "silma-sts", "labse": "LaBSE",
                "muffakir": "Muffakir-v2", "qwen3": "Qwen3-0.6B", "qwen3_8b": "Qwen3-8B",
                "bgem3": "BGE-M3", "e5": "mE5-large", "bm25": "BM25"}
        for d in (A1, B2):
            d["sys"] = d.system.map(nice)
        k = ["sys", "corpus", "qid"]
        mm = A1[k + ["overall_quality"]].merge(B2[k + ["overall_quality"]], on=k,
                                               suffixes=("_1", "_2"))
        add(("judge1-judge2 cell rho", 0.67,
             round(float(_sp(mm.overall_quality_1, mm.overall_quality_2)[0]), 2)))

    # §7 / Appendix: the judge's per-aspect behaviour
    J = load("trackB_rejudged.csv")
    if J is not None:
        nice = {"bm25": "BM25"}
        J = J.assign(sys=J.system.map(lambda s: nice.get(s, s)))
        for corpus, val in (("fannorflop", 4.26), ("diwan", 4.48)):
            g = J[J.corpus == corpus].groupby("sys")["direct_textual_relevance"].mean()
            add((f"BM25 literal-match score {corpus}", val, float(g["BM25"])))
            add((f"BM25 literal-match rank {corpus}", 1,
                 int(g.rank(ascending=False)["BM25"])))
    return c


def main():
    tex = open(TEX).read() if os.path.exists(TEX) else ""
    ok = bad = 0
    print(f"{'claim':44s} {'paper':>9s} {'data':>9s}")
    print("-" * 66)
    for name, paper_val, data_val in claims():
        if isinstance(paper_val, int) and isinstance(data_val, (int, np.integer)):
            match = paper_val == data_val
        else:
            tol = 0.06 if abs(float(paper_val)) > 1 else 0.0006  # percentages vs rates
            match = abs(float(paper_val) - float(data_val)) < tol
        flag = "OK " if match else "MISMATCH"
        if match:
            ok += 1
        else:
            bad += 1
        print(f"{flag} {name:40s} {paper_val:>9} {data_val:>9.3f}"
              if not isinstance(data_val, (int, np.integer))
              else f"{flag} {name:40s} {paper_val:>9} {data_val:>9}")
    print("-" * 66)
    print(f"{ok} verified, {bad} mismatched")

    # every \input'd table must exist
    missing = [m for m in re.findall(r"\\input\{([^}]+)\}", tex)
               if not os.path.exists(os.path.join(ROOT, "paper", m + ".tex"))]
    if missing:
        print("MISSING \\input files:", missing)
        bad += len(missing)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
