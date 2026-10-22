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
        # section 7: removing BM25 restores agreement on FannOrFlop
        from scipy.stats import kendalltau
        g = ffd[ffd.system != "BM25"]
        t, p = kendalltau(g["trackA_R@5"], g.trackB_overall)
        add(("Kendall tau FoF excl. BM25", 0.56, round(float(t), 2)))
        add(("its p", 0.045, float(p)))

    # section 5.1: the quotation interaction
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

    # Appendix: normalisation ablation across three encoders
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

    c += edit_claims()
    return c


def edit_claims():
    """Claims from the edited-query experiments, the extra judges and annotator, and the corpus
    checks (§5.2, §6, §9, Limitations, Appendices F, G, I, K, L)."""
    c = []
    add = c.append
    CF = load("rb_counterfactual.csv")
    if CF is not None:
        cf = CF[CF.metric == "R@5"].set_index(["variant", "subset", "system"]).sort_index()

        def v(variant, system, col, subset="all"):
            return float(cf.loc[(variant, subset, system), col])

        inj = ("inj_open", "inj_mid", "inj_final")
        for var in inj:
            add((f"injection BM25 gain {var}", 0.378, v(var, "BM25", "delta")))
            add((f"injection BM25 edited R@5 {var}", 1.000, v(var, "BM25", "edited")))
        for var, val in zip(inj, (0.232, 0.207, 0.128)):
            add((f"injection Qwen3-8B gain {var}", val, v(var, "Qwen3-8B", "delta")))
        add(("injection max DiD p", 0.004, round(max(v(x, "BM25", "DiD_p") for x in inj), 3)))
        add(("injection: only gte (opening) gains more than BM25", 1,
             int(sum(v(x, s, "delta") > v(x, "BM25", "delta") for x in inj
                     for s in cf.loc[(x, "all")].index if s != "BM25"))))
        add(("injection: the one is gte at opening", 0.384, v("inj_open", "gte", "delta")))
        add(("hand rewrite n", 79, int(v("hand_para", "BM25", "n"))))
        add(("hand rewrite BM25 before", 1.000, v("hand_para", "BM25", "orig")))
        add(("hand rewrite BM25 after", 0.646, v("hand_para", "BM25", "edited")))
        add(("hand rewrite Qwen3-8B before", 0.861, v("hand_para", "Qwen3-8B", "orig")))
        add(("hand rewrite Qwen3-8B after", 0.671, v("hand_para", "Qwen3-8B", "edited")))
        add(("hand rewrite DiD", -0.165, v("hand_para", "BM25", "DiD_vs_ref")))
        add(("hand rewrite DiD p", 0.007, round(v("hand_para", "BM25", "DiD_p"), 3)))
        add(("hand rewrite gte before", 0.785, v("hand_para", "gte", "orig")))
        add(("hand rewrite gte after", 0.405, v("hand_para", "gte", "edited")))
        add(("hand rewrite: only gte falls further than BM25", 1,
             int((cf.loc[("hand_para", "all")].delta < v("hand_para", "BM25", "delta")).sum())))
        add(("LLM rewrites n (lenient)", 55, int(v("para_span_v2", "BM25", "n", "validated_lenient"))))
        add(("LLM rewrites BM25 loss", -0.473, v("para_span_v2", "BM25", "delta", "validated_lenient")))
        add(("LLM rewrites Qwen3-8B loss", -0.255,
             v("para_span_v2", "Qwen3-8B", "delta", "validated_lenient")))
        add(("LLM rewrites DiD p", 0.006, round(v("para_span_v2", "BM25", "DiD_p", "validated_lenient"), 3)))
        add(("deletion BM25 loss", -0.600, v("del_span", "BM25", "delta")))
        add(("deletion Qwen3-8B loss", -0.413, v("del_span", "Qwen3-8B", "delta")))
        add(("deletion DiD p", 0.008, round(v("del_span", "BM25", "DiD_p"), 3)))
    from common import DATA, ar_tokens
    hp = os.path.join(DATA, "rebuttal", "fof_hand_paraphrase_queries.csv")
    if os.path.exists(hp):
        h = pd.read_csv(hp)
        add(("hand rewrite length before", 10.9, round(h.original.map(lambda s: len(ar_tokens(s))).mean(), 1)))
        add(("hand rewrite length after", 11.3, round(h.prompt.map(lambda s: len(ar_tokens(s))).mean(), 1)))
    cq = os.path.join(DATA, "rebuttal", "fof_counterfactual_queries.csv")
    if os.path.exists(cq):
        q = pd.read_csv(cq)
        d = q[q.variant == "del_span"]
        o = q[(q.variant == "orig") & q.qid.isin(d.qid)]
        add(("deletion length before", 10.9, round(o.prompt.map(lambda s: len(ar_tokens(s))).mean(), 1)))
        add(("deletion length after", 6.4, round(d.prompt.map(lambda s: len(ar_tokens(s))).mean(), 1)))

    # slope test on other metrics
    SM = load("rb_slope_metrics.csv")
    if SM is not None:
        sm = SM.set_index(["metric", "system"])
        for m, did, p, holm in (("R@1", 0.18, 0.009, 0.085), ("MRR", 0.16, 0.006, 0.050),
                                ("nDCG@10", 0.14, 0.009, 0.079)):
            add((f"slope DiD {m}", did, round(float(sm.loc[(m, "BM25"), "DiD_vs_ref"]), 2)))
            add((f"slope DiD p {m}", p, round(float(sm.loc[(m, "BM25"), "p"]), 3)))
            add((f"slope DiD Holm {m}", holm, round(float(sm.loc[(m, "BM25"), "p_holm"]), 3)))
        add(("BM25 steepest slope on all four metrics", 4,
             int(sum(SM[SM.metric == m].set_index("system").delta.idxmax() == "BM25"
                     for m in ("R@1", "R@5", "MRR", "nDCG@10")))))

    # covariate model
    MV, SH, INT = load("rb_multivariate_coef.csv"), load("rb_multivariate_shapley.csv"), \
        load("rb_multivariate_interaction.csv")
    if MV is not None:
        add(("containment coef change < 0.07 for all systems", 10,
             int(((MV.C_bivariate_b - MV.C_controlled_b).abs() < 0.07).sum())))
        add(("containment p < 1e-4 for all systems", 10, int((MV.C_controlled_p < 1e-4).sum())))
        ov = SH[SH.group == "overlap"]["share"]
        add(("overlap share min (%)", 59, round(100 * ov.min())))
        add(("overlap share max (%)", 85, round(100 * ov.max())))
        add(("containment x BM25", 2.99, round(float(INT.b.iloc[0]), 2)))
        add(("containment x BM25 p < 1e-15", 1, int(INT.p.iloc[0] < 1e-15)))

    # third judge, neutral rubric, aboutness-only ordering
    JA = load("rb_judge_agreement.csv")
    if JA is not None:
        def jv(analysis, corpus, pair, system, quantity):
            r = JA[(JA.analysis == analysis) & (JA.corpus == corpus) & (JA.judge_or_pair == pair)
                   & (JA.system == system) & (JA.quantity == quantity)]
            return float(r.value.iloc[0])
        for corp in ("fannorflop", "diwan"):
            add((f"judge3 BM25 rank {corp}", 9, int(jv("R4a", corp, "judge3o", "bm25", "rank"))))
            m = JA[(JA.judge_or_pair == "judge3o") & (JA.corpus == corp)
                   & (JA.quantity == "mean_overall")].set_index("system").value
            add((f"judge3: only LaBSE below BM25 {corp}", 1, int((m < m["bm25"]).sum())))
            add((f"judge3: LaBSE is it {corp}", 1, int(m.idxmin() == "labse")))
        add(("judge3n BM25 rank FoF", 10, int(jv("R4c", "fannorflop", "judge3n", "bm25", "rank"))))
        add(("judge3n BM25 rank Diwan", 9, int(jv("R4c", "diwan", "judge3n", "bm25", "rank"))))
        for corp, tau in (("fannorflop", 0.67), ("diwan", 0.89)):
            add((f"tau Gemini-judge3 {corp}", tau,
                 round(jv("R4a", corp, "judge1-judge3o", "all", "kendall_tau"), 2)))
        for corp, tau in (("fannorflop", 0.90), ("diwan", 0.97)):
            add((f"tau judge3 neutral-original {corp}", tau,
                 round(jv("R4c", corp, "judge3o-judge3n", "all", "kendall_tau"), 2)))
        sh = JA[JA.quantity == "mean_shift"]
        add(("neutral rubric raises every system", 1, int((sh.value > 0).all())))
        g = sh.groupby("system").value.mean()
        add(("smallest gain is BM25", 1, int(g.idxmin() == "bm25")))
        add(("smallest gain", 0.10, round(float(g.min()), 2)))
        add(("largest gain", 0.27, round(float(g.max()), 2)))

    for rub in ("original", "neutral"):
        j3 = load(f"trackB_judge3_{rub}.csv")
        if j3 is not None:
            add((f"judge3 {rub} cells", 2000, len(j3)))
            add((f"judge3 {rub} queries per corpus", 100, int(j3.groupby("corpus").qid.nunique().min())))

    # second annotator
    IT, CE = load("rb_iaa_items.csv"), load("rb_iaa_cells.csv")
    if IT is not None:
        from scipy.stats import spearmanr, wilcoxon
        from sklearn.metrics import cohen_kappa_score
        a, b = IT.r1.astype(int), IT.r2.astype(int)
        add(("items graded by both raters", 126, len(IT)))
        add(("quadratic kappa", 0.53, round(cohen_kappa_score(a, b, weights="quadratic"), 2)))
        add(("within one point (%)", 90, round(100 * np.mean(abs(a - b) <= 1))))
        add(("rater 1 mean", 0.68, round(a.mean(), 2)))
        add(("rater 2 mean", 0.90, round(b.mean(), 2)))
        piv = CE.pivot_table(index=["corpus", "qid"], columns="system", values=["r1", "r2"])
        add(("rater 1 BM25 vs jina p", 0.005, round(wilcoxon(piv["r1"]["bm25"], piv["r1"]["jina"])[1], 3)))
        add(("rater 2 BM25 vs jina p", 0.028, round(wilcoxon(piv["r2"]["bm25"], piv["r2"]["jina"])[1], 3)))
        J1 = load("trackB_rejudged.csv")
        mm = CE.merge(J1[["corpus", "qid", "system", "overall_quality"]],
                      on=["corpus", "qid", "system"])
        add(("shared cells", 40, len(mm)))
        add(("rater 2 vs judge rho", 0.64, round(spearmanr(mm.r2, mm.overall_quality)[0], 2)))
        add(("rater 1 vs judge rho", 0.75, round(spearmanr(mm.r1, mm.overall_quality)[0], 2)))

    # fusion with every dense encoder
    if F is not None:
        rows = []
        for (corp, mdl), g in F[F.subset == "all"].groupby(["corpus", "model"]):
            rows.append((corp, g[g.variant.str.startswith("hybrid")]["R@5"].max()
                         - g[~g.variant.str.startswith("hybrid")]["R@5"].max()))
        G = pd.DataFrame(rows, columns=["corpus", "d"])
        ff, dw = G[G.corpus == "fannorflop"].d, G[G.corpus == "diwan"].d
        add(("lexical arm helps on FoF, encoders", 9, int((ff > 0).sum())))
        add(("lexical arm hurts on Diwan, encoders", 9, int((dw < 0).sum())))
        add(("FoF gain min", 0.061, round(ff.min(), 3)))
        add(("FoF gain max", 0.234, round(ff.max(), 3)))
        add(("Diwan loss min", -0.097, round(dw.min(), 3)))
        add(("Diwan loss max", -0.027, round(dw.max(), 3)))

    # spans, era, poets, targets, duplicates
    CV = load("rb_query_covariates.csv")
    if CV is not None:
        q = CV[(CV.corpus == "fannorflop") & (CV.lcn >= 3)]
        add(("quoted spans", 80, len(q)))
        add(("spans in opening verse", 37, int((q.span_pos == "opening").sum())))
        add(("spans in a middle verse", 41, int((q.span_pos == "middle").sum())))
        add(("spans in final verse", 2, int((q.span_pos == "final").sum())))
        add(("spans in no other poem", 77, int((q.span_df == 0).sum())))
    CC = load("rb_corpus_checks.csv")
    if CC is not None:
        cc = CC.set_index(["check", "system", "quantity"]).value
        add(("era dense coef", 0.27, round(cc[("era", "dense", "coef")], 2)))
        add(("era dense p", 0.010, round(cc[("era", "dense", "p")], 3)))
        add(("era BM25 coef", -0.36, round(cc[("era", "bm25", "coef")], 2)))
        add(("era BM25 p", 0.14, round(cc[("era", "bm25", "p")], 2)))
        add(("era interaction p", 0.015, round(cc[("era", "interaction", "p")], 3)))
        pf = CC[CC.check == "poet_freq"].set_index(["system", "quantity"]).value
        add(("poet freq max |rho| <= .08", 1, int(pf.xs("abs_rho", level=1).max() <= 0.08)))
        add(("poet freq min p > .2", 1, int(pf.xs("p", level=1).drop("all", errors="ignore").min() > 0.2)))
        add(("poems per poet min", 1, int(cc[("poet_freq", "all", "min_poems_per_poet")])))
        add(("poems per poet max", 612, int(cc[("poet_freq", "all", "max_poems_per_poet")])))
        add(("target tests all p > .05", 1, int(CC[CC.check == "targets"].value.min() > 0.05)))
        add(("FoF poems kept", 4687, int(cc[("fof_filter", "all", "n_poems")])))
        add(("FoF max verses", 16, int(cc[("fof_filter", "all", "max_verses")])))
        add(("Diwan poems with a twin", 274, int(cc[("duplicates", "diwan", "n_poems_with_twin")])))
        add(("Diwan twin groups", 135, int(cc[("duplicates", "diwan", "n_groups")])))
        add(("gold poems with a twin", 28, int(cc[("duplicates", "diwan", "n_gold_with_twin")])))
        ch = CC[(CC.check == "duplicates") & (CC.quantity == "R@5_change_if_twin_counts")].value
        add(("max R@5 change crediting twins (pts)", 0.4, round(100 * ch.max(), 1)))
        add(("systems unchanged crediting twins", 9, int((ch == 0).sum())))

    # explanation audit
    au_fn = os.path.join(ROOT, "human_checks", "explanation_audit", "answers.csv")
    if os.path.exists(au_fn):
        au = pd.read_csv(au_fn)
        au.columns = [x.strip().lstrip("﻿") for x in au.columns]
        add(("explanations audited", 50, len(au)))
        add(("faithful (%)", 60, round(100 * (au.faithful == 2).mean())))
        add(("minor error (%)", 32, round(100 * (au.faithful == 1).mean())))
        add(("misread (%)", 8, round(100 * (au.faithful == 0).mean())))
        add(("substantive additions (%)", 20, round(100 * (au.adds == 2).mean())))

    # spot check controls (Limitations)
    kf = os.path.join(ROOT, "spotcheck", "key.csv")
    af = os.path.join(ROOT, "spotcheck", "spotcheck_answers.csv")
    if os.path.exists(kf) and os.path.exists(af):
        sp = pd.read_csv(kf).merge(pd.read_csv(af), on="item_id")
        ctrl = sp[sp.kind == "control"]
        add(("spot-check controls judged correct", 11, int((ctrl.satisfies == 2).sum())))
        add(("spot-check controls", 12, len(ctrl)))
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
