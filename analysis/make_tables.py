"""Emit every LaTeX table in the paper from results/derived/*.csv.

No number is ever typed into the .tex by hand: `make_tables.py` writes paper/tab_*.tex and
main.tex \\input{}s them. Running this after any change to the pipeline refreshes the paper.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DERIVED, ROOT  # noqa: E402
from eacl_metrics import stars  # noqa: E402

PAPER = os.path.join(ROOT, "paper")
NA = "n/a"
ORDER = ["Qwen3-8B", "BGE-M3", "mE5-large", "jina-v3", "silma-sts", "gte", "Qwen3-0.6B",
         "LaBSE", "Muffakir-v2", "BM25"]


def w(name, body):
    with open(os.path.join(PAPER, name), "w") as f:
        f.write(body)
    print("wrote", name)


def t1_data():
    qf = pd.read_csv(os.path.join(DERIVED, "query_features.csv"))
    rows = []
    for corp, label in [("fannorflop", "FannOrFlop"), ("diwan", "Diwan")]:
        g = qf[qf.corpus == corp]
        rows.append((label, len(g), g.q_len.mean(), 100 * g.has_verbatim3.mean(),
                     g.C_verses.mean(), g.C_expl_ai.mean(),
                     g.C_expl_ff.mean() if corp == "fannorflop" else np.nan))
    b = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
         "Query set & $n$ & len & \\%\\,quote & $C_{\\mathrm{vrs}}$ & $C_{\\mathrm{ai}}$ & "
         "$C_{\\mathrm{ff}}$ \\\\", "\\midrule"]
    for lab, n, ln, vq, cv, ca, cf in rows:
        cfs = NA if np.isnan(cf) else f"{cf:.3f}"
        b.append(f"{lab} & {n} & {ln:.1f} & {vq:.1f} & {cv:.3f} & {ca:.3f} & {cfs} \\\\")
    b += ["\\bottomrule", "\\end{tabular}"]
    w("tab_data.tex", "\n".join(b))


def t2_tracka():
    A = pd.read_csv(os.path.join(DERIVED, "eacl_trackA.csv"))
    M = pd.read_csv(os.path.join(DERIVED, "eacl_mcnemar.csv"))
    A = A[A.subset == "all"]
    # Stars use Holm-corrected p-values, with the family being the 10 systems compared against
    # their own verse baseline under one (corpus, contrast). Uncorrected stars would overstate
    # three FannOrFlop AI-explanation contrasts.
    from multiplicity import holm
    M = M[M.subset == "all"].copy()
    M["p_holm"] = M.groupby(["corpus", "contrast"])["p"].transform(lambda s: holm(s.values))
    M["sig"] = M["p_holm"].map(stars)
    sig = {(r.system, r.corpus, r.contrast): r.sig for r in M.itertuples()}

    def cell(sysname, corpus, mode):
        r = A[(A.system == sysname) & (A.corpus == corpus) & (A["mode"] == mode)]
        if r.empty:
            return NA
        r = r.iloc[0]
        s = sig.get((sysname, corpus, f"{mode}_vs_verses"), "")
        s = "" if (isinstance(s, float) and np.isnan(s)) or s == "" else f"$^{{{s}}}$"
        return f"{r['R@5']:.3f}{s}"

    b = ["\\begin{tabular}{l ccc c cc}", "\\toprule",
         "& \\multicolumn{3}{c}{\\textit{FannOrFlop} (tip-of-the-tongue)} & &"
         " \\multicolumn{2}{c}{\\textit{Diwan} (abstract)} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){6-7}",
         "System & verses & AI expl. & FoF expl. & & verses & AI expl. \\\\", "\\midrule"]
    for s in ORDER:
        if s == "BM25":
            b.append("\\midrule")
        nm = "\\textbf{BM25}" if s == "BM25" else s
        b.append(f"{nm} & {cell(s,'fannorflop','verses')} & {cell(s,'fannorflop','expl_ai')} & "
                 f"{cell(s,'fannorflop','expl_ff')} & & {cell(s,'diwan','verses')} & "
                 f"{cell(s,'diwan','expl_ai')} \\\\")
    b += ["\\bottomrule", "\\end{tabular}"]
    w("tab_main.tex", "\n".join(b))


def t3_controlled():
    A = pd.read_csv(os.path.join(DERIVED, "eacl_trackA.csv"))
    Me = pd.read_csv(os.path.join(DERIVED, "eacl_mechanism.csv"))
    ff = A[(A.corpus == "fannorflop") & (A["mode"] == "verses")].set_index(["system", "subset"])
    dw = A[(A.corpus == "diwan") & (A["mode"] == "verses") & (A.subset == "all")] \
        .set_index("system")["R@5"]
    mech = Me[(Me.corpus == "diwan") & (Me["mode"] == "expl_ai")].set_index("system")

    # quotation slope and its difference against the reference dense encoder (§5 interaction)
    itr = pd.read_csv(os.path.join(DERIVED, "eacl_interaction.csv")).set_index("system")

    b = ["\\begin{tabular}{l cccc cc}", "\\toprule",
         "& \\multicolumn{2}{c}{FannOrFlop R@5} & \\multicolumn{2}{c}{slope} &"
         " \\multicolumn{2}{c}{$\\rho$} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
         "System & quote & no-qt. & $\\Delta$ & vs.\\,ref & lex. & emb. \\\\",
         "& \\scriptsize $n{=}80$ & \\scriptsize $n{=}164$ & & & & \\\\",
         "\\midrule"]
    for s in ORDER:
        if s == "BM25":
            b.append("\\midrule")
        try:
            q, nq = ff.loc[(s, "quote"), "R@5"], ff.loc[(s, "noquote"), "R@5"]
        except KeyError:
            continue
        nm = "\\textbf{BM25}" if s == "BM25" else s
        rl = mech.loc[s, "rho_lexical"] if s in mech.index else np.nan
        re_ = mech.loc[s, "rho_embedding"] if s in mech.index else np.nan
        lex = NA if np.isnan(rl) else (f"\\textbf{{{rl:.2f}}}" if rl > re_ else f"{rl:.2f}")
        emb = NA if np.isnan(re_) else (f"\\textbf{{{re_:.2f}}}" if re_ >= rl else f"{re_:.2f}")
        if s == "gte":            # measure uses gte's own encoder; flag it, see Appendix C
            emb += "$^{\\dagger}$"
        if s in itr.index:
            d = f"{itr.loc[s, 'delta']:.2f}"
            p = itr.loc[s, "DiD_p"]
            if np.isnan(p):                       # the reference encoder itself
                did = "ref."
            else:
                did = f"${itr.loc[s, 'DiD_vs_ref']:+.2f}$" + (f"$^{{{stars(p)}}}$" if stars(p) else "")
                if p < .05:
                    did = f"\\textbf{{{did}}}"
        else:
            d = did = NA
        b.append(f"{nm} & {q:.3f} & {nq:.3f} & {d} & {did} & {lex} & {emb} \\\\")
    b += ["\\bottomrule", "\\end{tabular}"]
    w("tab_controlled.tex", "\n".join(b))


def t4_trackb():
    fb = os.path.join(DERIVED, "eacl_trackB.csv")
    if not os.path.exists(fb):
        print("skip tab_trackb (judge still running)")
        return
    B = pd.read_csv(fb)
    A = pd.read_csv(os.path.join(DERIVED, "eacl_trackA.csv"))
    A = A[(A["mode"] == "verses") & (A.subset == "all")].set_index(["corpus", "system"])["R@5"]
    hum = {}
    hp = os.path.join(DERIVED, "human_annotation_cells.csv")
    if os.path.exists(hp):
        H = pd.read_csv(hp)
        H["sys"] = H.system.map({"bm25": "BM25", "jina": "jina-v3", "hybrid": "hybrid"})
        hum = H.groupby(["corpus", "sys"])["score"].mean().to_dict()

    b = ["\\begin{tabular}{l cc c cc}", "\\toprule",
         "& \\multicolumn{2}{c}{\\textit{FannOrFlop}} & & \\multicolumn{2}{c}{\\textit{Diwan}} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){5-6}",
         "System & Track A & Track B & & Track A & Track B \\\\", "\\midrule"]
    for s in ORDER:
        if s == "BM25":
            b.append("\\midrule")
        nm = "\\textbf{BM25}" if s == "BM25" else s
        cells = []
        for c in ("fannorflop", "diwan"):
            r = B[(B.corpus == c) & (B.system == s)]
            ta = A.get((c, s), np.nan)
            tb = r["overall_quality"].iloc[0] if not r.empty else np.nan
            cells += [f"{ta:.3f}" if not np.isnan(ta) else NA,
                      f"{tb:.2f}" if not np.isnan(tb) else NA]
        b.append(f"{nm} & {cells[0]} & {cells[1]} & & {cells[2]} & {cells[3]} \\\\")
    if hum:
        b.append("\\midrule\n\\multicolumn{6}{l}{\\textit{blind human relevance of top-3 "
                 "(0 to 2), 20 queries}} \\\\")
        for s in ("jina-v3", "hybrid", "BM25"):
            f_ = hum.get(("fannorflop", s), np.nan)
            d_ = hum.get(("diwan", s), np.nan)
            nm = "\\textbf{BM25}" if s == "BM25" else s
            b.append(f"{nm} & \\multicolumn{{2}}{{c}}{{{f_:.2f}}} & & "
                     f"\\multicolumn{{2}}{{c}}{{{d_:.2f}}} \\\\")
    b += ["\\bottomrule", "\\end{tabular}"]
    w("tab_trackb.tex", "\n".join(b))


def t5_fusion():
    F = pd.read_csv(os.path.join(DERIVED, "eacl_fusion.csv"))
    F = F[F.model == "Qwen3-8B"]
    rows = [("verses", "verses only"), ("expl_ai", "AI expl.\\ only"),
            ("expl_ff", "FoF expl.\\ only"), ("fuse_verses+expl_ai", "vrs.\\,$\\oplus$\\,AI expl."),
            ("fuse_verses+expl_ff", "vrs.\\,$\\oplus$\\,FoF expl."),
            ("hybrid_bm25+dense", "vrs.\\,$\\oplus$\\,BM25"),
            ("hybrid_bm25+dense+expl_ai", "vrs.\\,$\\oplus$\\,BM25\\,$\\oplus$\\,AI expl."),
            ("hybrid_bm25+dense+expl_ff", "vrs.\\,$\\oplus$\\,BM25\\,$\\oplus$\\,FoF expl.")]

    def g(variant, corpus, subset):
        r = F[(F.variant == variant) & (F.corpus == corpus) & (F.subset == subset)]
        return f"{r['R@5'].iloc[0]:.3f}" if not r.empty else NA

    b = ["\\begin{tabular}{l ccc c c}", "\\toprule",
         "& \\multicolumn{3}{c}{\\textit{FannOrFlop}} & & \\textit{Diwan} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){6-6}",
         "Index configuration & all & quote & no-qt. & & all \\\\", "\\midrule"]
    for v, lab in rows:
        b.append(f"{lab} & {g(v,'fannorflop','all')} & {g(v,'fannorflop','quote')} & "
                 f"{g(v,'fannorflop','noquote')} & & {g(v,'diwan','all')} \\\\")
    b += ["\\bottomrule", "\\end{tabular}"]
    w("tab_fusion.tex", "\n".join(b))


def t6_edits():
    """Edits to the same queries: a verbatim half-verse added, or the quoted wording rewritten."""
    C = pd.read_csv(os.path.join(DERIVED, "rb_counterfactual.csv"))
    C = C[(C.metric == "R@5") & (C.subset == "all")].set_index(["variant", "system"])
    cols = ["inj_open", "inj_mid", "inj_final", "hand_para"]

    def nz(x):                          # signed, no leading zero, to fit one column
        t = f"{abs(x):.3f}".replace("0.", ".", 1)
        return ("{+}" if x >= 0 else "{-}") + t
    b = ["\\begin{tabular}{l ccc c}", "\\toprule",
         "& \\multicolumn{3}{c}{half-verse added, $n{=}164$} & rewritten \\\\",
         "\\cmidrule(lr){2-4}",
         "System & opening & middle & final & $n{=}79$ \\\\", "\\midrule"]
    for s in ORDER:
        if s == "BM25":
            b.append("\\midrule")
        nm = "\\textbf{BM25}" if s == "BM25" else s
        b.append(nm + " & " + " & ".join(f"${nz(C.loc[(v, s), 'delta'])}$" for v in cols) + " \\\\")
    b.append("\\midrule")
    b.append("difference & " + " & ".join(
        f"${nz(C.loc[(v, 'BM25'), 'DiD_vs_ref'])}^{{{stars(C.loc[(v, 'BM25'), 'DiD_p'])}}}$"
        for v in cols) + " \\\\")
    b += ["\\bottomrule", "\\end{tabular}"]
    w("tab_edits.tex", "\n".join(b))


if __name__ == "__main__":
    t1_data()
    t2_tracka()
    t3_controlled()
    t4_trackb()
    t5_fusion()
    t6_edits()
