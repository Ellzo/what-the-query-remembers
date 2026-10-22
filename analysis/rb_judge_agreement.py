"""Track B system orderings under three judges and two rubrics.

Judges: judge1 = Gemini 3.5 Flash (trackB_rejudged.csv), judge2 = Gemma-3-27B-it
(trackB_judge2.csv), judge3o / judge3n = Qwen3-32B with the original / neutral rubric
(trackB_judge3_{original,neutral}.csv). Unparsed rows are dropped. Every pair of judges is
compared on the cells both scored.

The `analysis` column groups the rows: R4a = system means, ranks and pairwise agreement under the
original rubric; R4b = the primary judge's ordering from the aboutness aspects alone; R4c = the
neutral rubric and its shift from the original.

Writes results/derived/rb_judge_agreement.csv
"""
import itertools
import os
import sys

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DERIVED  # noqa: E402

KEY = ["system", "corpus", "qid"]
FILES = {"judge1": "trackB_rejudged.csv", "judge2": "trackB_judge2.csv",
         "judge3o": "trackB_judge3_original.csv", "judge3n": "trackB_judge3_neutral.csv"}
ABOUT = ["semantic_relevance", "thematic_coherence", "semantic_coverage", "precision"]


def load(name):
    d = pd.read_csv(os.path.join(DERIVED, FILES[name]))
    if "parsed" in d:
        n0 = len(d)
        d = d[d.parsed.astype(str) == "True"]
        print(f"{name}: dropped {n0 - len(d)} unparsed rows of {n0}")
    return d


def main():
    J = {n: load(n) for n in FILES}
    rows = []
    add = lambda *r: rows.append(dict(zip(["analysis", "corpus", "judge_or_pair", "system",  # noqa
                                           "quantity", "value"], r)))
    # R4a / R4c: pairwise, on the intersection of the two judges' cells
    for a, b in itertools.combinations(FILES, 2):
        common = J[a][KEY].merge(J[b][KEY])
        ana = "R4c" if "judge3n" in (a, b) else "R4a"
        for c in ("fannorflop", "diwan"):
            ma = J[a].merge(common, on=KEY)
            mb = J[b].merge(common, on=KEY)
            sa = ma[ma.corpus == c].groupby("system").overall_quality.mean()
            sb = mb[mb.corpus == c].groupby("system").overall_quality.mean()
            t, p = kendalltau(sa, sb.reindex(sa.index))
            cell = ma[ma.corpus == c].merge(mb[mb.corpus == c], on=KEY)
            rho = spearmanr(cell.overall_quality_x, cell.overall_quality_y)[0]
            pair = f"{a}-{b}"
            add(ana, c, pair, "all", "kendall_tau", t)
            add(ana, c, pair, "all", "kendall_p", p)
            add(ana, c, pair, "all", "cell_spearman", rho)
            add(ana, c, pair, "all", "n_cells", len(cell))
    # per-judge means and BM25 rank, on the 100-query subset all four judges share
    common = J["judge1"][KEY]
    for n in FILES:
        common = common.merge(J[n][KEY])
    for n in FILES:
        m = J[n].merge(common, on=KEY)
        for c in ("fannorflop", "diwan"):
            s = m[m.corpus == c].groupby("system").overall_quality.mean()
            r = s.rank(ascending=False)
            ana = "R4c" if n == "judge3n" else "R4a"
            for sysname, v in s.items():
                add(ana, c, n, sysname, "mean_overall", v)
            add(ana, c, n, "bm25", "rank", r["bm25"])
    # R4c: shift from the original to the neutral rubric, same model
    m = J["judge3o"].merge(J["judge3n"], on=KEY, suffixes=("_o", "_n"))
    for (c, sname), g in m.groupby(["corpus", "system"]):
        add("R4c", c, "judge3n-judge3o", sname, "mean_shift",
            (g.overall_quality_n - g.overall_quality_o).mean())
    # R4b: judge 1, all 200 queries
    d = J["judge1"].copy()
    d["about"] = d[ABOUT].mean(axis=1)
    for c in ("fannorflop", "diwan"):
        g = d[d.corpus == c].groupby("system")[["about", "overall_quality"]].mean()
        add("R4b", c, "judge1", "bm25", "rank_aboutness", g.about.rank(ascending=False)["bm25"])
        add("R4b", c, "judge1", "all", "tau_aboutness_vs_overall",
            kendalltau(g.about, g.overall_quality)[0])
    others = ["semantic_relevance", "thematic_coherence", "metaphor_understanding",
              "semantic_coverage", "precision"]
    d["fof"] = (d.corpus == "fannorflop").astype(int)
    full = smf.ols("overall_quality ~ " + " + ".join(others + ["direct_textual_relevance", "fof"]),
                   d).fit()
    red = smf.ols("overall_quality ~ " + " + ".join(others + ["fof"]), d).fit()
    add("R4b", "all", "judge1", "all", "coef_direct_textual", full.params["direct_textual_relevance"])
    add("R4b", "all", "judge1", "all", "r2_with", full.rsquared)
    add("R4b", "all", "judge1", "all", "r2_without", red.rsquared)
    add("R4b", "all", "judge1", "all", "spearman_literal_vs_semrel",
        spearmanr(d.direct_textual_relevance, d.semantic_relevance)[0])
    out = pd.DataFrame(rows)
    out["value"] = out.value.astype(float).round(4)
    out.to_csv(os.path.join(DERIVED, "rb_judge_agreement.csv"), index=False)
    pd.set_option("display.width", 200)
    print(out[out.quantity.isin(["kendall_tau", "rank", "rank_aboutness", "cell_spearman",
                                 "coef_direct_textual", "r2_with", "r2_without"])]
          .to_string(index=False))


if __name__ == "__main__":
    main()
