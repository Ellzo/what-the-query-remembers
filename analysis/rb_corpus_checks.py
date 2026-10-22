"""Corpus-level checks that need the poem metadata in data/: era, poet frequency, target
selection, the Fann or Flop length filter and duplicate poems in Diwan.

The results go to one long-format CSV so that verify_paper.py can check them without the corpora.

  era         logit of R@5 success on Fann or Flop, era as an ordinal 0-3, controlling for
              query-verse containment and query length, SEs clustered by query
  poet_freq   |Spearman rho| between a poet's poem count and R@5 success, per system
  targets     whether the gold poems differ from the rest of each corpus (chi-square and
              Mann-Whitney p-values)
  fof_filter  the Fann or Flop subset is the poems with at most 16 verses
  duplicates  Diwan poems whose normalised text is identical to another poem's, and how much
              verse-mode R@5 changes if a duplicate of the gold poem is counted as a hit

Writes:
  results/derived/rb_corpus_checks.csv   check, system, quantity, value
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, DERIVED, normalize_ar  # noqa: E402
from eacl_metrics import SYSTEMS, load_run  # noqa: E402

ERA = {"قبل الإسلام": 0, "الشعراء المخضرمون": 0, "العصر الإسلامي": 0, "العصر الأموي": 0,
       "العصر العباسي": 1, "المغرب والأندلس": 2, "الدولة الفاطمية": 2, "الدولة الايوبية": 2,
       "الدولة المملوكية": 2, "عصر بين الدولتين": 2, "العصر العثماني": 3, "العصر الحديث": 3}


def hit5(run):
    return (pd.to_numeric(run.rank_of_gold) <= 5).astype(int)


def main():
    rows = []
    add = lambda check, system, q, v: rows.append((check, system, q, float(v)))  # noqa: E731

    ff = pd.read_csv(os.path.join(DATA, "flipp_flop_ai.csv"))
    ff["pid"] = ff.id.astype(str)
    dw = pd.read_csv(os.path.join(DATA, "diwan_with_explanations.csv"))
    dw["pid"] = dw.poem_id.astype(str)

    # era, Fann or Flop only (Diwan's era labels are too coarse for this)
    era = ff.set_index("pid").era.map(ERA)
    qf = pd.read_csv(os.path.join(DERIVED, "query_features.csv"))
    qf = qf[qf.corpus == "fannorflop"].set_index("qid")
    parts = []
    for s in SYSTEMS:
        r = load_run(s, "fannorflop", "verses").set_index("qid")
        parts.append(pd.DataFrame({"system": s, "y": hit5(r), "era": r.gold_pid.astype(str).map(era),
                                   "Cv": qf.C_verses, "qlen": qf.q_len}))
    E = pd.concat(parts).reset_index()
    E["Cz"] = (E.Cv - E.Cv.mean()) / E.Cv.std()
    E["bm25"] = (E.system == "bm25").astype(int)
    dense = E[E.system != "bm25"]
    m = smf.logit("y ~ era + Cz + qlen + C(system)", dense).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(dense.qid)[0]})
    add("era", "dense", "coef", m.params.era)
    add("era", "dense", "p", m.pvalues.era)
    b = E[E.system == "bm25"]
    mb = smf.logit("y ~ era + Cz + qlen", b).fit(disp=0)
    add("era", "bm25", "coef", mb.params.era)
    add("era", "bm25", "p", mb.pvalues.era)
    mi = smf.logit("y ~ era + era:bm25 + Cz + Cz:bm25 + qlen + C(system)", E).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(E.qid)[0]})
    add("era", "interaction", "coef", mi.params["era:bm25"])
    add("era", "interaction", "p", mi.pvalues["era:bm25"])
    add("era", "all", "n_queries", E.qid.nunique())

    # poet frequency as a proxy for pretraining exposure
    cnt = ff.author.value_counts()
    author = ff.set_index("pid").author
    for s in SYSTEMS:
        r = load_run(s, "fannorflop", "verses")
        rho, p = spearmanr(r.gold_pid.astype(str).map(author).map(cnt), hit5(r))
        add("poet_freq", s, "abs_rho", abs(rho))
        add("poet_freq", s, "p", p)
    add("poet_freq", "all", "min_poems_per_poet", cnt.min())
    add("poet_freq", "all", "max_poems_per_poet", cnt.max())

    # were the gold poems a biased draw from each corpus?
    gq = pd.read_csv(os.path.join(DATA, "accuracy_prompts.csv")).drop_duplicates("prompt")
    is_gold = ff.pid.isin(gq.poem_id.astype(str))
    for c in ("era", "genre"):
        add("targets", "fannorflop", f"p_{c}", chi2_contingency(pd.crosstab(is_gold, ff[c]))[1])
    add("targets", "fannorflop", "p_length",
        mannwhitneyu(ff[is_gold].verse_count, ff[~is_gold].verse_count).pvalue)
    dq = pd.read_csv(os.path.join(DATA, "diwan_prompts_final.csv"))
    is_gold = dw.pid.isin(dq.poem_id.astype(str))
    for c in ("poet_era", "poem_genre", "poet_name"):
        add("targets", "diwan", f"p_{c}", chi2_contingency(pd.crosstab(is_gold, dw[c]))[1])
    add("targets", "diwan", "p_length",
        mannwhitneyu(dw[is_gold].verses_count, dw[~is_gold].verses_count).pvalue)

    # the Fann or Flop subset used here
    add("fof_filter", "all", "n_poems", len(ff))
    add("fof_filter", "all", "max_verses", ff.verse_count.max())

    # duplicate poems in Diwan
    key = dw.verses_text.map(normalize_ar)
    size = key.map(key.value_counts())
    add("duplicates", "diwan", "n_poems_with_twin", int((size > 1).sum()))
    add("duplicates", "diwan", "n_groups", int((key.value_counts() > 1).sum()))
    twins = {}
    for k, g in dw.groupby(key):
        if len(g) > 1:
            for pid in g.pid:
                twins[pid] = set(g.pid) - {pid}
    gold_ids = dq.poem_id.astype(str)
    add("duplicates", "diwan", "n_gold_with_twin", int(gold_ids.drop_duplicates().isin(twins).sum()))
    for s in SYSTEMS:
        r = load_run(s, "diwan", "verses")
        strict = hit5(r).mean()
        loose = np.mean([
            any(p == str(g) or p in twins.get(str(g), ()) for p in list(map(str, top))[:5])
            for g, top in zip(r.gold_pid, r.top_pids)])
        add("duplicates", s, "R@5_change_if_twin_counts", loose - strict)

    out = pd.DataFrame(rows, columns=["check", "system", "quantity", "value"])
    out.to_csv(os.path.join(DERIVED, "rb_corpus_checks.csv"), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
