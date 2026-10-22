"""Does lexical overlap still predict success once correlated query and target features are
controlled?

Unit: one (query, system) pair, verse mode, Recall@5 success (0/1), both known-item sets.
Predictors (z-scored within the pooled data):
  overlap block   C_verses (containment), lcn (longest verbatim span), quoted_frac
  q_len           query length in tokens
  pn              proper-noun tokens in the query (rb_proper_nouns.csv, if present)
  target block    gold_nverses, gold_confus (query-free TF-IDF nearest-neighbour similarity)
  corpus          fixed effect (Fann or Flop vs Diwan)

Models
  (1) per system: logit(success) ~ all predictors + corpus. Reports the containment coefficient
      with and without controls.
  (2) pooled: logit ~ (overlap + controls) * is_bm25 + corpus, cluster-robust SEs by query. The
      containment x BM25 interaction is the multivariate version of the quotation interaction.
  (3) group Shapley decomposition of McFadden pseudo-R^2 per system over the predictor groups
      {overlap, length, proper nouns, target, corpus}.

Writes results/derived/rb_multivariate_{coef,shapley,interaction}.csv
"""
import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DERIVED  # noqa: E402
from eacl_metrics import NICE, SYSTEMS, load_run  # noqa: E402

warnings.filterwarnings("ignore")


def data():
    qf = pd.read_csv(os.path.join(DERIVED, "query_features.csv"))
    cv = pd.read_csv(os.path.join(DERIVED, "rb_query_covariates.csv"))
    d = qf[["corpus", "qid", "C_verses"]].merge(
        cv[["corpus", "qid", "q_len", "lcn", "quoted_frac", "gold_nverses", "gold_confus"]],
        on=["corpus", "qid"])
    pn_fn = os.path.join(DERIVED, "rb_proper_nouns.csv")
    has_pn = os.path.exists(pn_fn)
    if has_pn:
        d = d.merge(pd.read_csv(pn_fn)[["corpus", "qid", "pn_llm"]], on=["corpus", "qid"])
        d = d.rename(columns={"pn_llm": "pn"})
    rows = []
    for s in SYSTEMS:
        for c in ("fannorflop", "diwan"):
            r = load_run(s, c, "verses")
            rk = pd.to_numeric(r.rank_of_gold, errors="coerce")
            rows.append(pd.DataFrame({"corpus": c, "qid": r.qid, "system": s,
                                      "y": ((rk <= 5) & rk.notna()).astype(int)}))
    y = pd.concat(rows)
    d = y.merge(d, on=["corpus", "qid"])
    d["fof"] = (d.corpus == "fannorflop").astype(int)
    d["bm25"] = (d.system == "bm25").astype(int)
    d["qkey"] = d.corpus + ":" + d.qid
    num = ["C_verses", "lcn", "quoted_frac", "q_len", "gold_nverses", "gold_confus"] + \
          (["pn"] if has_pn else [])
    for c in num:
        d[c + "_z"] = (d[c] - d[c].mean()) / d[c].std()
    return d, has_pn


def groups(has_pn):
    g = {"overlap": ["C_verses_z", "lcn_z", "quoted_frac_z"], "length": ["q_len_z"],
         "target": ["gold_nverses_z", "gold_confus_z"], "corpus": ["fof"]}
    if has_pn:
        g["proper_nouns"] = ["pn_z"]
    return g


def fit(formula, df, cluster=None):
    m = smf.logit(formula, df)
    kw = {"disp": 0, "maxiter": 200}
    if cluster is not None:
        return m.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df[cluster])[0]}, **kw)
    return m.fit(**kw)


def mcfadden(df, cols):
    if not cols:
        return 0.0
    try:
        return fit("y ~ " + " + ".join(cols), df).prsquared
    except Exception:
        return np.nan


def main():
    d, has_pn = data()
    G = groups(has_pn)
    allcols = [c for v in G.values() for c in v]
    coef, shap = [], []
    for s in SYSTEMS:
        ds = d[d.system == s]
        m0 = fit("y ~ C_verses_z + fof", ds, "qkey")
        m1 = fit("y ~ " + " + ".join(allcols), ds, "qkey")
        # containment alone as the overlap measure, with every non-overlap control
        ctrl = [c for k, v in G.items() if k != "overlap" for c in v]
        m2 = fit("y ~ C_verses_z + " + " + ".join(ctrl), ds, "qkey")
        rec = {"system": NICE[s]}
        for name, m in (("C_bivariate", m0), ("C_controlled", m2)):
            rec[f"{name}_b"] = m.params["C_verses_z"]
            rec[f"{name}_p"] = m.pvalues["C_verses_z"]
        for c in ctrl:
            rec[f"{c}_b"] = m2.params[c]
            rec[f"{c}_p"] = m2.pvalues[c]
        rec["pseudoR2_full"] = m1.prsquared
        coef.append(rec)
        # group Shapley over pseudo-R2
        names = list(G)
        cache = {}

        def v(S):
            k = tuple(sorted(S))
            if k not in cache:
                cache[k] = mcfadden(ds, [c for g in k for c in G[g]])
            return cache[k]
        n = len(names)
        for g in names:
            phi = 0.0
            others = [x for x in names if x != g]
            for r in range(n):
                for S in itertools.combinations(others, r):
                    w = np.math.factorial(r) * np.math.factorial(n - r - 1) / np.math.factorial(n)
                    phi += w * (v(S + (g,)) - v(S))
            shap.append({"system": NICE[s], "group": g, "shapley_R2": phi,
                         "share": phi / v(tuple(names))})
    coef = pd.DataFrame(coef)
    shap = pd.DataFrame(shap)

    # pooled interaction model: does BM25 depend on containment more than dense encoders,
    # holding the controls fixed (and letting every control's effect also differ for BM25)?
    ctrl = [c for k, v in G.items() if k != "overlap" for c in v]
    f = ("y ~ C_verses_z + C_verses_z:bm25 + " + " + ".join(f"{c} + {c}:bm25" for c in ctrl)
         + " + C(system)")  # system dummies absorb the BM25 main effect
    mp = fit(f, d, "qkey")
    inter = {"term": "C_verses_z:bm25", "b": mp.params["C_verses_z:bm25"],
             "se": mp.bse["C_verses_z:bm25"], "p": mp.pvalues["C_verses_z:bm25"],
             "n_obs": int(mp.nobs), "n_queries": d.qkey.nunique()}
    # same, FoF only (the corpus where both registers exist)
    df_ = d[d.fof == 1]
    ctrl_f = [c for c in ctrl if c != "fof"]
    mf = fit("y ~ C_verses_z + C_verses_z:bm25 + " + " + ".join(f"{c} + {c}:bm25" for c in ctrl_f)
             + " + C(system)", df_, "qkey")
    inter_f = {"term": "C_verses_z:bm25 (FoF only)", "b": mf.params["C_verses_z:bm25"],
               "se": mf.bse["C_verses_z:bm25"], "p": mf.pvalues["C_verses_z:bm25"],
               "n_obs": int(mf.nobs), "n_queries": df_.qkey.nunique()}
    coef.to_csv(os.path.join(DERIVED, "rb_multivariate_coef.csv"), index=False)
    shap.to_csv(os.path.join(DERIVED, "rb_multivariate_shapley.csv"), index=False)
    pd.DataFrame([inter, inter_f]).to_csv(os.path.join(DERIVED, "rb_multivariate_interaction.csv"),
                                          index=False)
    pd.set_option("display.width", 220)
    print("proper nouns included:", has_pn)
    print(coef[["system", "C_bivariate_b", "C_bivariate_p", "C_controlled_b", "C_controlled_p",
                "pseudoR2_full"]].round(4).to_string(index=False))
    print(coef[[c for c in coef.columns if c.endswith("_b") or c == "system"]].round(3)
          .to_string(index=False))
    print(shap.pivot(index="system", columns="group", values="share").round(2).to_string())
    print(pd.DataFrame([inter, inter_f]).round(5).to_string(index=False))


if __name__ == "__main__":
    main()
