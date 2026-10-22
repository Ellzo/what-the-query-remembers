"""The quotation-slope interaction (paper section 5.1) repeated on R@1, MRR and nDCG@10.

Same test as interaction.py (split at a verbatim >=3-token span, paired bootstrap of the
difference-in-differences against Qwen3-8B, 10^4 resamples, seed 0, Holm across the nine
comparisons within each metric) for every per-query metric the paper reports. R@5 is included as
a check: it must reproduce eacl_interaction.csv exactly.

Writes results/derived/rb_slope_metrics.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DERIVED  # noqa: E402
from eacl_metrics import NICE, SYSTEMS, load_run  # noqa: E402
from interaction import CORPUS, MAIN_THRESHOLD, NBOOT, REF  # noqa: E402

METRICS = {
    "R@1": lambda r: (r == 1).astype(float),
    "R@5": lambda r: (r <= 5).astype(float),
    "MRR": lambda r: 1.0 / r,
    "nDCG@10": lambda r: np.where(r <= 10, 1.0 / np.log2(r + 1), 0.0),
}


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (len(ps) - k) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    qf = pd.read_csv(os.path.join(DERIVED, "query_features.csv"))
    qf = qf[qf.corpus == CORPUS].set_index("qid")
    quote = [q for q in qf.index if qf.loc[q, "lcn_verses"] >= MAIN_THRESHOLD]
    noq = [q for q in qf.index if qf.loc[q, "lcn_verses"] < MAIN_THRESHOLD]

    ranks = {}
    for s in SYSTEMS:
        df = load_run(s, CORPUS, "verses")
        r = pd.to_numeric(df["rank_of_gold"], errors="coerce").fillna(10**6).to_numpy()
        ranks[s] = dict(zip(df["qid"], r))

    rows = []
    for mname, f in METRICS.items():
        # same resampling scheme as interaction.py, so R@5 reproduces it
        rng = np.random.default_rng(0)
        iq = rng.integers(0, len(quote), size=(NBOOT, len(quote)))
        inn = rng.integers(0, len(noq), size=(NBOOT, len(noq)))

        def delta(s):
            vq = f(np.array([ranks[s][q] for q in quote]))
            vn = f(np.array([ranks[s][q] for q in noq]))
            return vq[iq].mean(1) - vn[inn].mean(1), vq.mean(), vn.mean()

        rb, rq, rn = delta(REF)
        block = []
        for s in SYSTEMS:
            b, mq, mn = delta(s)
            dd = b - rb
            p = np.nan if s == REF else min(1.0, 2 * min(np.mean(dd <= 0), np.mean(dd >= 0)))
            block.append({"metric": mname, "system": NICE[s], "quote": mq, "noquote": mn,
                          "delta": mq - mn, "DiD_vs_ref": (mq - mn) - (rq - rn),
                          "DiD_lo": np.percentile(dd, 2.5), "DiD_hi": np.percentile(dd, 97.5),
                          "p": p})
        b = pd.DataFrame(block)
        mask = b.system != NICE[REF]
        b.loc[mask, "p_holm"] = holm(b.loc[mask, "p"])
        rows.append(b)
    out = pd.concat(rows).round(4)
    out.to_csv(os.path.join(DERIVED, "rb_slope_metrics.csv"), index=False)
    pd.set_option("display.width", 200)
    print(f"split lcn>={MAIN_THRESHOLD}: quote n={len(quote)}, no-quote n={len(noq)}; ref={NICE[REF]}")
    for m in METRICS:
        print(f"\n== {m}")
        print(out[out.metric == m].drop(columns="metric").sort_values("delta", ascending=False)
              .to_string(index=False))


if __name__ == "__main__":
    main()
