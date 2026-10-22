"""The query-register x system interaction (paper section 5.1) and its robustness to the split
threshold.

On Fann or Flop, BM25 and Qwen3-8B do not differ significantly on the non-quoting subset
(exact McNemar p = 0.69, n = 164). What the data support is the interaction: BM25's sensitivity
to remembered wording is significantly larger than that of the strongest dense encoder.

Reported here, all from analysis/runs/*__gold.json.gz:

  1. Per system, the quotation effect  Delta = R@5(quote) - R@5(no-quote)  with bootstrap 95% CI.
  2. The difference-in-differences against the reference dense encoder (Qwen3-8B), with a paired
     bootstrap p-value. This is the interaction test.
  3. Exact McNemar for BM25 vs Qwen3-8B on every subset, including the non-significant ones.
  4. The same split repeated at n-gram thresholds 2..5.

Writes:
  results/derived/eacl_interaction.csv      per-system Delta, CI, DiD, p        (threshold 3)
  results/derived/eacl_threshold_sweep.csv  R@5 by subset x system x threshold  (thresholds 2..5)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DERIVED  # noqa: E402
from eacl_metrics import NICE, SYSTEMS, load_run, mcnemar  # noqa: E402

REF = "qwen3_8b"          # reference dense encoder: the strongest, and the one §5 contrasts
CORPUS = "fannorflop"     # the only corpus with both registers present
THRESHOLDS = (2, 3, 4, 5)
MAIN_THRESHOLD = 3        # the paper's definition of a verbatim quotation
NBOOT = 10000


def success_by_qid(system, corpus, mode="verses"):
    """qid -> 1.0 if the gold poem is in the top 5, else 0.0."""
    df = load_run(system, corpus, mode)
    if df is None:
        return None
    r = pd.to_numeric(df["rank_of_gold"], errors="coerce")
    return dict(zip(df["qid"], ((r <= 5) & r.notna()).astype(float)))


def main():
    qf = pd.read_csv(os.path.join(DERIVED, "query_features.csv"))
    qf = qf[qf.corpus == CORPUS].set_index("qid")
    succ = {s: success_by_qid(s, CORPUS) for s in SYSTEMS}
    succ = {s: v for s, v in succ.items() if v is not None}

    # threshold sweep
    sweep = []
    for thr in THRESHOLDS:
        quote = [q for q in qf.index if qf.loc[q, "lcn_verses"] >= thr]
        noq = [q for q in qf.index if qf.loc[q, "lcn_verses"] < thr]
        for s, v in succ.items():
            sweep.append({
                "system": NICE[s], "threshold": thr,
                "n_quote": len(quote), "n_noquote": len(noq),
                "R@5_quote": round(float(np.mean([v[q] for q in quote])), 3),
                "R@5_noquote": round(float(np.mean([v[q] for q in noq])), 3),
            })
    sweep = pd.DataFrame(sweep)

    # BM25 vs the reference encoder on each subset and threshold, with the paired test
    for thr in THRESHOLDS:
        quote = [q for q in qf.index if qf.loc[q, "lcn_verses"] >= thr]
        noq = [q for q in qf.index if qf.loc[q, "lcn_verses"] < thr]
        for label, ks in (("quote", quote), ("noquote", noq)):
            a = [succ["bm25"][q] for q in ks]
            b = [succ[REF][q] for q in ks]
            _, _, p = mcnemar(np.array(a, bool), np.array(b, bool))
            sweep.loc[(sweep.threshold == thr) & (sweep.system == "BM25"),
                      f"p_vs_{NICE[REF]}_{label}"] = round(p, 4)
    sweep.to_csv(os.path.join(DERIVED, "eacl_threshold_sweep.csv"), index=False)

    # interaction, at the paper's threshold and at every threshold in the sweep
    def interaction_at(thr):
        """Per-system quotation slope and its DiD against the reference encoder."""
        quote = [q for q in qf.index if qf.loc[q, "lcn_verses"] >= thr]
        noq = [q for q in qf.index if qf.loc[q, "lcn_verses"] < thr]
        # One shared set of bootstrap resamples for every system, so the DiD is paired: the
        # same queries are resampled for each system in each replicate.
        rng = np.random.default_rng(0)
        iq = rng.integers(0, len(quote), size=(NBOOT, len(quote)))
        inn = rng.integers(0, len(noq), size=(NBOOT, len(noq)))

        def delta_boot(system):
            vq = np.array([succ[system][q] for q in quote])
            vn = np.array([succ[system][q] for q in noq])
            return vq[iq].mean(1) - vn[inn].mean(1), vq.mean() - vn.mean()

        return quote, noq, delta_boot

    sweep_did = []
    for thr in THRESHOLDS:
        _, _, db = interaction_at(thr)
        rb, ro = db(REF)
        for s in succ:
            b, obs = db(s)
            dd = b - rb
            sweep_did.append({
                "threshold": thr, "system": NICE[s], "delta": round(float(obs), 3),
                "DiD_vs_ref": round(float(obs - ro), 3),
                "DiD_p": np.nan if s == REF else round(
                    2 * min(float(np.mean(dd <= 0)), float(np.mean(dd >= 0))), 4),
            })
    sweep_did = pd.DataFrame(sweep_did)
    sweep = sweep.merge(sweep_did, on=["threshold", "system"], how="left")
    sweep.to_csv(os.path.join(DERIVED, "eacl_threshold_sweep.csv"), index=False)

    quote, noq, delta_boot = interaction_at(MAIN_THRESHOLD)
    nq, nn = len(quote), len(noq)
    ref_boot, ref_obs = delta_boot(REF)
    rows = []
    for s in succ:
        b, obs = delta_boot(s)
        did = obs - ref_obs
        did_boot = b - ref_boot
        # two-sided bootstrap p for DiD != 0
        p = 2 * min(float(np.mean(did_boot <= 0)), float(np.mean(did_boot >= 0)))
        a_v = np.array([succ[s][q] for q in quote] + [succ[s][q] for q in noq], bool)
        r_v = np.array([succ[REF][q] for q in quote] + [succ[REF][q] for q in noq], bool)
        _, _, p_all = mcnemar(a_v, r_v)
        rows.append({
            "system": NICE[s],
            "R@5_quote": round(float(np.mean([succ[s][q] for q in quote])), 3),
            "R@5_noquote": round(float(np.mean([succ[s][q] for q in noq])), 3),
            "delta": round(float(obs), 3),
            "delta_lo": round(float(np.percentile(b, 2.5)), 3),
            "delta_hi": round(float(np.percentile(b, 97.5)), 3),
            "DiD_vs_ref": round(float(did), 3),
            "DiD_p": round(p, 4) if s != REF else np.nan,
            "p_vs_ref_all": round(p_all, 4) if s != REF else np.nan,
        })
    out = pd.DataFrame(rows).sort_values("delta", ascending=False)
    out.to_csv(os.path.join(DERIVED, "eacl_interaction.csv"), index=False)

    pd.set_option("display.width", 160)
    print(f"Interaction, {CORPUS} verse mode, split at lcn >= {MAIN_THRESHOLD} "
          f"(quote n={nq}, no-quote n={nn}); reference = {NICE[REF]}\n")
    print(out.to_string(index=False))
    print("\nThreshold sweep (R@5 by subset):\n")
    print(sweep.pivot_table(index="system", columns="threshold",
                            values=["R@5_quote", "R@5_noquote"]).to_string())
    print(f"\nwrote eacl_interaction.csv, eacl_threshold_sweep.csv -> {DERIVED}")


if __name__ == "__main__":
    main()
