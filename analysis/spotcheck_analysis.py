"""Analyse the blinded known-item spot check: how often is a Track A "failure" actually correct?

Reads spotcheck/key.csv (which item was a failure and which a control) and the annotator's export
(spotcheck/spotcheck_answers.csv, columns item_id,satisfies,note). The annotator never saw that
distinction.

Reported:
  * control pass rate: items where the gold poem itself was shown. These should score 2. A low
    rate means the annotator is strict or the task is unclear, and the failure estimate below is
    then a lower bound.
  * failure "acceptable" rate: failures scored 2 (strict) and 2 or 1 (lenient). This is the share
    of apparent Track A failures that are unlabelled correct answers, i.e. how much single-gold
    known-item evaluation understates absolute Recall.
  * a corrected Recall@5 estimate for the system the sample was drawn from.

Writes results/derived/eacl_spotcheck.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import beta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DERIVED, ROOT  # noqa: E402

SPOT = os.path.join(ROOT, "spotcheck")
SYSTEM = "Qwen3-8B"


def wilson(k, n):
    """Jeffreys interval; avoids the degenerate width of Wald at rates near 0 or 1."""
    if n == 0:
        return (np.nan, np.nan)
    lo = beta.ppf(0.025, k + 0.5, n - k + 0.5) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 0.5, n - k + 0.5) if k < n else 1.0
    return float(lo), float(hi)


def main():
    # the UI exports as spotcheck_answers.csv; accept either name
    for cand in ("spotcheck_answers.csv", "answers.csv"):
        ans_path = os.path.join(SPOT, cand)
        if os.path.exists(ans_path):
            break
    else:
        print(f"missing {SPOT}/spotcheck_answers.csv\n"
              f"Open spotcheck/spotcheck.html, complete it, and download the CSV there.")
        return
    key = pd.read_csv(os.path.join(SPOT, "key.csv"))
    ans = pd.read_csv(ans_path)
    d = key.merge(ans, on="item_id", how="left")
    unjudged = int(d["satisfies"].isna().sum())
    if unjudged:
        print(f"warning: {unjudged} items not judged; they are excluded")
    d = d.dropna(subset=["satisfies"])
    d["satisfies"] = d["satisfies"].astype(int)

    rows = []
    ctrl = d[d.kind == "control"]
    fail = d[d.kind == "failure"]
    c_ok = int((ctrl.satisfies == 2).sum())
    print(f"controls (gold poem shown): {c_ok}/{len(ctrl)} scored 2 "
          f"({100*c_ok/max(len(ctrl),1):.0f}%)")
    print(f"  control mean score {ctrl.satisfies.mean():.2f} vs failure mean "
          f"{fail.satisfies.mean():.2f}")

    for label, mask in (("strict (score 2)", fail.satisfies == 2),
                        ("lenient (score 2 or 1)", fail.satisfies >= 1)):
        k, n = int(mask.sum()), len(fail)
        lo, hi = wilson(k, n)
        print(f"failures judged acceptable, {label}: {k}/{n} = {k/n:.3f} [{lo:.3f},{hi:.3f}]")
        rows.append({"criterion": label, "k": k, "n": n, "rate": round(k / n, 3),
                     "lo": round(lo, 3), "hi": round(hi, 3)})

    # per-corpus rates: the acceptance rate need not be the same in both registers
    print("\nby corpus (strict / lenient):")
    for corpus, g in fail.groupby("corpus"):
        s, l = int((g.satisfies == 2).sum()), int((g.satisfies >= 1).sum())
        print(f"  {corpus:11s} n={len(g):3d}  strict {s}/{len(g)}={s/len(g):.2f}  "
              f"lenient {l}/{len(g)}={l/len(g):.2f}")
        rows.append({"criterion": f"strict, {corpus}", "k": s, "n": len(g),
                     "rate": round(s / len(g), 3), **dict(zip(("lo", "hi"),
                                                             [round(v, 3) for v in wilson(s, len(g))]))})

    # corrected Recall@5: measured R@5 plus the share of missed queries judged acceptable.
    # Assumes the sampled acceptance rate applies to all missed queries of that corpus.
    A = pd.read_csv(os.path.join(DERIVED, "eacl_trackA.csv"))
    A = A[(A.system == SYSTEM) & (A["mode"] == "verses") & (A.subset == "all")]
    print(f"\ncorrected Recall@5 for {SYSTEM} (uniform-rate assumption):")
    for _, r in A.iterrows():
        base = float(r["R@5"])
        for row in rows[:2]:                      # pooled strict and lenient
            corrected = base + (1 - base) * row["rate"]
            print(f"  {r['corpus']:11s} {base:.3f} -> {corrected:.3f}  ({row['criterion']})")
            row[f"corrected_R@5_{r['corpus']}"] = round(corrected, 3)

    # Does the §5 interaction survive the correction? Inflate every system's subset score by the
    # share of its misses judged acceptable, s -> s + (1-s)r, and recompute the slope difference.
    # Two settings for r: the pooled rate, and a register-specific one that uses the higher
    # abstract-corpus rate for non-quoting queries (the case least favourable to our claim).
    I = pd.read_csv(os.path.join(DERIVED, "eacl_interaction.csv")).set_index("system")
    r_pool = rows[0]["rate"]
    r_q = float((fail[fail.corpus == "fannorflop"].satisfies == 2).mean())
    r_nq = float((fail[fail.corpus == "diwan"].satisfies == 2).mean())
    print("\ninteraction after correcting for unlabelled correct answers:")
    for label, rq, rnq in (("uncorrected", 0.0, 0.0),
                           ("pooled rate", r_pool, r_pool),
                           ("register-specific", r_q, r_nq)):
        def slope(s):
            q = I.loc[s, "R@5_quote"]
            n = I.loc[s, "R@5_noquote"]
            return (q + (1 - q) * rq) - (n + (1 - n) * rnq)
        did = slope("BM25") - slope("Qwen3-8B")
        print(f"  {label:18s} BM25 slope {slope('BM25'):+.3f}  "
              f"Qwen3-8B {slope('Qwen3-8B'):+.3f}  DiD {did:+.3f}")
        rows.append({"criterion": f"DiD, {label}", "rate": round(did, 3)})

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DERIVED, "eacl_spotcheck.csv"), index=False)
    print(f"\nwrote {DERIVED}/eacl_spotcheck.csv")


if __name__ == "__main__":
    main()
