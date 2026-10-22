"""Turn the hand rewrites of the quoting queries (human_checks/hand_paraphrase) into a query file.

Checks each rewrite the same way the task page did: after the paper's normalisation, it must share
no 3-token span with the target poem, except tokens belonging to a proper name the writer chose to
keep (those rows are flagged `name_ok` in the answers). Rows flagged `skip` are dropped, and so is
any row that still shares wording without the name flag.

Usage: python3 analysis/rb_ingest_hand_paraphrase.py [human_checks/hand_paraphrase/answers.csv]
Writes data/rebuttal/fof_hand_paraphrase_queries.csv (variant hand_para).
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, ar_tokens  # noqa: E402
from rb_build_counterfactuals import poems  # noqa: E402
from rb_paraphrase_spans import lcn, offending  # noqa: E402

HT = os.path.join(os.path.dirname(DATA), "human_checks", "hand_paraphrase")
OUT = os.path.join(DATA, "rebuttal", "fof_hand_paraphrase_queries.csv")


def main():
    ans = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HT, "answers.csv"))
    ans.columns = [c.strip().lstrip("﻿") for c in ans.columns]
    key = pd.read_csv(os.path.join(HT, "key.csv"))
    cf = pd.read_csv(os.path.join(DATA, "rebuttal", "fof_counterfactual_queries.csv"))
    orig = cf[cf.variant == "orig"].set_index("qid")
    m = ans.merge(key, on="id")
    P = poems()
    rows, dropped = [], []
    for r in m.itertuples():
        pt = ar_tokens(" ".join(P[str(r.gold_pid)]))
        rw = "" if pd.isna(r.rewrite) else str(r.rewrite).strip()
        shared = offending(rw, pt) if rw else []
        # tokens the writer kept as a name: whatever the page reported as still shared
        name_tok = set(ar_tokens(str(r.remaining_shared))) if r.name_ok else set()
        resid = offending(rw, pt, mask=name_tok)
        why = ("skip" if r.skip else "empty" if not rw else
               "still quoting" if resid else None)
        if why:
            dropped.append((r.id, r.qid, why, " | ".join(shared)))
            continue
        rows.append({"qid": r.qid, "gold_pid": r.gold_pid, "variant": "hand_para",
                     "lcn_orig": orig.lcn_orig.get(r.qid), "original": orig.prompt.get(r.qid),
                     "prompt": rw, "name_kept": int(bool(r.name_ok)),
                     "lcn_new": lcn(rw, pt), "lcn_new_names_masked": lcn(rw, pt, mask=name_tok)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"kept {len(out)} of {len(m)}; dropped {len(dropped)}")
    for d in dropped:
        print("  dropped", d)
    print(f"names kept in {out.name_kept.sum()} queries; "
          f"lcn>=3 ignoring kept names: {(out.lcn_new_names_masked >= 3).sum()}")
    print("mean query tokens: original "
          f"{out.original.map(lambda s: len(ar_tokens(s))).mean():.1f} -> rewrite "
          f"{out.prompt.map(lambda s: len(ar_tokens(s))).mean():.1f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
