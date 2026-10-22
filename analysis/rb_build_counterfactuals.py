"""Edited copies of the same Fann or Flop queries (same gold poem, same information need), so that
remembered wording is manipulated rather than stratified.

Variants (verse mode, Fann or Flop, the 244 queries):
  orig            the query as released
  del_span        [quoting queries, n=80] every token covered by a >=3-gram shared with the target
                  poem is deleted, along with a cue word (مطلعها, ...) directly before it.
                  Removes the wording and the information it carried, so it is a control only.
  inj_open        [non-quoting queries, n=164] the query + " ومن أبياتها: " + the first hemistich
  inj_mid         of the opening / middle / final verse of the target poem, verbatim. Adds
  inj_final       wording without changing the information need. Only the position varies.

Rewrites of the quoted wording are built elsewhere: rb_paraphrase_spans.py (LLM) and
rb_ingest_hand_paraphrase.py (by hand).

Writes data/rebuttal/fof_counterfactual_queries.csv (qid, gold_pid, variant, prompt).
"""
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, ar_tokens, normalize_ar  # noqa: E402
from unified_run import load_queries  # noqa: E402

OUTDIR = os.path.join(DATA, "rebuttal")
CUE = re.compile(r"^(مطلعها|ومطلعها|مطلع|يقول|فيها|ومنها|بيت|البيت)$")
JOIN = " ومن أبياتها: "


def poems():
    df = pd.read_csv(f"{DATA}/flipp_flop_ai.csv")
    return {str(r.id): [v for v in str(r.poem_verses).split("\n") if v.strip()]
            for r in df.itertuples()}


def shared_mask(prompt, poem_tokens, n=3):
    """Per whitespace token of the prompt: True if it lies in a >=n-gram shared with the poem.

    Each surface token is normalised on its own; tokens that normalise to nothing (punctuation)
    are never masked, and tokens that normalise to several are treated as one unit.
    """
    surf = prompt.split()
    norm = [" ".join(ar_tokens(t)) for t in surf]
    idx = [i for i, t in enumerate(norm) if t]
    toks = [norm[i] for i in idx]
    grams = {tuple(poem_tokens[i:i + n]) for i in range(len(poem_tokens) - n + 1)}
    cov = [False] * len(toks)
    for i in range(len(toks) - n + 1):
        if tuple(" ".join(toks[i:i + n]).split()[:n]) in grams or \
                tuple(toks[i:i + n]) in grams:
            for j in range(i, i + n):
                cov[j] = True
    mask = [False] * len(surf)
    for k, i in enumerate(idx):
        mask[i] = cov[k]
    return surf, mask


def delete_span(prompt, poem_tokens):
    surf, mask = shared_mask(prompt, poem_tokens)
    keep = []
    for i, (t, m) in enumerate(zip(surf, mask)):
        if m:
            continue
        nxt_masked = i + 1 < len(surf) and mask[i + 1]
        if nxt_masked and CUE.match(normalize_ar(t)):
            continue
        keep.append(t)
    return " ".join(keep).strip(" ،,:")


def hemistich(verse):
    parts = [p.strip() for p in re.split(r"\s{2,}", verse) if p.strip()]
    return parts[0] if parts else verse.strip()


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    P = poems()
    q = load_queries("fannorflop", "gold")
    rows = []
    for r in q.itertuples():
        verses = P[r.gold_pid]
        ptoks = ar_tokens(" ".join(verses))
        qt = ar_tokens(r.prompt)
        lcn = max((n for n in range(1, len(qt) + 1)
                   if any(tuple(qt[i:i + n]) in {tuple(ptoks[j:j + n])
                                                 for j in range(len(ptoks) - n + 1)}
                          for i in range(len(qt) - n + 1))), default=0)
        base = {"qid": r.qid, "gold_pid": r.gold_pid, "lcn_orig": lcn}
        rows.append({**base, "variant": "orig", "prompt": r.prompt})
        if lcn >= 3:
            rows.append({**base, "variant": "del_span", "prompt": delete_span(r.prompt, ptoks)})
        else:
            nv = len(verses)
            for name, k in (("inj_open", 0), ("inj_mid", nv // 2), ("inj_final", nv - 1)):
                rows.append({**base, "variant": name,
                             "prompt": r.prompt.rstrip(" .،") + JOIN + hemistich(verses[k])})
    df = pd.DataFrame(rows)
    # re-measure the manipulated quantity on every variant
    df["lcn_new"] = [
        max([n for n in range(1, 13) if any(
            tuple(ar_tokens(p)[i:i + n]) in {tuple(ar_tokens(" ".join(P[g]))[j:j + n])
                                             for j in range(len(ar_tokens(" ".join(P[g]))) - n + 1)}
            for i in range(len(ar_tokens(p)) - n + 1))] or [0])
        for p, g in zip(df.prompt, df.gold_pid)]
    df.to_csv(os.path.join(OUTDIR, "fof_counterfactual_queries.csv"), index=False)
    print(df.groupby("variant").agg(n=("qid", "size"), lcn_orig=("lcn_orig", "mean"),
                                    lcn_new=("lcn_new", "mean"),
                                    still_quoting=("lcn_new", lambda s: int((s >= 3).sum()))))


if __name__ == "__main__":
    main()
