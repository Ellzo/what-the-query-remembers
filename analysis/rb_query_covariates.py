"""Per-query covariates beyond containment, for both known-item sets: where a quoted span sits in
the poem, how distinctive it is, query length, and target-poem length and confusability. Proper
nouns are added separately (rb_proper_nouns.py) because they need a tagger.

Per query (same tokenisation and normalisation as query_features.py):
  q_len               tokens
  lcn                 longest verbatim n-gram shared with the target verses
  quoted_frac         share of query tokens covered by any >=3-gram shared with the target
  span_verse          1-based index of the verse holding the longest shared span (NaN if lcn<3)
  span_pos            opening / final / middle (verse 1, last verse, anything else)
  span_df             number of other poems in the corpus containing the longest span
                      (0 = unique to the target)
  has_cue             query uses an explicit "opening/verse" cue word (مطلع / بيت / أبيات)
  gold_nverses        verses in the target poem
  gold_ntok           tokens in the target poem
  gold_confus         target-poem confusability: max cosine between the target poem's mean
                      verse vector and any other poem's (query-free; see below)

Confusability uses bag-of-words TF-IDF cosine over verses, not a neural encoder, so that it is
independent of every dense system being evaluated.

Writes results/derived/rb_query_covariates.csv
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DERIVED, ar_tokens, normalize_ar  # noqa: E402
from unified_run import load_queries, load_units  # noqa: E402

CUE = re.compile(r"مطلع|بيت|ابيات")


def ngrams(t, n):
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


def longest_span(qt, verses_tok):
    """Longest query n-gram found inside a single verse, or across the joined poem."""
    joined = [w for v in verses_tok for w in v]
    best, best_span = 0, None
    for n in range(min(12, len(qt)), 0, -1):
        dn = ngrams(joined, n)
        hits = [tuple(qt[i:i + n]) for i in range(len(qt) - n + 1) if tuple(qt[i:i + n]) in dn]
        if hits:
            best, best_span = n, hits[0]
            break
    return best, best_span


def span_verse(span, verses_tok):
    for i, v in enumerate(verses_tok):
        if len(v) >= len(span) and span in ngrams(v, len(span)):
            return i + 1
    # span crosses a verse boundary: locate by its first token run in the joined text
    joined, owner = [], []
    for i, v in enumerate(verses_tok):
        joined += v
        owner += [i + 1] * len(v)
    for i in range(len(joined) - len(span) + 1):
        if tuple(joined[i:i + len(span)]) == span:
            return owner[i]
    return np.nan


def quoted_fraction(qt, joined, n=3):
    dn = ngrams(joined, n)
    cov = np.zeros(len(qt), bool)
    for i in range(len(qt) - n + 1):
        if tuple(qt[i:i + n]) in dn:
            cov[i:i + n] = True
    return float(cov.mean()) if len(qt) else 0.0


def confusability(units):
    from sklearn.feature_extraction.text import TfidfVectorizer
    pids = list(units)
    docs = [" ".join(units[p]) for p in pids]
    X = TfidfVectorizer(tokenizer=str.split, lowercase=False, token_pattern=None,
                        sublinear_tf=True, min_df=2).fit_transform(docs)
    out = {}
    B = 512
    for s in range(0, len(pids), B):
        S = (X[s:s + B] @ X.T).toarray()
        for j in range(S.shape[0]):
            S[j, s + j] = -1
        m = S.max(1)
        for j, v in enumerate(m):
            out[pids[s + j]] = float(v)
    return out


def main():
    rows = []
    for corpus in ("fannorflop", "diwan"):
        units = load_units(corpus, "verses", norm=normalize_ar)
        vt = {p: [v.split() for v in u] for p, u in units.items()}
        joined = {p: [w for v in vs for w in v] for p, vs in vt.items()}
        conf = confusability(units)
        q = load_queries(corpus, "gold")
        for r in q.itertuples():
            qt = ar_tokens(r.prompt)
            vs = vt.get(r.gold_pid)
            if not vs:
                continue
            lcn, span = longest_span(qt, vs)
            sv = span_verse(span, vs) if lcn >= 3 else np.nan
            nv = len(vs)
            pos = (np.nan if np.isnan(sv) else
                   "opening" if sv == 1 else "final" if sv == nv else "middle")
            sdf = np.nan
            if lcn >= 3:
                sdf = sum(1 for p, j in joined.items()
                          if p != r.gold_pid and span in ngrams(j, len(span)))
            rows.append({
                "corpus": corpus, "qid": r.qid, "q_len": len(qt), "lcn": lcn,
                "quoted_frac": quoted_fraction(qt, joined[r.gold_pid]),
                "span_verse": sv, "span_rel": (sv - 1) / max(nv - 1, 1) if lcn >= 3 else np.nan,
                "span_pos": pos, "span_df": sdf,
                "has_cue": bool(CUE.search(normalize_ar(r.prompt))),
                "gold_nverses": nv, "gold_ntok": len(joined[r.gold_pid]),
                "gold_confus": conf.get(r.gold_pid, np.nan),
            })
        print(corpus, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DERIVED, "rb_query_covariates.csv"), index=False)
    print(df.groupby("corpus")[["q_len", "lcn", "quoted_frac", "gold_nverses", "gold_confus"]]
          .mean().round(3))


if __name__ == "__main__":
    main()
