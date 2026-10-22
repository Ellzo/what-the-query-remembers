"""The lexical-bridge mechanism (paper section 5.3), on the unified pipeline.

Three analyses:
  (a) Within query: per-query change in query-document lexical containment (explanation minus
      verses) against the per-query change in retrieval success, Spearman, per corpus and model.
      A query benefits from augmentation to the degree the explanation raises its overlap with
      the target.
  (b) Dose-response: R@5 by quartile of query-verse containment.
  (c) Embedding control: the same within-query analysis with a semantic overlap measure (cosine
      between the query embedding and the gold document embedding) instead of lexical
      containment, to test whether the bridge is a surface-form effect for every system.

Writes results/derived/eacl_mechanism.csv and eacl_dose_response.csv
"""
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DERIVED, ROOT  # noqa: E402
from eacl_metrics import NICE  # noqa: E402

RUNS = os.path.join(ROOT, "analysis", "runs")
MODELS = ["gte", "jina", "silma", "labse", "muffakir", "qwen3", "qwen3_8b", "bgem3", "e5", "bm25"]


def load(system, corpus, mode):
    fn = os.path.join(RUNS, f"{system}__{corpus}__{mode}__gold.json.gz")
    if not os.path.exists(fn):
        return None
    with gzip.open(fn, "rt", encoding="utf-8") as f:
        return {r["qid"]: r for r in json.load(f)}


def embedding_overlap(corpus):
    """Cosine(query, gold document) per query, per mode, using one encoder (gte)."""
    cache = os.path.join(DERIVED, f"embcos_{corpus}.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    import torch
    from sentence_transformers import SentenceTransformer
    from unified_run import load_queries, load_units
    from common import normalize_ar_light
    m = SentenceTransformer("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True,
                            device="cuda", model_kwargs={"torch_dtype": torch.float16})
    q = load_queries(corpus, "gold")
    modes = ["verses", "expl_ai"] + (["expl_ff"] if corpus == "fannorflop" else [])
    out = {"qid": list(q["qid"])}
    qe = m.encode([normalize_ar_light(p) for p in q["prompt"]], batch_size=64,
                  normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    for mode in modes:
        units = load_units(corpus, mode)
        docs = [" \n ".join(units.get(g, [""])) for g in q["gold_pid"]]
        de = m.encode(docs, batch_size=32, normalize_embeddings=True, convert_to_numpy=True,
                      show_progress_bar=False)
        out[f"cos_{mode}"] = (qe * de).sum(1)
    del m
    torch.cuda.empty_cache()
    df = pd.DataFrame(out)
    df.to_csv(cache, index=False)
    return df


def main():
    qf = pd.read_csv(os.path.join(DERIVED, "query_features.csv"))
    rows, dose = [], []

    for corpus in ("diwan", "fannorflop"):
        f = qf[qf.corpus == corpus].set_index("qid")
        emb = embedding_overlap(corpus).set_index("qid")
        expl_modes = ["expl_ai"] + (["expl_ff"] if corpus == "fannorflop" else [])

        for mode in expl_modes:
            d_lex = (f[f"C_{mode}"] - f["C_verses"]).rename("d_lex")
            d_emb = (emb[f"cos_{mode}"] - emb["cos_verses"]).rename("d_emb")
            for system in MODELS:
                v, e = load(system, corpus, "verses"), load(system, corpus, mode)
                if v is None or e is None:
                    continue
                qids = [q for q in v if q in e and q in d_lex.index]
                succ_v = np.array([v[q]["rank_of_gold"] is not None
                                   and v[q]["rank_of_gold"] <= 5 for q in qids], float)
                succ_e = np.array([e[q]["rank_of_gold"] is not None
                                   and e[q]["rank_of_gold"] <= 5 for q in qids], float)
                d_ret = succ_e - succ_v
                if d_ret.std() == 0:
                    continue
                rl, pl = spearmanr(d_lex.loc[qids].values, d_ret)
                re_, pe = spearmanr(d_emb.loc[qids].values, d_ret)
                rows.append({"corpus": corpus, "mode": mode, "system": NICE[system],
                             "n": len(qids),
                             "rho_lexical": round(float(rl), 3), "p_lexical": pl,
                             "rho_embedding": round(float(re_), 3), "p_embedding": pe,
                             "delta_R@5": round(float(d_ret.mean()), 3)})

        # dose-response on verse mode by query-verse containment quartile
        qcut = pd.qcut(f["C_verses"], 4, labels=["Q1 (most figurative)", "Q2", "Q3",
                                                 "Q4 (most literal)"], duplicates="drop")
        for system in MODELS:
            v = load(system, corpus, "verses")
            if v is None:
                continue
            for mode in ["verses"] + expl_modes:
                r = load(system, corpus, mode)
                if r is None:
                    continue
                s = pd.Series({q: (r[q]["rank_of_gold"] is not None
                                   and r[q]["rank_of_gold"] <= 5) for q in r})
                idx = s.index.intersection(qcut.index)
                g = pd.DataFrame({"q": qcut.loc[idx], "s": s.loc[idx].astype(float)})
                for qq, gg in g.groupby("q", observed=True):
                    dose.append({"corpus": corpus, "system": NICE[system], "mode": mode,
                                 "quartile": str(qq), "R@5": round(float(gg["s"].mean()), 3),
                                 "n": len(gg)})

    M = pd.DataFrame(rows)
    D = pd.DataFrame(dose)
    M.to_csv(os.path.join(DERIVED, "eacl_mechanism.csv"), index=False)
    D.to_csv(os.path.join(DERIVED, "eacl_dose_response.csv"), index=False)

    print("=== WITHIN-QUERY: does Δoverlap predict Δretrieval? (Spearman) ===")
    for (c, mo), g in M.groupby(["corpus", "mode"]):
        print(f"\n-- {c} / {mo} (n={g.n.iloc[0]} queries, {len(g)} systems) --")
        print(f"  lexical containment : mean rho = {g.rho_lexical.mean():+.3f}  "
              f"(range {g.rho_lexical.min():+.3f}..{g.rho_lexical.max():+.3f}), "
              f"{(g.p_lexical < .05).sum()}/{len(g)} significant")
        print(f"  embedding cosine    : mean rho = {g.rho_embedding.mean():+.3f}  "
              f"(range {g.rho_embedding.min():+.3f}..{g.rho_embedding.max():+.3f}), "
              f"{(g.p_embedding < .05).sum()}/{len(g)} significant")
        better = (g.rho_lexical.abs() > g.rho_embedding.abs()).sum()
        print(f"  lexical is the stronger predictor for {better}/{len(g)} systems")

    print("\n=== DOSE-RESPONSE: verse-mode R@5 by query-verse overlap quartile ===")
    dv = D[D["mode"] == "verses"].pivot_table(index="quartile", columns="corpus",
                                              values="R@5", aggfunc="mean")
    print(dv.round(3).to_string())
    print("\n=== Diwan: AI-explanation mode by the same quartiles ===")
    da = D[(D["mode"] == "expl_ai") & (D.corpus == "diwan")].pivot_table(
        index="quartile", values="R@5", aggfunc="mean")
    print(da.round(3).to_string())


if __name__ == "__main__":
    main()
