"""Normalisation ablation across several dense encoders.

Repeats three normalisation settings on several encoders, so that the cost of unifying
alef / ya / ta-marbuta is measured per model rather than generalised from one.

Settings (Diwan, verse mode, the same chunking and max-pooling as the main pipeline):
  raw     no normalisation at all (diacritics retained)
  light   diacritics + tatweel only            <- used for dense encoders
  full    light + unify alef/ya/ta-marbuta + drop non-Arabic   <- used for BM25

Writes results/derived/eacl_norm_ablation.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DERIVED, normalize_ar, normalize_ar_light  # noqa: E402
from unified_run import (DENSE, build_chunks, encode, load_queries,  # noqa: E402
                         load_units, poem_scores_from_chunks)

SETTINGS = {"raw": lambda s: s if isinstance(s, str) else "",
            "light": normalize_ar_light,
            "full": normalize_ar}


def run(system, corpus, mode, model, norm):
    units = load_units(corpus, mode, norm=norm)
    texts, pids = build_chunks(units)
    uniq = sorted(set(pids))
    pid_to_idx = {p: i for i, p in enumerate(uniq)}
    chunk_to_poem = np.array([pid_to_idx[p] for p in pids])
    doc_emb = encode(model, system, texts, False)
    q = load_queries(corpus, "gold")
    q = q[q["gold_pid"].isin(pid_to_idx)].reset_index(drop=True)
    q_emb = encode(model, system, [norm(p) for p in q["prompt"]], True)
    sims = doc_emb @ q_emb.T
    ranks = []
    for i, r in enumerate(q.itertuples()):
        sc = poem_scores_from_chunks(sims[:, i], chunk_to_poem, len(uniq))
        order = np.argsort(-sc)
        ranks.append(int(np.where(order == pid_to_idx[r.gold_pid])[0][0]) + 1)
    r = np.array(ranks)
    return {"R@1": float(np.mean(r == 1)), "R@5": float(np.mean(r <= 5)), "n": len(r)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="jina,gte,bgem3")
    ap.add_argument("--corpus", default="diwan")
    ap.add_argument("--mode", default="verses")
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    rows = []
    for system in args.systems.split(","):
        print(f"[load] {DENSE[system]}", flush=True)
        model = SentenceTransformer(DENSE[system], trust_remote_code=True, device="cuda",
                                    model_kwargs={"torch_dtype": torch.float16})
        model.max_seq_length = min(getattr(model, "max_seq_length", 512) or 512, 512)
        for name, norm in SETTINGS.items():
            m = run(system, args.corpus, args.mode, model, norm)
            rows.append({"system": system, "normalisation": name, **m})
            print(f"  {system:8s} {name:6s} R@1={m['R@1']:.3f} R@5={m['R@5']:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    p = df.pivot_table(index="system", columns="normalisation", values="R@5")
    p["light_minus_full"] = p["light"] - p["full"]
    p["light_minus_raw"] = p["light"] - p["raw"]
    print("\nR@5 by normalisation:\n" + p.round(3).to_string())
    df.to_csv(os.path.join(DERIVED, "eacl_norm_ablation.csv"), index=False)
    print(f"\nwrote {DERIVED}/eacl_norm_ablation.csv")


if __name__ == "__main__":
    main()
