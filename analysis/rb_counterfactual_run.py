"""Retrieve every edited query variant with all ten systems (Fann or Flop, verse mode).

Uses unified_run's corpus loading, chunking, encoders and BM25 unchanged; only the query text
differs. Each system's corpus index is built once and all variants are scored against it.

Reproduction gate: the `orig` variant must give the same rank_of_gold as the released run
analysis/runs/{system}__fannorflop__verses__gold.json.gz for every query. A mismatch means the
model or the environment differs from the paper's run, and the edited-query results would not be
comparable. The script reports the agreement and refuses to write if it is below 100% unless
--allow-drift is given.

Usage:
  python3 analysis/rb_counterfactual_run.py --systems all [--queries data/rebuttal/<file>.csv]
Writes analysis/runs_rebuttal/{system}__fannorflop__verses__{variant}.json.gz
"""
import argparse
import gzip
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, ROOT, ar_tokens, normalize_ar, normalize_ar_light  # noqa: E402
from unified_run import (BM25, DENSE, TOPK_DUMP, build_chunks, encode, load_units,  # noqa: E402
                         poem_scores_from_chunks)

OUT = os.path.join(ROOT, "analysis", "runs_rebuttal")
RUNS = os.path.join(ROOT, "analysis", "runs")
CORPUS = "fannorflop"


def released_ranks(system):
    with gzip.open(os.path.join(RUNS, f"{system}__{CORPUS}__verses__gold.json.gz"), "rt") as f:
        return {r["qid"]: r["rank_of_gold"] for r in json.load(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="all")
    ap.add_argument("--queries", nargs="+",
                    default=[os.path.join(DATA, "rebuttal", "fof_counterfactual_queries.csv")])
    ap.add_argument("--allow-drift", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    Q = pd.concat([pd.read_csv(p) for p in args.queries]).drop_duplicates(["qid", "variant"])
    Q = Q[Q.prompt.notna()]  # paraphrases that failed the acceptance check are left out
    systems = list(DENSE) + ["bm25"] if args.systems == "all" else args.systems.split(",")

    for system in systems:
        t0 = time.time()
        norm = normalize_ar if system == "bm25" else normalize_ar_light
        units = load_units(CORPUS, "verses", norm=norm)
        texts, pids = build_chunks(units)
        uniq = sorted(set(pids))
        p2i = {p: i for i, p in enumerate(uniq)}
        c2p = np.array([p2i[p] for p in pids])
        model = None
        if system == "bm25":
            bm = BM25([ar_tokens(t) for t in texts])
        else:
            import torch
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(DENSE[system], trust_remote_code=True, device="cuda",
                                        model_kwargs={"torch_dtype": torch.float16})
            model.max_seq_length = min(getattr(model, "max_seq_length", 512) or 512, 512)
            doc = encode(model, system, texts, False)

        results = {}
        for variant, g in Q.groupby("variant"):
            g = g[g.gold_pid.astype(str).isin(p2i)].reset_index(drop=True)
            if system == "bm25":
                M = np.stack([poem_scores_from_chunks(bm.scores(ar_tokens(p)), c2p, len(uniq))
                              for p in g.prompt])
            else:
                qe = encode(model, system, [norm(p) for p in g.prompt], True)
                S = doc @ qe.T
                M = np.stack([poem_scores_from_chunks(S[:, i], c2p, len(uniq))
                              for i in range(S.shape[1])])
            rows = []
            for qi, r in enumerate(g.itertuples()):
                order = np.argsort(-M[qi])
                rank = int(np.where(order == p2i[str(r.gold_pid)])[0][0]) + 1
                rows.append({"qid": r.qid, "variant": variant, "prompt": r.prompt,
                             "gold_pid": str(r.gold_pid), "rank_of_gold": rank,
                             "top_pids": [uniq[i] for i in order[:TOPK_DUMP]]})
            results[variant] = rows

        rel = released_ranks(system)
        orig = {r["qid"]: r["rank_of_gold"] for r in results.get("orig", [])}
        agree = np.mean([orig[q] == rel[q] for q in orig]) if orig else float("nan")
        print(f"[{system}] reproduction of released ranks on orig: {agree:.4f} "
              f"(n={len(orig)})", flush=True)
        if orig and agree < 1.0 and not args.allow_drift:
            print(f"[{system}] REFUSING to write: drift from the released run", flush=True)
        else:
            for variant, rows in results.items():
                fn = os.path.join(OUT, f"{system}__{CORPUS}__verses__{variant}.json.gz")
                with gzip.open(fn, "wt", encoding="utf-8") as f:
                    json.dump(rows, f, ensure_ascii=False)
                r = np.array([x["rank_of_gold"] for x in rows])
                print(f"   {variant:12s} n={len(r):3d} R@5={np.mean(r <= 5):.3f}", flush=True)
        if model is not None:
            del model
            import torch
            torch.cuda.empty_cache()
        print(f"[{system}] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
