"""X1 - Unified retrieval pipeline for EACL 2027.

ONE pipeline for every system (dense + BM25), every corpus, every indexing mode, and both
query sets. Dumps per-query ranked lists so all downstream analysis (metrics, CIs, McNemar,
stratification, fusion, hybrid, annotation sampling) is derived from the same artifact.

Fixes carried over from the audit:
  * chunk_versic dropped tail verses at some poem lengths -> now covers the tail.
  * accuracy_prompts.csv has one duplicated prompt string (poem_6232 / poem_5171) -> the
    duplicate is dropped at load time (n 245 -> 244) and the choice is logged.
  * mode='explanations' meant HUMAN on FannOrFlop but AI on Diwan -> canonical mode vocabulary
    {verses, expl_ai, expl_ff} is used everywhere.
  * BM25 runs inside this harness (same corpus, same chunking, same max-pool), so lexical and
    dense numbers are finally like-for-like.
  * Normalisation is per-system and deliberate, not incidental: dense encoders get diacritic
    and tatweel stripping only (normalize_ar_light), because the additional letter unification
    (alef/ya/ta-marbuta) produces strings outside their subword vocabularies and costs ~9 R@5
    points; BM25 gets the full normalisation, where collapsing orthographic variants is the
    standard and helpful choice. Both are reported as an ablation.

Usage:
  python3 analysis/unified_run.py --systems all --corpora all
  python3 analysis/unified_run.py --systems bm25 --corpora diwan
"""
import argparse
import ast
import gzip
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, ROOT, ar_tokens, normalize_ar, normalize_ar_light  # noqa: E402

RUNS = os.path.join(ROOT, "analysis", "runs")
os.makedirs(RUNS, exist_ok=True)

DENSE = {
    "gte": "Alibaba-NLP/gte-multilingual-base",
    "jina": "jinaai/jina-embeddings-v3",
    "silma": "silma-ai/silma-embedding-sts-v0.1",
    "labse": "sentence-transformers/LaBSE",
    "muffakir": "mohamed2811/Muffakir_Embedding_V2",
    "qwen3": "Qwen/Qwen3-Embedding-0.6B",
    # added for EACL 2027: current strong multilingual retrievers. qwen3_8b is a within-family
    # capacity ablation against qwen3 (0.6B) - it tests whether scale escapes the lexical bridge.
    "qwen3_8b": "Qwen/Qwen3-Embedding-8B",
    "bgem3": "BAAI/bge-m3",
    "e5": "intfloat/multilingual-e5-large-instruct",
}
E5_TASK = "Given a search query describing a poem, retrieve the poem that matches it"
TOPK_DUMP = 100


# --------------------------------------------------------------------------- corpora
def load_units(corpus, mode, norm=None):
    """poem_id -> list of indexable units (one per verse) under the given mode.

    `norm` is the text normaliser: normalize_ar_light for dense encoders (default),
    normalize_ar for BM25 and the lexical-overlap measures.
    """
    norm = norm or normalize_ar_light
    if corpus == "diwan":
        df = pd.read_csv(f"{DATA}/diwan_with_explanations.csv")
        df["pid"] = df["poem_id"].astype(str)
        verse_col = "verses_list"
    else:
        df = pd.read_csv(f"{DATA}/flipp_flop_ai.csv")
        df["pid"] = df["id"].astype(str)
        verse_col = "poem_verses"

    def verses_of(row):
        if corpus == "diwan":
            try:
                return [norm(v) for v in ast.literal_eval(row[verse_col])]
            except Exception:
                return []
        return [norm(v) for v in str(row[verse_col]).split("\n") if v.strip()]

    def dictlist(row, col):
        try:
            items = ast.literal_eval(row[col])
        except Exception:
            return []
        return [norm(d.get("explanation", "")) for d in items if isinstance(d, dict)]

    units = {}
    for _, row in df.iterrows():
        if mode == "verses":
            u = verses_of(row)
        elif mode == "expl_ai":
            u = dictlist(row, "verses_ai_explanations")
        elif mode == "expl_ff":
            if corpus != "fannorflop":
                return {}
            u = dictlist(row, "explanation")
        else:
            raise ValueError(mode)
        u = [x for x in u if x]
        if u:
            units[row["pid"]] = u
    return units


def load_queries(corpus, qset):
    """Return DataFrame with columns qid, prompt, gold_pid (gold_pid None for thematic)."""
    if qset == "gold":
        if corpus == "diwan":
            g = pd.read_csv(f"{DATA}/diwan_prompts_final.csv")
        else:
            g = pd.read_csv(f"{DATA}/accuracy_prompts.csv")
            n0 = len(g)
            g = g.drop_duplicates(subset="prompt", keep="first")
            if len(g) != n0:
                print(f"  [dedup] accuracy_prompts {n0} -> {len(g)} (duplicate prompt string; "
                      f"kept first occurrence)", flush=True)
        g = g.rename(columns={"poem_id": "gold_pid"})
        g["gold_pid"] = g["gold_pid"].astype(str)
        g["qid"] = [f"g{i}" for i in range(len(g))]
        return g[["qid", "prompt", "gold_pid"]].reset_index(drop=True)
    t = pd.read_csv(f"{DATA}/gemini_prompts.csv")
    t["qid"] = "t" + t["id"].astype(str)
    t["gold_pid"] = None
    return t[["qid", "prompt", "gold_pid"]].reset_index(drop=True)


# --------------------------------------------------------------------------- chunking
def chunk_versic(units, n=3, o=1):
    """Overlapping windows of n verses, step n-o. Covers the tail (the old version did not)."""
    units = [u for u in units if u]
    if not units:
        return []
    if len(units) <= n:
        return [" \n ".join(units)]
    step = n - o
    starts = list(range(0, len(units) - n + 1, step))
    if starts[-1] + n < len(units):
        starts.append(len(units) - n)
    return [" \n ".join(units[i:i + n]) for i in starts]


def build_chunks(units_map):
    texts, pids = [], []
    for pid, units in units_map.items():
        for c in chunk_versic(units):
            if c.strip():
                texts.append(c)
                pids.append(pid)
    return texts, pids


# --------------------------------------------------------------------------- retrievers
class BM25:
    """Okapi BM25, self-contained (no external deps) so the released code has no hidden state."""

    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(corpus_tokens)
        self.dl = np.array([len(d) for d in corpus_tokens], dtype=np.float32)
        self.avgdl = float(self.dl.mean()) if self.N else 1.0
        df = Counter()
        self.post = {}
        for i, d in enumerate(corpus_tokens):
            for w, f in Counter(d).items():
                df[w] += 1
                self.post.setdefault(w, []).append((i, f))
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        self.denom_base = self.k1 * (1 - self.b + self.b * self.dl / self.avgdl)

    def scores(self, qtokens):
        s = np.zeros(self.N, dtype=np.float32)
        for w in qtokens:
            if w not in self.post:
                continue
            idf = self.idf[w]
            for i, f in self.post[w]:
                s[i] += idf * (f * (self.k1 + 1)) / (f + self.denom_base[i])
        return s


def encode(model, key, texts, is_query, bs=64):
    """Each model gets the query/passage convention its own card prescribes."""
    kw = {}
    if key == "jina":
        kw["task"] = "retrieval.query" if is_query else "retrieval.passage"
    elif key in ("qwen3", "qwen3_8b") and is_query:
        kw["prompt_name"] = "query"
    elif key == "e5":
        texts = [f"Instruct: {E5_TASK}\nQuery: {t}" for t in texts] if is_query else list(texts)
    if key in ("qwen3_8b",):
        bs = min(bs, 16)
    return model.encode(texts, batch_size=bs, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True, **kw)


# --------------------------------------------------------------------------- run
def dump(system, corpus, mode, qset, queries, uniq_pids, score_matrix):
    """score_matrix: (n_queries, n_poems). Writes top-K ids/scores + rank_of_gold."""
    rows = []
    pid_to_idx = {p: i for i, p in enumerate(uniq_pids)}
    for qi, r in enumerate(queries.itertuples()):
        sc = score_matrix[qi]
        order = np.argsort(-sc)[:TOPK_DUMP]
        rank = None
        if r.gold_pid is not None and r.gold_pid in pid_to_idx:
            full = np.argsort(-sc)
            rank = int(np.where(full == pid_to_idx[r.gold_pid])[0][0]) + 1
        rows.append({
            "qid": r.qid, "prompt": r.prompt, "gold_pid": r.gold_pid,
            "rank_of_gold": rank,
            "top_pids": [uniq_pids[i] for i in order],
            "top_scores": [round(float(sc[i]), 6) for i in order],
        })
    fn = os.path.join(RUNS, f"{system}__{corpus}__{mode}__{qset}.json.gz")
    with gzip.open(fn, "wt", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    gold_ranks = [r["rank_of_gold"] for r in rows if r["rank_of_gold"]]
    summ = ""
    if gold_ranks:
        gr = np.array(gold_ranks)
        summ = (f"  n={len(gr)} R@1={np.mean(gr == 1):.3f} R@5={np.mean(gr <= 5):.3f} "
                f"MRR={np.mean(1 / gr):.3f}")
    print(f"    wrote {os.path.basename(fn)}{summ}", flush=True)
    return rows


def poem_scores_from_chunks(chunk_scores, chunk_to_poem, n_poems):
    out = np.full(n_poems, -1e9, dtype=np.float32)
    np.maximum.at(out, chunk_to_poem, chunk_scores)
    return out


def run_cell(system, corpus, mode, model=None):
    norm = normalize_ar if system == "bm25" else normalize_ar_light
    units = load_units(corpus, mode, norm=norm)
    if not units:
        return
    texts, pids = build_chunks(units)
    uniq_pids = sorted(set(pids))
    pid_to_idx = {p: i for i, p in enumerate(uniq_pids)}
    chunk_to_poem = np.array([pid_to_idx[p] for p in pids])
    mean_tok = float(np.mean([len(t.split()) for t in texts]))
    print(f"  {system}/{corpus}/{mode}: {len(uniq_pids)} poems, {len(texts)} chunks, "
          f"mean {mean_tok:.1f} tok", flush=True)

    if system == "bm25":
        bm = BM25([ar_tokens(t) for t in texts])
        scorer = lambda q: bm.scores(ar_tokens(q))  # noqa: E731  (ar_tokens = aggressive)
        doc_emb = None
    else:
        doc_emb = encode(model, system, texts, False)
        scorer = None

    for qset in ("gold", "thematic"):
        queries = load_queries(corpus, qset)
        if qset == "gold":
            queries = queries[queries["gold_pid"].isin(pid_to_idx)].reset_index(drop=True)
        if doc_emb is None:
            M = np.stack([poem_scores_from_chunks(scorer(p), chunk_to_poem,
                                                  len(uniq_pids)) for p in queries["prompt"]])
        else:
            q_emb = encode(model, system, [norm(p) for p in queries["prompt"]], True)
            sims = doc_emb @ q_emb.T  # (chunks, queries)
            M = np.stack([poem_scores_from_chunks(sims[:, i], chunk_to_poem, len(uniq_pids))
                          for i in range(sims.shape[1])])
        dump(system, corpus, mode, qset, queries, uniq_pids, M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="all")
    ap.add_argument("--corpora", default="all")
    args = ap.parse_args()

    systems = list(DENSE) + ["bm25"] if args.systems == "all" else args.systems.split(",")
    corpora = ["diwan", "fannorflop"] if args.corpora == "all" else args.corpora.split(",")
    modes = {"diwan": ["verses", "expl_ai"],
             "fannorflop": ["verses", "expl_ai", "expl_ff"]}

    for system in systems:
        t0 = time.time()
        model = None
        if system != "bm25":
            import torch
            from sentence_transformers import SentenceTransformer
            print(f"[load] {DENSE[system]}", flush=True)
            model = SentenceTransformer(DENSE[system], trust_remote_code=True, device="cuda",
                                        model_kwargs={"torch_dtype": torch.float16})
            model.max_seq_length = min(getattr(model, "max_seq_length", 512) or 512, 512)
        for corpus in corpora:
            for mode in modes[corpus]:
                run_cell(system, corpus, mode, model)
        if model is not None:
            del model
            import torch
            torch.cuda.empty_cache()
        print(f"[done] {system} in {time.time() - t0:.0f}s\n", flush=True)


if __name__ == "__main__":
    main()
