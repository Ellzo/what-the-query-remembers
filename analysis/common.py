"""Shared IO and parsing helpers: path constants, Arabic normalisation and tokenisation.

All numbers in the paper are derived from results/ and data/ via these helpers.
"""
import ast
import csv
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")
DERIVED = os.path.join(RESULTS, "derived")
FIGS = os.path.join(ROOT, "figures_new")
os.makedirs(DERIVED, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

METRICS = [
    "semantic_relevance",
    "thematic_coherence",
    "metaphor_understanding",
    "semantic_coverage",
    "direct_textual_relevance",
    "precision",
    "overall_quality",
]

# Diacritics + tatweel for Arabic normalisation
_AR_DIAC = re.compile(r"[ً-ٰٟـ]")
_NON_AR = re.compile(r"[^؀-ۿ\s]")


def normalize_ar(text: str) -> str:
    """Aggressive normalisation: diacritics/tatweel + unify alef/ya/ta-marbuta + drop non-Arabic.

    Retained for BM25 and for the lexical-overlap measures, where collapsing orthographic
    variants is the standard choice. NOT used for dense encoders: unifying ta-marbuta and
    alef-maqsura produces strings outside the models' subword vocabularies and costs ~9 R@5
    points on jina-v3 (measured; see norm_ablation.py).
    """
    if not isinstance(text, str):
        return ""
    t = _AR_DIAC.sub("", text)
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    t = t.replace("ى", "ي").replace("ة", "ه")
    t = _NON_AR.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def normalize_ar_light(text: str) -> str:
    """Minimal normalisation: strip diacritics and tatweel only, preserving letter identity.

    This is the preprocessing applied to text given to dense encoders. Diacritic removal is
    essential (keeping them costs ~35 R@5 points); letter unification is not.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", _AR_DIAC.sub("", text)).strip()


def ar_tokens(text: str) -> list:
    return normalize_ar(text).split()


def _parse_model_chunk(fn: str):
    b = os.path.basename(fn)
    m = re.match(r"(accuracy_results|gemini_results)__embed_(.+?)__chunk_(.+)\.csv", b)
    if not m:
        m = re.match(r"(accuracy_results|gemini_results)__(.+?)__chunk_(.+)\.csv", b)
    if not m:
        return None
    return {"kind": m.group(1), "model": m.group(2), "chunk": m.group(3), "path": fn}


def list_accuracy_files(corpus: str):
    out = []
    for fn in sorted(glob.glob(os.path.join(RESULTS, corpus, "accuracy_results__*.csv"))):
        info = _parse_model_chunk(fn)
        if info:
            out.append(info)
    return out


def list_gemini_files(corpus: str):
    out = []
    for fn in sorted(glob.glob(os.path.join(RESULTS, corpus, "gemini_results__*.csv"))):
        info = _parse_model_chunk(fn)
        if info:
            out.append(info)
    return out


def load_accuracy(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ("top1", "topk"):
        df[c] = df[c].astype(str).str.strip().str.lower().map(
            {"true": 1, "false": 0, "1": 1, "0": 0}
        )
    df = df.dropna(subset=["top1", "topk"])
    df["top1"] = df["top1"].astype(int)
    df["topk"] = df["topk"].astype(int)
    return df


def load_gemini(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return clean_metric_cols(df)


def clean_metric_cols(df: pd.DataFrame, metrics=METRICS) -> pd.DataFrame:
    df = df.copy()
    for c in metrics:
        if c in df.columns:
            s = df[c].astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0]
            df[c] = pd.to_numeric(s, errors="coerce")
    return df


def parse_retrieved(cell: str):
    """Parse the retrieved_results list-of-dicts; return list of poem_ids in rank order."""
    if not isinstance(cell, str) or not cell.strip():
        return []
    try:
        items = ast.literal_eval(cell)
    except Exception:
        return []
    ids = []
    for it in items:
        pid = None
        meta = it.get("metadata") if isinstance(it, dict) else None
        if isinstance(meta, dict):
            pid = meta.get("poem_id")
        if pid is None and isinstance(it, dict):
            pid = it.get("poem_id")
        ids.append(pid)
    return ids


# prompt and gold-id loaders
def load_prompts():
    """Return dict of the three query sets."""
    out = {}
    out["accuracy"] = pd.read_csv(os.path.join(DATA, "accuracy_prompts.csv"))  # prompt, poem_id
    out["diwan_gold"] = pd.read_csv(os.path.join(DATA, "diwan_prompts_final.csv"))  # poem_id, prompt
    out["thematic"] = pd.read_csv(os.path.join(DATA, "gemini_prompts.csv"))  # id, prompt
    return out


if __name__ == "__main__":
    for corp in ("diwan", "fannorflop"):
        print(corp, len(list_accuracy_files(corp)), "acc files;",
              len(list_gemini_files(corp)), "gemini files")
    p = load_prompts()
    for k, v in p.items():
        print(k, v.shape, list(v.columns))
