"""Second Track B judge: an open-weights model run locally, as a judge-independence control.

Why. Every prior review round objected to a single LLM judge, and the objection is sharpened here
by the fact that Gemini 2.0 Flash, which generated our explanations, was withdrawn from the API
during this work: judgments made with a proprietary judge cannot be re-derived once the model is
retired. An open-weights judge with fixed weights does not have that property, so this run is both
a robustness check and a permanently reproducible one.

Design. The rubric text is byte-identical to the primary judge's prompt (`rejudge.py`), so any
disagreement is attributable to the model rather than to the instructions. The only deviation is the
response format: with `--brief` the model returns the seven scores plus one short overall
justification instead of seven justifications, which keeps local decoding tractable. This is
reported in the paper.

Usage:
  python3 analysis/judge2.py --nqueries 100            # 100 queries x 10 systems x 2 corpora
  python3 analysis/judge2.py --nqueries 4 --smoke      # throughput probe

Writes results/derived/trackB_judge2.csv incrementally (resumable).
"""
import argparse
import json
import os
import re
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DERIVED, METRICS  # noqa: E402
from rejudge import PROMPT, SYSTEMS, TOPK, load_run, parse_json, poem_texts  # noqa: E402

OUT = os.path.join(DERIVED, "trackB_judge2.csv")
MODEL = "unsloth/gemma-3-27b-it-bnb-4bit"

# The primary judge's prompt ends with a response-format block asking for seven justifications.
# --brief swaps only that block; every rubric line above it is unchanged.
BRIEF_FORMAT = """
**تعليمات الإجابة:**
- اكتب تقييمك في تنسيق JSON فقط
- لا تضف أي نص إضافي خارج JSON
- استخدم الأرقام فقط للدرجات (1-5)
- اكتب تبريراً واحداً موجزاً جداً (أقل من 20 كلمة) للتقييم العام فقط

**مثال على التنسيق المطلوب:**
```json
{{
  "semantic_relevance": 4,
  "thematic_coherence": 3,
  "metaphor_understanding": 2,
  "semantic_coverage": 4,
  "direct_textual_relevance": 5,
  "precision": 4,
  "overall_quality": 4,
  "overall_quality_justification": "..."
}}
```
"""


def build_prompt(query, results_block, brief):
    p = PROMPT
    if brief:
        head = p.split("**تعليمات الإجابة:**")[0]
        p = head + BRIEF_FORMAT.lstrip("\n")
    return p.format(k=TOPK, query=query, results=results_block)


def build_jobs(nqueries, corpora, systems, brief, seed=0):
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = {(r.system, r.corpus, r.qid) for r in prev.itertuples()}
        print(f"resuming: {len(done)} rows already scored")

    jobs = []
    for corpus in corpora:
        texts = poem_texts(corpus)
        # Same query subsample for every system, so comparisons stay paired.
        qids = [r["qid"] for r in load_run(systems[0], corpus)]
        if nqueries and nqueries < len(qids):
            qids = list(pd.Series(qids).sample(nqueries, random_state=seed).sort_values())
        keep = set(qids)
        for system in systems:
            for row in load_run(system, corpus):
                if row["qid"] not in keep or (system, corpus, row["qid"]) in done:
                    continue
                blocks = [f"{i}. الأبيات:\n{texts.get(str(pid), '')}"
                          for i, pid in enumerate(row["top_pids"][:TOPK], 1)]
                jobs.append({
                    "system": system, "corpus": corpus, "qid": row["qid"],
                    "query": row["prompt"],
                    "prompt": build_prompt(row["prompt"], "\n\n".join(blocks), brief),
                })
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--nqueries", type=int, default=100)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--systems", default="all")
    ap.add_argument("--corpora", default="all")
    ap.add_argument("--brief", action="store_true", default=True)
    ap.add_argument("--full-justifications", dest="brief", action="store_false")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    systems = SYSTEMS if args.systems == "all" else args.systems.split(",")
    corpora = ["diwan", "fannorflop"] if args.corpora == "all" else args.corpora.split(",")
    jobs = build_jobs(args.nqueries, corpora, systems, args.brief)
    print(f"{len(jobs)} cells to judge with {args.model} (brief={args.brief})")
    if not jobs:
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cuda:0", torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.eval()

    t0, n_out, written = time.time(), 0, 0
    for i in range(0, len(jobs), args.batch):
        chunk = jobs[i:i + args.batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": j["prompt"]}],
                                         tokenize=False, add_generation_prompt=True)
                 for j in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=4096, add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        n_out += int((gen != tok.pad_token_id).sum())
        rows = []
        for j, g in zip(chunk, gen):
            d = parse_json(tok.decode(g, skip_special_tokens=True)) or {}
            rec = {"system": j["system"], "corpus": j["corpus"], "qid": j["qid"],
                   "query": j["query"]}
            rec.update({m: d.get(m) for m in METRICS})
            rec["parsed"] = bool(d)
            rows.append(rec)
        pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        written += len(rows)
        el = time.time() - t0
        print(f"  {written}/{len(jobs)} cells | {n_out/el:.0f} out-tok/s | "
              f"eta {(len(jobs)-written)*el/max(written,1)/60:.0f} min", flush=True)
        if args.smoke and written >= args.batch:
            break

    df = pd.read_csv(OUT)
    print(f"\n{len(df)} rows, parse failures: {int((~df.parsed.astype(bool)).sum())}")


if __name__ == "__main__":
    main()
