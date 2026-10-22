"""Third Track B judge, from a model family other than Google's (Qwen3-32B, 4-bit, local), under
the original rubric and under a neutral rubric.

The neutral rubric tests whether the literal-match aspect (`direct_textual_relevance`, where 1 is
best) is what puts BM25 near the bottom.

  --rubric original   identical to judge2 (same rubric lines, same brief response format).
  --rubric neutral    the same prompt with criterion 5 (direct textual relevance) and its
                      "we want to measure whether the system understands implicit meaning" framing
                      removed, and the aspect dropped from the response format. Nothing else changes.

Same 100-query subsample as judge2 (seed 0), all ten systems, both corpora, top-3, first 6 verses of
each poem as in the primary judge. Greedy decoding. Qwen3 "thinking" is disabled.

Usage:
  python3 analysis/judge3.py --rubric original
  python3 analysis/judge3.py --rubric neutral
  python3 analysis/judge3.py --rubric neutral --smoke
Writes results/derived/trackB_judge3_{rubric}.csv (resumable).
"""
import argparse
import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DERIVED, METRICS  # noqa: E402
from judge2 import BRIEF_FORMAT, build_prompt  # noqa: E402
from rejudge import PROMPT, SYSTEMS, TOPK, load_run, parse_json, poem_texts  # noqa: E402

MODEL = "unsloth/Qwen3-32B-bnb-4bit"
NEUTRAL_METRICS = [m for m in METRICS if m != "direct_textual_relevance"]


def neutral_prompt(query, results_block):
    head = PROMPT.split("**تعليمات الإجابة:**")[0]
    lines = head.split("\n")
    # drop criterion 5 (the literal-matching aspect with its inverted scale and framing)
    lines = [ln for ln in lines if not ln.startswith("5. **المباشرة النصية")]
    fmt = "\n".join(ln for ln in BRIEF_FORMAT.split("\n")
                    if "direct_textual_relevance" not in ln)
    p = "\n".join(lines) + fmt.lstrip("\n")
    return p.format(k=TOPK, query=query, results=results_block)


def repair(out):
    """Make a CSV written by an earlier version rectangular (it appended a `raw` column only in
    batches that contained a parse failure) and drop unparsed rows so they are re-judged."""
    import csv
    rows = list(csv.reader(open(out, encoding="utf-8")))
    h = rows[0] + ([] if "raw" in rows[0] else ["raw"])
    fixed = [r + [""] * (len(h) - len(r)) for r in rows[1:]]
    df = pd.DataFrame(fixed, columns=h)
    bad = df.parsed.astype(str) != "True"
    df[~bad].to_csv(out, index=False)
    return int(bad.sum())


def build_jobs(out, rubric, nqueries=100, seed=0):
    done = set()
    if os.path.exists(out):
        print(f"repair: dropped {repair(out)} unparsed rows for re-judging")
        prev = pd.read_csv(out)
        done = {(r.system, r.corpus, r.qid) for r in prev.itertuples()}
        print(f"resuming: {len(done)} rows already scored")
    jobs = []
    for corpus in ("diwan", "fannorflop"):
        texts = poem_texts(corpus)
        qids = [r["qid"] for r in load_run(SYSTEMS[0], corpus)]
        keep = set(pd.Series(qids).sample(nqueries, random_state=seed))  # = judge2's subsample
        for system in SYSTEMS:
            for row in load_run(system, corpus):
                if row["qid"] not in keep or (system, corpus, row["qid"]) in done:
                    continue
                blocks = [f"{i}. الأبيات:\n{texts.get(str(pid), '')}"
                          for i, pid in enumerate(row["top_pids"][:TOPK], 1)]
                rb = "\n\n".join(blocks)
                p = (build_prompt(row["prompt"], rb, brief=True) if rubric == "original"
                     else neutral_prompt(row["prompt"], rb))
                jobs.append({"system": system, "corpus": corpus, "qid": row["qid"],
                             "query": row["prompt"], "prompt": p})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rubric", choices=["original", "neutral"], required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = os.path.join(DERIVED, f"trackB_judge3_{args.rubric}.csv")
    if args.smoke:
        out = out.replace(".csv", "_smoke.csv")
    metrics = METRICS if args.rubric == "original" else NEUTRAL_METRICS
    jobs = build_jobs(out, args.rubric)
    print(f"{len(jobs)} cells to judge with {args.model} ({args.rubric} rubric)")
    if args.smoke:
        print(jobs[0]["prompt"])
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

    t0, written = time.time(), 0
    for i in range(0, len(jobs), args.batch):
        chunk = jobs[i:i + args.batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": j["prompt"]}],
                                         tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False) for j in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=4096, add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)[:, enc["input_ids"].shape[1]:]
        rows = []
        for j, g in zip(chunk, gen):
            raw = tok.decode(g, skip_special_tokens=True)
            d = parse_json(raw) or {}
            rec = {"system": j["system"], "corpus": j["corpus"], "qid": j["qid"],
                   "query": j["query"]}
            rec.update({m: d.get(m) for m in metrics})
            rec["justification"] = d.get("overall_quality_justification")
            rec["parsed"] = bool(d) and all(isinstance(d.get(m), (int, float)) for m in metrics)
            rec["raw"] = "" if rec["parsed"] else raw[:500]
            rows.append(rec)
        # fixed column set on every append, so the CSV stays rectangular
        pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        written += len(rows)
        el = time.time() - t0
        print(f"  {written}/{len(jobs)} | eta {(len(jobs)-written)*el/written/60:.0f} min",
              flush=True)
        if args.smoke and written >= 2 * args.batch:
            break
    df = pd.read_csv(out)
    print(f"\n{len(df)} rows, parse failures: {int((~df.parsed.astype(bool)).sum())}")


if __name__ == "__main__":
    main()
