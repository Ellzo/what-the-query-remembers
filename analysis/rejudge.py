"""The primary LLM judge (Gemini 3.5 Flash) over the unified pipeline's thematic retrievals.

Judging runs on the same rank dumps as Track A, so the two tracks are pipeline-consistent.
Details of the setup:
  1. The judge sees each system's top 3 poems, and the prompt says 3.
  2. Verse mode only, so no model-generated explanation is shown to the judge and self-preference
     toward our Gemini-generated explanations cannot arise.
  3. Each poem is shown as its first 6 verses.

The rubric has 7 aspects scored 1-5 with justifications; direct_textual_relevance has an
inverted scale (1 = best = least literal matching).

Resumable: already-scored (system, corpus, qid) rows are skipped on rerun.

Usage:
  python3 analysis/rejudge.py --systems all --corpora all --model gemini-3.5-flash

Auth: Vertex AI with application-default credentials; set VERTEX_PROJECT.
"""
import argparse
import ast
import concurrent.futures as cf
import gzip
import json
import os
import re
import sys
import threading
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, DERIVED, ROOT  # noqa: E402

RUNS = os.path.join(ROOT, "analysis", "runs")
OUT = os.path.join(DERIVED, "trackB_rejudged.csv")
METRICS = ["semantic_relevance", "thematic_coherence", "metaphor_understanding",
           "semantic_coverage", "direct_textual_relevance", "precision", "overall_quality"]
SYSTEMS = ["gte", "jina", "silma", "labse", "muffakir", "qwen3", "qwen3_8b",
           "bgem3", "e5", "bm25"]
TOPK = 3

PROMPT = """
أنت خبير في الأدب العربي والشعر الكلاسيكي. سأقدم لك استعلام بحث (prompt) و {k} نتائج مستردة من فهرس للشعر العربي. كل نتيجة تحتوي على الأبيات الشعرية.

المطلوب منك تقييم جودة النتائج المستردة وفقاً للمعايير التالية:

**الاستعلام:** {query}

**النتائج المستردة:**
{results}

**معايير التقييم:**
قيم كل معيار بدرجة من 1 إلى 5 (1 = ضعيف جداً، 5 = ممتاز):

1. **الصلة الدلالية (Semantic Relevance)**: مدى ارتباط النتائج بالاستعلام من الناحية الدلالية والمعنوية
2. **الترابط الموضوعي (Thematic Coherence)**: مدى تناسق النتائج مع بعضها البعض موضوعياً
3. **فهم الاستعارات (Metaphor Understanding)**: مدى فهم النظام للاستعارات والمجازات في الاستعلام وإرجاع نتائج تتماشى مع هذه الاستعارات والمجازات
4. **التغطية الدلالية (Semantic Coverage)**: مدى تغطية النتائج لجوانب مختلفة من الاستعلام
5. **المباشرة النصية (Direct Textual Relevance)**: مدى احتواء النتائج على إشارات مباشرة أو كلمات مشابهة للاستعلام (نريد أن نقيس هل النظام يعتمد على التطابق المباشر للكلمات أم يفهم المعنى الضمني، ولهذا فإن الدرجة 1 هي أفضل درجة بالنسبة لهذا المعيار لأنها تمثل قدرة النظام لفهم المعنى الضمني)

**التقييم العام:**
1. **دقة الاسترداد (Precision)**: هل جميع النتائج ال{k} ذات صلة بالاستعلام؟ أم أن بعضها غير ذي صلة؟
2. **تقييم شامل (Overall Quality)**: تقييم عام لجودة النتائج

**تعليمات الإجابة:**
- اكتب تقييمك في تنسيق JSON فقط
- لا تضف أي نص إضافي خارج JSON
- استخدم الأرقام فقط للدرجات (1-5)
- اكتب تبريراً موجزاً لكل درجة

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
  "semantic_relevance_justification": "...",
  "thematic_coherence_justification": "...",
  "metaphor_understanding_justification": "...",
  "semantic_coverage_justification": "...",
  "direct_textual_relevance_justification": "...",
  "precision_justification": "...",
  "overall_quality_justification": "..."
}}
```
"""


def poem_texts(corpus):
    if corpus == "diwan":
        df = pd.read_csv(f"{DATA}/diwan_with_explanations.csv")
        df["pid"] = df["poem_id"].astype(str)

        def txt(r):
            try:
                return "\n".join(ast.literal_eval(r["verses_list"])[:6])
            except Exception:
                return str(r.get("verses_text", ""))[:600]
    else:
        df = pd.read_csv(f"{DATA}/flipp_flop_ai.csv")
        df["pid"] = df["id"].astype(str)

        def txt(r):
            return "\n".join(str(r["poem_verses"]).split("\n")[:6])
    return {r["pid"]: txt(r) for _, r in df.iterrows()}


def load_run(system, corpus):
    fn = os.path.join(RUNS, f"{system}__{corpus}__verses__thematic.json.gz")
    with gzip.open(fn, "rt", encoding="utf-8") as f:
        return json.load(f)


def parse_json(text):
    t = re.sub(r"```(?:json)?", "", text).strip().rstrip("`")
    for cand in re.findall(r"\{.*\}", t, re.DOTALL) or [t]:
        try:
            d = json.loads(cand)
            if isinstance(d, dict) and any(m in d for m in METRICS):
                return d
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="all")
    ap.add_argument("--corpora", default="all")
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", default=os.environ.get("VERTEX_PROJECT", ""),
                    help="Google Cloud project id for Vertex AI")
    ap.add_argument("--location", default="global")
    args = ap.parse_args()

    from google import genai as google_genai
    from google.genai.types import HttpOptions
    client = google_genai.Client(vertexai=True, project=args.project, location=args.location,
                                 http_options=HttpOptions(api_version="v1"))

    systems = SYSTEMS if args.systems == "all" else args.systems.split(",")
    corpora = ["diwan", "fannorflop"] if args.corpora == "all" else args.corpora.split(",")

    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = {(r.system, r.corpus, r.qid) for r in prev.itertuples()}
        print(f"resuming: {len(done)} rows already scored")

    jobs = []
    for corpus in corpora:
        texts = poem_texts(corpus)
        for system in systems:
            for row in load_run(system, corpus)[: args.limit or None]:
                if (system, corpus, row["qid"]) in done:
                    continue
                blocks = []
                for i, pid in enumerate(row["top_pids"][:TOPK], 1):
                    blocks.append(f"{i}. الأبيات:\n{texts.get(str(pid), '')}")
                jobs.append({"system": system, "corpus": corpus, "qid": row["qid"],
                             "query": row["prompt"], "pids": row["top_pids"][:TOPK],
                             "prompt": PROMPT.format(k=TOPK, query=row["prompt"],
                                                     results="\n\n".join(blocks))})
    print(f"{len(jobs)} judgments to run on {args.model}")
    if not jobs:
        return

    lock = threading.Lock()
    results, errors = [], [0]

    def work(job):
        for attempt in range(4):
            try:
                r = client.models.generate_content(model=args.model, contents=job["prompt"])
                d = parse_json(r.text or "")
                if d is None:
                    raise ValueError("unparseable")
                rec = {k: job[k] for k in ("system", "corpus", "qid", "query")}
                rec["pids"] = ";".join(map(str, job["pids"]))
                for m in METRICS:
                    v = d.get(m)
                    rec[m] = pd.to_numeric(str(v).strip()[:3].rstrip("."), errors="coerce") \
                        if v is not None else None
                    rec[m + "_just"] = str(d.get(m + "_justification", ""))[:400]
                return rec
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    with lock:
                        errors[0] += 1
                    return None
                time.sleep(2 ** attempt + 0.5)
        return None

    def checkpoint():
        """Append completed rows to OUT so an interrupted run never loses finished work."""
        if not results:
            return
        new = pd.DataFrame(results)
        if os.path.exists(OUT) and os.path.getsize(OUT) > 10:
            new = pd.concat([pd.read_csv(OUT), new], ignore_index=True)
            new = new.drop_duplicates(subset=["system", "corpus", "qid"], keep="last")
        new.to_csv(OUT, index=False)
        results.clear()

    t0 = time.time()
    try:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, rec in enumerate(ex.map(work, jobs), 1):
                if rec:
                    results.append(rec)
                if i % 100 == 0:
                    checkpoint()
                    print(f"  {i}/{len(jobs)}  saved  err={errors[0]}  "
                          f"{time.time() - t0:.0f}s", flush=True)
    finally:
        checkpoint()

    new = pd.read_csv(OUT)
    print(f"wrote {OUT}  rows={len(new)}  failures={errors[0]}")
    print(new.groupby(["corpus", "system"])["overall_quality"]
          .agg(["mean", "count"]).round(2).to_string())


if __name__ == "__main__":
    main()
