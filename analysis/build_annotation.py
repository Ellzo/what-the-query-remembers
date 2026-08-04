"""E4 - build the blinded human relevance annotation task.

Purpose: an independent, judge-free test of the thematic-track result. The LLM judge says BM25
collapses on thematic queries while dense encoders do not. If a human annotator, blind to system
identity, reproduces that ordering, the thematic track is validated. If not, the finding becomes
"the judge's rubric penalises lexical systems", which is reported instead. Decided in advance.

Design:
  * 20 thematic queries, 10 per corpus, stratified across the query list.
  * 3 systems in verse mode from the UNIFIED pipeline: bm25, jina-v3, and RRF(bm25, jina).
  * top-3 poems each -> pooled, deduplicated, shuffled; all system identity stripped.
  * graded relevance 0/1/2 against the theme, plus a free-text note.

Outputs:
  annotation/task.csv         canonical file (item_id, corpus, query, poem_id, verses, relevance, note)
  annotation/key.csv          which systems retrieved each item, at which rank  (NOT shown to annotator)
  annotation/annotate.html    self-contained offline UI that exports the filled task.csv
  annotation/GUIDELINES.md    instructions (goes in the paper appendix)
"""
import ast
import gzip
import html
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, ROOT  # noqa: E402

RUNS = os.path.join(ROOT, "analysis", "runs")
OUT = os.path.join(ROOT, "annotation")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20260801)
N_QUERIES_PER_CORPUS = 10
TOPN = 3


def load_run(system, corpus, mode="verses", qset="thematic"):
    fn = os.path.join(RUNS, f"{system}__{corpus}__{mode}__{qset}.json.gz")
    with gzip.open(fn, "rt", encoding="utf-8") as f:
        return {r["qid"]: r for r in json.load(f)}


def rrf(list_a, list_b, k=60):
    score = defaultdict(float)
    for lst in (list_a, list_b):
        for rank, pid in enumerate(lst, start=1):
            score[pid] += 1.0 / (k + rank)
    return [p for p, _ in sorted(score.items(), key=lambda kv: -kv[1])]


def poem_texts(corpus):
    if corpus == "diwan":
        df = pd.read_csv(f"{DATA}/diwan_with_explanations.csv")
        df["pid"] = df["poem_id"].astype(str)
        def txt(r):
            try:
                return "\n".join(ast.literal_eval(r["verses_list"]))
            except Exception:
                return str(r.get("verses_text", ""))
    else:
        df = pd.read_csv(f"{DATA}/flipp_flop_ai.csv")
        df["pid"] = df["id"].astype(str)
        def txt(r):
            return str(r["poem_verses"])
    return {r["pid"]: txt(r) for _, r in df.iterrows()}


def main():
    items, key_rows = [], []
    for corpus in ("diwan", "fannorflop"):
        runs = {s: load_run(s, corpus) for s in ("bm25", "jina")}
        texts = poem_texts(corpus)
        qids = sorted(runs["bm25"].keys(), key=lambda q: int(q[1:]))
        pick = [qids[i] for i in np.linspace(0, len(qids) - 1, N_QUERIES_PER_CORPUS).astype(int)]

        for qid in pick:
            query = runs["bm25"][qid]["prompt"]
            lists = {s: runs[s][qid]["top_pids"] for s in runs}
            lists["hybrid"] = rrf(lists["bm25"][:20], lists["jina"][:20])
            pool = {}
            for s, lst in lists.items():
                for rank, pid in enumerate(lst[:TOPN], start=1):
                    pool.setdefault(pid, []).append((s, rank))
            for pid, provenance in pool.items():
                items.append({"corpus": corpus, "qid": qid, "query": query,
                              "poem_id": pid, "verses": texts.get(pid, "")})
                key_rows.append({"corpus": corpus, "qid": qid, "poem_id": pid,
                                 "systems": ";".join(f"{s}@{r}" for s, r in provenance)})

    df = pd.DataFrame(items)
    key = pd.DataFrame(key_rows)
    # shuffle within query so system order carries no signal; then assign opaque item ids
    df = df.sample(frac=1.0, random_state=7).sort_values(["corpus", "qid"], kind="stable")
    df = df.reset_index(drop=True)
    df.insert(0, "item_id", [f"i{i:04d}" for i in range(len(df))])
    key = key.merge(df[["item_id", "corpus", "qid", "poem_id"]],
                    on=["corpus", "qid", "poem_id"], how="right")
    df["relevance"] = ""
    df["note"] = ""

    task_cols = ["item_id", "corpus", "qid", "query", "poem_id", "verses", "relevance", "note"]
    df[task_cols].to_csv(os.path.join(OUT, "task.csv"), index=False)
    key.to_csv(os.path.join(OUT, "key.csv"), index=False)

    n_by_q = df.groupby(["corpus", "qid"]).size()
    print(f"items: {len(df)}  queries: {df.qid.nunique() * 2 if False else len(n_by_q)}  "
          f"mean items/query: {n_by_q.mean():.1f}")
    print(df.groupby("corpus").size().to_string())
    write_html(df)
    write_guidelines()
    print("wrote annotation/task.csv, key.csv, annotate.html, GUIDELINES.md")


def write_html(df):
    blocks = []
    for corpus, g in df.groupby("corpus", sort=False):
        for qid, gg in g.groupby("qid", sort=False):
            q = html.escape(str(gg.iloc[0]["query"]))
            blocks.append(f'<section><h2 dir="rtl">{q}</h2>'
                          f'<div class="meta">{corpus} &middot; {qid} &middot; '
                          f'{len(gg)} poems</div>')
            for r in gg.itertuples():
                verses = html.escape(str(r.verses)).replace("\n", "<br>")
                blocks.append(f'''<article id="{r.item_id}">
<div class="poem" dir="rtl">{verses}</div>
<div class="ctl"><span class="lbl">Relevance to the theme:</span>
{''.join(f'<label><input type="radio" name="r_{r.item_id}" value="{v}"> {v} &ndash; {t}</label>'
         for v, t in [(2, "clearly on-theme"), (1, "partly / tangentially"), (0, "off-theme")])}
<input class="note" placeholder="note (optional)" name="n_{r.item_id}">
</div></article>''')
            blocks.append("</section>")
    body = "\n".join(blocks)
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Poetry relevance annotation</title><style>
body{{font:15px/1.6 system-ui,sans-serif;max-width:900px;margin:0 auto;padding:24px;background:#faf9f7;color:#1a1a1a}}
h1{{font-size:22px}} h2{{font-size:20px;margin:0 0 4px;font-family:"Amiri","Noto Naskh Arabic",serif}}
section{{margin:36px 0;border-top:3px solid #333;padding-top:12px}}
.meta{{color:#777;font-size:12px;margin-bottom:12px}}
article{{background:#fff;border:1px solid #e2e0dc;border-radius:8px;padding:14px;margin:10px 0}}
.poem{{font-family:"Amiri","Noto Naskh Arabic",serif;font-size:19px;line-height:2;margin-bottom:10px}}
.ctl{{display:flex;flex-wrap:wrap;gap:12px;align-items:center;border-top:1px solid #eee;padding-top:10px}}
.lbl{{font-size:13px;color:#555}} label{{font-size:13px;cursor:pointer}}
.note{{flex:1;min-width:200px;padding:5px 8px;border:1px solid #ddd;border-radius:5px;font-size:13px}}
#bar{{position:sticky;bottom:0;background:#fff;border-top:2px solid #333;padding:12px;display:flex;
gap:14px;align-items:center;margin-top:24px}}
button{{padding:9px 18px;font-size:14px;border:0;border-radius:6px;background:#1a1a1a;color:#fff;cursor:pointer}}
#prog{{font-variant-numeric:tabular-nums}}
</style></head><body>
<h1>Thematic relevance annotation</h1>
<p>For each poem, judge only whether it is <em>about the theme stated above it</em>. Ignore poetic
quality, era and author. Systems are anonymised and the order is shuffled. See
<code>GUIDELINES.md</code>. Progress saves automatically in this browser; click
<strong>Download CSV</strong> when finished.</p>
{body}
<div id="bar"><span id="prog"></span><button onclick="dl()">Download CSV</button></div>
<script>
const K='poetry_annot_v1';
function state(){{const s={{}};document.querySelectorAll('input').forEach(i=>{{
 if(i.type==='radio'){{if(i.checked)s[i.name]=i.value}}else if(i.value)s[i.name]=i.value}});return s}}
function save(){{localStorage.setItem(K,JSON.stringify(state()));prog()}}
function prog(){{const n=document.querySelectorAll('article').length;
 const d=Object.keys(state()).filter(k=>k.startsWith('r_')).length;
 document.getElementById('prog').textContent=d+' / '+n+' judged';}}
function load(){{const s=JSON.parse(localStorage.getItem(K)||'{{}}');
 for(const[k,v]of Object.entries(s)){{if(k.startsWith('r_')){{
  const el=document.querySelector(`input[name="${{k}}"][value="${{v}}"]`);if(el)el.checked=true;
 }}else{{const el=document.querySelector(`input[name="${{k}}"]`);if(el)el.value=v}}}}prog()}}
function dl(){{const s=state();let out='item_id,relevance,note\\n';
 document.querySelectorAll('article').forEach(a=>{{const id=a.id;
  const r=s['r_'+id]??'';const n=(s['n_'+id]??'').replace(/"/g,'""');
  out+=id+','+r+',"'+n+'"\\n'}});
 const b=new Blob([out],{{type:'text/csv;charset=utf-8'}});const u=URL.createObjectURL(b);
 const a=document.createElement('a');a.href=u;a.download='annotations.csv';a.click()}}
document.addEventListener('input',save);load();
</script></body></html>"""
    with open(os.path.join(OUT, "annotate.html"), "w", encoding="utf-8") as f:
        f.write(doc)


def write_guidelines():
    g = """# Annotation guidelines — thematic relevance

You will see a series of **themes** (short Arabic phrases). Under each theme are several poems,
retrieved by different search systems. The systems are anonymised and the poems are shuffled, so
you cannot tell which system produced which poem. That is deliberate.

## Your task

For each poem, judge **only** this: is the poem *about* the stated theme?

| Score | Meaning |
|---|---|
| **2** | Clearly on-theme. The theme is a fair description of what this poem is about. |
| **1** | Partly or tangentially on-theme. The theme touches the poem (a passing image, one verse, a related idea) but does not describe it. |
| **0** | Off-theme. The theme does not describe this poem. |

## What NOT to judge

- **Not** poetic quality, eloquence or fame.
- **Not** whether the poem shares *words* with the theme. A poem that repeats the theme's words but
  is about something else is **0**. A poem that never uses the theme's words but is genuinely about
  it is **2**. This distinction is the whole point of the exercise.
- **Not** the author or era.

## Practical notes

- Only a few verses of each poem are shown; judge on what you can see.
- If you are torn between two scores, pick the lower one and add a note.
- Use the note field whenever the case is interesting or the call was hard.
- Work through the file in order. Your progress is saved automatically in the browser.
"""
    with open(os.path.join(OUT, "GUIDELINES.md"), "w", encoding="utf-8") as f:
        f.write(g)


if __name__ == "__main__":
    main()
