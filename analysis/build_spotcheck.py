"""Build the blinded false-negative spot-check for known-item evaluation.

The problem. Track A assumes exactly one correct poem per query. In corpora of 3.5k-4.7k poems an
abstract query may describe several poems, so a "failure" (gold poem outside the top 5) may be a
system returning a perfectly good answer that happens not to be the one we labelled. Limitations
currently concedes this in words. This task measures it.

Design.
  * Failures: queries where the strongest system (Qwen3-8B, verse mode) ranked the gold poem
    outside the top 5. We show its top-1 retrieved poem, never the gold one.
  * Controls: queries where the same system ranked the gold poem first. We show that poem, which
    is by construction a correct answer. Controls are indistinguishable from failures in the UI.
  * Everything is shuffled together and stripped of provenance, so the annotator cannot tell a
    failure item from a control.

The controls are the point of the design: they calibrate leniency. If controls score 2 and failure
items score 0, the annotator is discriminating and the failure rate is trustworthy. If controls and
failures score alike, the annotator is not discriminating and the estimate is reported as such.

Outputs (spotcheck/): task.csv, key.csv (not shown to the annotator), spotcheck.html, GUIDELINES.md
"""
import gzip
import html
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_annotation import poem_texts  # noqa: E402
from common import ROOT  # noqa: E402

RUNS = os.path.join(ROOT, "analysis", "runs")
OUT = os.path.join(ROOT, "spotcheck")
os.makedirs(OUT, exist_ok=True)
RNG = 20260803
SYSTEM = "qwen3_8b"
N_FAIL_PER_CORPUS = 25
N_CTRL_PER_CORPUS = 6


def load_run(corpus):
    fn = os.path.join(RUNS, f"{SYSTEM}__{corpus}__verses__gold.json.gz")
    with gzip.open(fn, "rt", encoding="utf-8") as f:
        return json.load(f)


def main():
    items, key_rows = [], []
    for corpus in ("diwan", "fannorflop"):
        rows = load_run(corpus)
        texts = poem_texts(corpus)
        fails, ctrls = [], []
        for r in rows:
            rk = r.get("rank_of_gold")
            if rk is None or rk > 5:
                fails.append(r)
            elif rk == 1:
                ctrls.append(r)
        f_pick = pd.Series(range(len(fails))).sample(
            min(N_FAIL_PER_CORPUS, len(fails)), random_state=RNG).tolist()
        c_pick = pd.Series(range(len(ctrls))).sample(
            min(N_CTRL_PER_CORPUS, len(ctrls)), random_state=RNG).tolist()

        for i in f_pick:
            r = fails[i]
            shown = str(r["top_pids"][0])
            items.append({"corpus": corpus, "qid": r["qid"], "query": r["prompt"],
                          "poem_id": shown, "verses": texts.get(shown, "")})
            key_rows.append({"corpus": corpus, "qid": r["qid"], "poem_id": shown,
                             "kind": "failure", "rank_of_gold": r.get("rank_of_gold"),
                             "gold_pid": r["gold_pid"]})
        for i in c_pick:
            r = ctrls[i]
            shown = str(r["gold_pid"])
            items.append({"corpus": corpus, "qid": r["qid"], "query": r["prompt"],
                          "poem_id": shown, "verses": texts.get(shown, "")})
            key_rows.append({"corpus": corpus, "qid": r["qid"], "poem_id": shown,
                             "kind": "control", "rank_of_gold": 1,
                             "gold_pid": r["gold_pid"]})

    df = pd.DataFrame(items).sample(frac=1.0, random_state=7).reset_index(drop=True)
    df.insert(0, "item_id", [f"s{i:03d}" for i in range(len(df))])
    key = pd.DataFrame(key_rows).merge(
        df[["item_id", "corpus", "qid", "poem_id"]], on=["corpus", "qid", "poem_id"], how="right")
    df["satisfies"] = ""
    df["note"] = ""
    df[["item_id", "corpus", "qid", "query", "poem_id", "verses", "satisfies", "note"]] \
        .to_csv(os.path.join(OUT, "task.csv"), index=False)
    key.to_csv(os.path.join(OUT, "key.csv"), index=False)
    print(f"{len(df)} items ({(key.kind=='failure').sum()} failures, "
          f"{(key.kind=='control').sum()} controls)")
    write_html(df)
    write_guidelines()
    print(f"wrote {OUT}/task.csv, key.csv, spotcheck.html, GUIDELINES.md")


def write_html(df):
    blocks = []
    for r in df.itertuples():
        q = html.escape(str(r.query))
        verses = html.escape(str(r.verses)).replace("\n", "<br>")
        opts = "".join(
            f'<label><input type="radio" name="r_{r.item_id}" value="{v}"> {v} &ndash; {t}</label>'
            for v, t in [(2, "yes, fully answers the request"),
                         (1, "partly / arguably"),
                         (0, "no")])
        blocks.append(f'''<article id="{r.item_id}">
<div class="q" dir="rtl">{q}</div>
<div class="poem" dir="rtl">{verses}</div>
<div class="ctl"><span class="lbl">Does this poem satisfy the request?</span>{opts}
<input class="note" placeholder="note (optional)" name="n_{r.item_id}"></div></article>''')
    body = "\n".join(blocks)
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Known-item spot check</title><style>
body{{font:15px/1.6 system-ui,sans-serif;max-width:900px;margin:0 auto;padding:24px;background:#faf9f7;color:#1a1a1a}}
h1{{font-size:22px}}
article{{background:#fff;border:1px solid #e2e0dc;border-radius:8px;padding:14px;margin:16px 0}}
.q{{font-family:"Amiri","Noto Naskh Arabic",serif;font-size:18px;font-weight:600;
border-bottom:1px solid #eee;padding-bottom:8px;margin-bottom:10px}}
.poem{{font-family:"Amiri","Noto Naskh Arabic",serif;font-size:19px;line-height:2;margin-bottom:10px}}
.ctl{{display:flex;flex-wrap:wrap;gap:12px;align-items:center;border-top:1px solid #eee;padding-top:10px}}
.lbl{{font-size:13px;color:#555}} label{{font-size:13px;cursor:pointer}}
.note{{flex:1;min-width:200px;padding:5px 8px;border:1px solid #ddd;border-radius:5px;font-size:13px}}
#bar{{position:sticky;bottom:0;background:#fff;border-top:2px solid #333;padding:12px;display:flex;
gap:14px;align-items:center;margin-top:24px}}
button{{padding:9px 18px;font-size:14px;border:0;border-radius:6px;background:#1a1a1a;color:#fff;cursor:pointer}}
#prog{{font-variant-numeric:tabular-nums}}
</style></head><body>
<h1>Known-item spot check</h1>
<p>Each card shows a <strong>search request</strong> and <strong>one poem</strong> a system returned
for it. Judge only this: <em>would this poem satisfy someone who issued that request?</em> Do not
try to guess whether it is the poem we originally had in mind &mdash; that is exactly what we are
testing. Ignore poetic quality, author and era. See <code>GUIDELINES.md</code>. Progress saves in
this browser; click <strong>Download CSV</strong> when finished.</p>
{body}
<div id="bar"><span id="prog"></span><button onclick="dl()">Download CSV</button></div>
<script>
const K='poetry_spotcheck_v1';
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
function dl(){{const s=state();let out='item_id,satisfies,note\\n';
 document.querySelectorAll('article').forEach(a=>{{const id=a.id;
  const r=s['r_'+id]??'';const n=(s['n_'+id]??'').replace(/"/g,'""');
  out+=id+','+r+',"'+n+'"\\n'}});
 const b=new Blob([out],{{type:'text/csv;charset=utf-8'}});const u=URL.createObjectURL(b);
 const a=document.createElement('a');a.href=u;a.download='spotcheck_answers.csv';a.click()}}
document.addEventListener('input',save);load();
</script></body></html>"""
    with open(os.path.join(OUT, "spotcheck.html"), "w", encoding="utf-8") as f:
        f.write(doc)


def write_guidelines():
    g = """# Spot-check guidelines — does the retrieved poem satisfy the request?

Each card shows one **search request** (an Arabic description of a poem someone is looking for) and
**one poem** that a retrieval system returned for it.

## Your task

Judge only this: **would this poem satisfy someone who issued that request?**

| Score | Meaning |
|---|---|
| **2** | Yes. Someone searching with this request would be satisfied to receive this poem. |
| **1** | Partly. It fits some of the request but misses a clear part of it, or the fit is arguable. |
| **0** | No. This poem does not answer the request. |

## Important

- **Do not** try to work out whether this is the specific poem the request was written for. Some
  cards show that poem and some do not, and you are not told which. Guessing defeats the purpose.
- A request may legitimately describe more than one poem. If this poem is a genuine answer, it
  scores 2 even if you suspect a different poem was intended.
- Ignore poetic quality, fame, author and era.
- If you are torn, pick the lower score and add a note.
- Only some verses of each poem are shown; judge on what you can see.

## Why this exists

Our benchmark assumes exactly one correct poem per request. This task measures how often that
assumption undercounts a system: how many apparent failures are actually acceptable answers.
"""
    with open(os.path.join(OUT, "GUIDELINES.md"), "w", encoding="utf-8") as f:
        f.write(g)


if __name__ == "__main__":
    main()
