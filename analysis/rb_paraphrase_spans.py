"""Rewrite only the quoted wording of each quoting Fann or Flop query with an LLM, keeping its
meaning.

For each of the 80 queries that share a verbatim >=3-token span with the target poem, the span(s)
are marked in the query and an open-weights LLM (Qwen3-32B, 4-bit, local, greedy) is asked to
restate the marked words in different words with the same meaning, leaving the rest of the query
as it is and adding no information. The model sees the query only, never the poem, so it cannot
copy other wording from the poem.

Acceptance check (the paper's definition of quotation): the rewrite must share no >=3-token span
with the target poem, after the same normalisation. A rewrite that fails is retried with the
offending words listed (up to 4 attempts, sampling on retries). Queries still failing after 4
attempts are left out and reported. Every accepted rewrite is then rated by a person, blind to
retrieval results (human_checks/paraphrase_check/).

Writes data/rebuttal/fof_paraphrase_queries.csv (qid, gold_pid, variant=para_span, prompt, ...)

--version v2: v1 changed the meaning of some spans, garbled proper names, and sometimes
"paraphrased" by respelling (e.g. أكثري أو فأقلي -> أكثري أو قلّي). v2 (a) tells the model to keep
proper names exactly (a name is information, not wording) and to preserve the precise meaning, with
worked examples, (b) enables Qwen3 thinking, and (c) runs the acceptance check with the query's
proper-noun tokens (rb_proper_nouns.csv) masked, so a kept name does not count as a quotation.
v2 writes fof_paraphrase_queries_v2.csv, variant para_span_v2. Only rewrites rated valid are
analysed.
"""
import argparse
import json
import os
import re
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, ar_tokens  # noqa: E402
from rb_build_counterfactuals import poems, shared_mask  # noqa: E402

MODEL = "unsloth/Qwen3-32B-bnb-4bit"
OUT = os.path.join(DATA, "rebuttal", "fof_paraphrase_queries.csv")

PROMPT = """أنت محرر لغوي عربي. أمامك استعلام بحث كتبه شخص يتذكر قصيدة عربية، وفيه كلمات منقولة حرفياً من القصيدة، وقد وُضعت بين العلامتين ⟦ و ⟧.

المطلوب: أعد كتابة الاستعلام كاملاً بحيث:
1. تُستبدل الكلمات الموجودة بين ⟦ ⟧ بصياغة أخرى بكلمات مختلفة تؤدي المعنى نفسه (إعادة صياغة، لا حذف).
2. يبقى باقي الاستعلام كما هو قدر الإمكان.
3. لا تُضف أي معلومة غير موجودة في الاستعلام الأصلي، ولا تحذف أي معلومة.
4. لا تستخدم العلامتين ⟦ ⟧ في الناتج.
{feedback}
الاستعلام:
{query}

أجب بصيغة JSON فقط: {{"rewrite": "..."}}"""


PROMPT_V2 = """أنت محرر لغوي عربي متمكن من الشعر العربي القديم. أمامك استعلام بحث كتبه شخص يتذكر قصيدة، وفيه كلمات منقولة حرفياً من القصيدة وُضعت بين العلامتين ⟦ و ⟧.

المطلوب: أعد كتابة الاستعلام بحيث تؤدي الكلمات التي بين ⟦ ⟧ المعنى نفسه تماماً بألفاظ أخرى، كما لو أن الباحث تذكّر معنى البيت لا لفظه.

القواعد:
1. حافظ على المعنى الدقيق للكلمات المنقولة، ولا تغيّر الصورة أو المعنى. افهم معنى البيت أولاً ثم عبّر عنه بلغة عربية فصيحة واضحة.
2. لا يكفي تغيير حرف أو صيغة الكلمة أو ترتيب الكلمات؛ استخدم ألفاظاً مختلفة فعلاً.
3. أسماء العلم (أشخاص، قبائل، أماكن، أيام) تبقى كما هي حرفياً ولا تُغيَّر.
4. باقي الاستعلام خارج ⟦ ⟧ يبقى كما هو.
5. لا تُضف معلومة جديدة ولا تحذف معلومة.
6. لا تستخدم ⟦ ⟧ في الناتج.

أمثلة:
- "غزل فيه ⟦القمر يغسل وجه البحر⟧" ← "غزل فيه ضوء القمر ينعكس على سطح الماء"
- "وصف ⟦سيف يقطع الهام كالبرق⟧" ← "وصف سيف يشق الرؤوس بسرعة الوميض"
- "مدح ⟦خالد بن جعفر⟧ وكرمه" ← "مدح خالد بن جعفر وكرمه" (اسم علم يبقى كما هو)
{feedback}
الاستعلام:
{query}

أجب بصيغة JSON فقط: {{"rewrite": "..."}}"""


def lcn(q, ptoks, cap=12, mask=()):
    qt = [t for t in ar_tokens(q) if t not in set(mask)]
    best = 0
    for n in range(1, min(cap, len(qt)) + 1):
        g = {tuple(ptoks[i:i + n]) for i in range(len(ptoks) - n + 1)}
        if any(tuple(qt[i:i + n]) in g for i in range(len(qt) - n + 1)):
            best = n
        else:
            break
    return best


def offending(q, ptoks, n=3, mask=()):
    qt = [t for t in ar_tokens(q) if t not in set(mask)]
    g = {tuple(ptoks[i:i + n]) for i in range(len(ptoks) - n + 1)}
    return sorted({" ".join(qt[i:i + n]) for i in range(len(qt) - n + 1) if tuple(qt[i:i + n]) in g})


def mark(prompt, ptoks):
    surf, mask = shared_mask(prompt, ptoks)
    out, inside = [], False
    for t, m in zip(surf, mask):
        if m and not inside:
            out.append("⟦" + t)
            inside = True
        elif m:
            out.append(t)
        else:
            if inside:
                out[-1] += "⟧"
                inside = False
            out.append(t)
    if inside:
        out[-1] += "⟧"
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tries", type=int, default=4)
    ap.add_argument("--version", choices=["v1", "v2"], default="v1")
    args = ap.parse_args()
    v2 = args.version == "v2"
    tmpl = PROMPT_V2 if v2 else PROMPT
    out_fn = OUT.replace(".csv", "_v2.csv") if v2 else OUT
    names = {}
    if v2:
        pn = pd.read_csv(os.path.join(os.path.dirname(OUT), "..", "..", "results", "derived",
                                      "rb_proper_nouns.csv"))
        pn = pn[pn.corpus == "fannorflop"].fillna({"names": ""})
        names = {r.qid: [t for n in r.names.split("|") if n for t in ar_tokens(n)]
                 for r in pn.itertuples()}
    P = poems()
    cf = pd.read_csv(os.path.join(DATA, "rebuttal", "fof_counterfactual_queries.csv"))
    todo = cf[(cf.variant == "orig") & (cf.lcn_orig >= 3)].reset_index(drop=True)
    ptok = {g: ar_tokens(" ".join(P[str(g)])) for g in todo.gold_pid}
    todo["marked"] = [mark(p, ptok[g]) for p, g in zip(todo.prompt, todo.gold_pid)]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="cuda:0",
                                                 torch_dtype=torch.bfloat16)
    model.eval()

    def generate(prompts, sample):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=v2)
                 for p in prompts]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        kw = dict(do_sample=True, temperature=0.8, top_p=0.95) if sample else dict(do_sample=False)
        with torch.inference_mode():
            g = model.generate(**enc, max_new_tokens=2048 if v2 else 200, pad_token_id=tok.pad_token_id, **kw)
        return [tok.decode(x, skip_special_tokens=True) for x in g[:, enc["input_ids"].shape[1]:]]

    def parse(raw):
        raw = raw.split("</think>")[-1]
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            return json.loads(m.group(0))["rewrite"].strip() if m else None
        except Exception:
            return None

    state = {i: {"tries": 0, "rewrite": None, "feedback": ""} for i in range(len(todo))}
    for attempt in range(args.max_tries):
        pending = [i for i, s in state.items() if s["rewrite"] is None]
        if not pending:
            break
        bs = 4 if v2 else 16  # thinking traces are long; 16 x 2048 tokens OOMs on 32 GB
        for b in range(0, len(pending), bs):
            ids = pending[b:b + bs]
            prompts = [tmpl.format(query=todo.marked[i], feedback=state[i]["feedback"])
                       for i in ids]
            for i, raw in zip(ids, generate(prompts, sample=attempt > 0)):
                s = state[i]
                s["tries"] += 1
                rw = parse(raw)
                g = todo.gold_pid[i]
                mk = names.get(todo.qid[i], [])
                if rw and "⟦" not in rw and lcn(rw, ptok[g], mask=mk) < 3:
                    s["rewrite"] = rw
                else:
                    bad = offending(rw, ptok[g], mask=mk) if rw else []
                    s["feedback"] = ("\nتنبيه: المحاولة السابقة ما زالت تنقل من القصيدة حرفياً هذه "
                                     "الكلمات، فغيّرها: " + "، ".join(bad) + "\n") if bad else ""
        print(f"attempt {attempt + 1}: accepted "
              f"{sum(s['rewrite'] is not None for s in state.values())}/{len(todo)}", flush=True)

    rows = []
    for i, s in state.items():
        r = todo.iloc[i]
        rows.append({"qid": r.qid, "gold_pid": r.gold_pid,
                     "variant": "para_span_v2" if v2 else "para_span",
                     "lcn_orig": r.lcn_orig, "original": r.prompt, "marked": r.marked,
                     "prompt": s["rewrite"], "tries": s["tries"],
                     "lcn_new": lcn(s["rewrite"], ptok[r.gold_pid]) if s["rewrite"] else None,
                     "lcn_new_names_masked": lcn(s["rewrite"], ptok[r.gold_pid],
                                                 mask=names.get(r.qid, []))
                     if s["rewrite"] else None})
    df = pd.DataFrame(rows)
    df.to_csv(out_fn, index=False)
    print(f"wrote {out_fn}: {df.prompt.notna().sum()} accepted, {df.prompt.isna().sum()} failed")


if __name__ == "__main__":
    main()
