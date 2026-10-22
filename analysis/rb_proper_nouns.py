"""Proper nouns in each known-item query, used as a covariate in rb_multivariate.py.

Two measures:
  pn_llm     number of proper-noun tokens (people, tribes, places, poet names) tagged by Qwen3-32B
             (4-bit, greedy, thinking on), asked to copy each proper noun exactly as it appears and
             told explicitly that genre words and common nouns are not names (without that
             instruction it tagged words such as هجاء, حبيبة, الكحل).
             Only tags that occur verbatim in the query are kept.
  pn_in_poem how many of those tagged tokens also occur in the target poem (normalised), which
             separates "names the poem mentions" from other names.

Writes results/derived/rb_proper_nouns.csv
"""
import json
import os
import re
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, DERIVED, ar_tokens  # noqa: E402
from unified_run import load_queries, load_units  # noqa: E402

MODEL = "unsloth/Qwen3-32B-bnb-4bit"
PROMPT = """استخرج أسماء العلم فقط الواردة في النص التالي، أي الأسماء التي تدل على شخص بعينه أو قبيلة أو مكان أو بلد أو يوم/وقعة بعينها (مثل: عنترة، عبلة، بني جحش، مصر، يوم المجيمر). انسخ كل اسم كما ورد في النص حرفياً.
ليست أسماء علم: أغراض الشعر (هجاء، مدح، رثاء، غزل، فخر، وعيد)، والأسماء العامة (حبيبة، شاعر، قصيدة، الكحل، الحناء، المدينة بمعنى أي مدينة)، والصفات والشتائم. إن كان اللفظ يحتمل أن يكون اسماً عاماً فلا تدرجه.

النص: {q}

أجب بصيغة JSON فقط: {{"names": ["..."]}} (قائمة فارغة إن لم يوجد)"""


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="cuda:0",
                                                 torch_dtype=torch.bfloat16)
    rows = []
    for corpus in ("fannorflop", "diwan"):
        q = load_queries(corpus, "gold")
        units = load_units(corpus, "verses")
        prompts = list(q.prompt)
        out = []
        for b in range(0, len(prompts), 16):
            texts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(q=p)}],
                                             tokenize=False, add_generation_prompt=True,
                                             enable_thinking=True) for p in prompts[b:b + 16]]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.inference_mode():
                g = model.generate(**enc, max_new_tokens=1024, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            out += [tok.decode(x, skip_special_tokens=True)
                    for x in g[:, enc["input_ids"].shape[1]:]]
        for r, raw in zip(q.itertuples(), out):
            raw = raw.split("</think>")[-1]
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            try:
                names = json.loads(m.group(0)).get("names", []) if m else []
            except Exception:
                names = []
            names = [n for n in names if isinstance(n, str) and n.strip() and n in r.prompt]
            ntok = [t for n in names for t in ar_tokens(n)]
            ptok = set(ar_tokens(" ".join(units.get(r.gold_pid, []))))
            rows.append({"corpus": corpus, "qid": r.qid, "names": "|".join(names),
                         "pn_llm": len(ntok), "pn_in_poem": sum(t in ptok for t in ntok),
                         "parsed": m is not None})
        print(corpus, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DERIVED, "rb_proper_nouns.csv"), index=False)
    print(df.groupby("corpus")[["pn_llm", "pn_in_poem"]].mean(), df.parsed.mean())


if __name__ == "__main__":
    main()
