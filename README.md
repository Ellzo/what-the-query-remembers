# What the Query Remembers

Code, per-query outputs and human judgments for the paper *What the Query Remembers: Query
Abstraction Governs Lexical and Dense Retrieval of Classical Arabic Poetry*, by Omar Farouk Zouak,
Selma Mani, Larbi Said Chikh and Raid Abderrezak Ouahioune (National School of Artificial
Intelligence, Algeria).

The study benchmarks nine dense encoders and BM25 over two corpora of Classical Arabic poetry,
three indexing modes and two evaluation protocols. Which retriever wins depends on how much of a
poem's wording the query still carries. Editing the wording of the same queries confirms this:
adding one remembered half-verse helps BM25 far more than the strongest dense encoder, and
rewriting the quoted words removes BM25's lead.

Every number in the paper is recomputed from the files in this repository. Running
`python3 analysis/verify_paper.py` checks all 197 numeric claims against the released files and
exits with a non-zero status if any of them disagree.

## Layout

```
analysis/              all scripts (see the table below)
analysis/runs/         per-query ranked lists, one gzipped JSON per
                       (system, corpus, indexing mode, query set)
analysis/runs_rebuttal/ per-query ranked lists for the edited queries
results/derived/       the CSVs every table, figure and claim is built from
data/                  the query benchmark and the edited queries (the corpora are a separate download)
annotation/            thematic relevance annotation: guidelines, blinded task, both raters' grades
spotcheck/             known-item spot check: guidelines, blinded task, judgments
human_checks/          explanation audit, ratings of LLM rewrites, hand rewrites of quoting queries
paper/                 the LaTeX source, so the tables and figures can be regenerated in place
```

`paper/` holds the LaTeX body, the bibliography, the figures and the generated tables, but not the
ACL style files, which come from the [official template](https://github.com/acl-org/acl-style-files).
Every `tab_*.tex` is written by `make_tables.py` or `appendix_tables.py` from `results/derived/`, so
you can delete them and regenerate them. The paper builds with XeLaTeX, which the Arabic examples
need (Amiri and TeXGyreTermesX fonts).

## Data

`data/` holds the queries, in three sets at different points of the query-abstraction spectrum.

| File | n | Register |
|---|---|---|
| `accuracy_prompts.csv` | 244 | tip of the tongue, written from the poem, may quote it |
| `diwan_prompts_final.csv` | 258 | abstract description of what a poem is about, no quotation |
| `gemini_prompts.csv` | 200 | thematic topics, written before the corpora were inspected |

The first two are known-item sets with a gold poem id. The thematic set has no target poem and is
used only in the judged track. `accuracy_prompts.csv` has 245 rows but 244 distinct queries; the
pipeline drops the duplicate at load time.

`data/rebuttal/` holds the edited queries: a verbatim half-verse appended to each non-quoting query
(three positions), the quoted words deleted, rewritten by an LLM, or rewritten by hand.

The two corpora, with our generated verse explanations added, are attached to the
[v1.0 release](https://github.com/Ellzo/what-the-query-remembers/releases/tag/v1.0) as
`corpora.zip`. Unzip it into `data/`:

```bash
unzip corpora.zip -d data/
```

This gives `data/flipp_flop_ai.csv` (4,687 Fann or Flop poems with at most 16 verses) and
`data/diwan_with_explanations.csv` (3,500 Diwan poems, 70 from each of the 50 most frequent poets).
`data/README.md` describes their columns. Only retrieval, `query_features.py`, `mechanism.py`,
`rb_query_covariates.py` and `rb_corpus_checks.py` need them. Everything else reads
`analysis/runs/` and `results/derived/`.

## Per-query outputs

`analysis/runs/` contains one file per cell of the grid, named
`{system}__{corpus}__{mode}__{queryset}.json.gz`. Each file is a list of records:

```json
{"qid": "g0", "prompt": "...", "gold_pid": "poem_2774", "rank_of_gold": 1,
 "top_pids": ["poem_2774", "..."], "top_scores": [0.83, "..."]}
```

`rank_of_gold` is null for thematic queries, which have no target. Systems are `bm25`, `gte`,
`jina`, `silma`, `labse`, `muffakir`, `qwen3`, `qwen3_8b`, `bgem3`, `e5`. Modes are `verses`,
`expl_ai` (our generated explanations) and `expl_ff` (the reference explanations distributed with
Fann or Flop). `analysis/runs_rebuttal/` uses the same format, with the edit variant in place of the
query set.

## Judge outputs and human judgments

`results/derived/trackB_rejudged.csv` holds all 4,000 judgments from the primary judge (Gemini 3.5
Flash), with the free-text justification for each of the seven rubric aspects.
`trackB_judge2.csv` holds 2,000 judgments from Gemma-3-27B-it, and `trackB_judge3_original.csv` and
`trackB_judge3_neutral.csv` hold Qwen3-32B's judgments under the original rubric and under a
rubric without the literal-match aspect.

`annotation/` and `spotcheck/` contain two blinded human tasks. In both, system identity was
removed and items were shuffled before annotation. `key.csv` records which system produced each
item and was not shown to annotators. `annotation/annotations.csv` and
`annotation/annotations_rater2.csv` are the two raters' grades of the same 126 items. The spot check
also mixes in control items where the displayed poem is the gold poem, so annotator leniency can be
measured.

`human_checks/` holds three smaller tasks, each with its guidelines, key and answers: an audit of 50
generated explanations, ratings of the LLM rewrites, and the hand rewrites of the quoting queries.

## Scripts

Ordered roughly as the pipeline runs.

| Script | What it does |
|---|---|
| `unified_run.py` | the retrieval pipeline: one harness for BM25 and every dense encoder, both corpora, all modes and query sets. Writes `analysis/runs/`. |
| `query_features.py` | per-query lexical statistics: containment, Jaccard, longest shared verbatim n-gram |
| `eacl_metrics.py` | Recall@1/@5, MRR, nDCG@10 with bootstrap confidence intervals, and paired exact McNemar tests |
| `interaction.py` | the quotation-slope interaction and its robustness to the split threshold |
| `mechanism.py` | dose-response curves and the within-query correlation between overlap change and retrieval change |
| `fusion.py` | reciprocal rank fusion over index configurations |
| `rejudge.py`, `judge2.py`, `judge3.py` | the three LLM judges over the thematic track |
| `judge_agreement2.py`, `rb_judge_agreement.py` | agreement between judges and with the human data, and the neutral-rubric comparison |
| `trackb_analysis.py` | thematic track scores and the dissociation between the two protocols |
| `build_annotation.py`, `annotation_analysis.py`, `rb_iaa.py` | the blinded relevance task, its scores, and agreement between the two raters |
| `build_spotcheck.py`, `spotcheck_analysis.py` | the blinded known-item spot check |
| `rb_build_counterfactuals.py`, `rb_paraphrase_spans.py`, `rb_ingest_hand_paraphrase.py` | build the edited queries |
| `rb_counterfactual_run.py`, `rb_counterfactual_analysis.py` | retrieve the edited queries and compare them with the originals |
| `rb_slope_metrics.py` | the quotation-slope test on R@1, MRR and nDCG@10 |
| `rb_query_covariates.py`, `rb_proper_nouns.py`, `rb_multivariate.py` | query and target covariates, and the logistic models that control for them |
| `rb_corpus_checks.py` | era, poet frequency, target selection, the Fann or Flop length filter, duplicate poems |
| `multiplicity.py` | Holm correction within each family of tests |
| `norm_ablation.py` | the text normalisation ablation |
| `make_tables.py`, `appendix_tables.py`, `eacl_figures.py` | write every table and figure |
| `verify_paper.py` | recompute every numeric claim in the paper and fail on any mismatch |

The `rb_` prefix marks the analyses added after the first round of review.

## Reproducing

```bash
pip install -r requirements.txt

python3 analysis/verify_paper.py        # checks all 197 claims, no GPU needed
python3 analysis/make_tables.py         # rebuild the paper tables
python3 analysis/appendix_tables.py
python3 analysis/eacl_figures.py
```

These read only the released CSVs and rank dumps, and run on CPU in about a minute.

Rerunning retrieval needs the corpora and a GPU:

```bash
python3 analysis/unified_run.py --systems all --corpora all
python3 analysis/rb_counterfactual_run.py --systems all
```

The full grid takes about half an hour on one 32 GB consumer GPU. `rb_counterfactual_run.py` first
checks that the unedited queries reproduce the released ranks exactly, and refuses to write if they
do not. The open-weights judges download 4-bit models (Gemma-3-27B-it, Qwen3-32B) and run on the
same GPU; the second judge takes about 70 minutes for its 2,000 cells:

```bash
python3 analysis/judge2.py --nqueries 100
python3 analysis/judge3.py --rubric original
python3 analysis/judge3.py --rubric neutral
```

The primary judge calls a hosted API and needs credentials: set `VERTEX_PROJECT` in the
environment. The model used to generate the explanations, Gemini 2.0 Flash, was withdrawn from that
API during this work, so that step cannot be rerun with the same model. This is why the generated
text and every per-query judge output are released.

## Notes

The embedding-based overlap measure in `mechanism.py` uses one fixed encoder for every system so
that all systems are scored on the same axis. That encoder is `gte`, which is itself one of the ten
systems, so its own correlation is not independent of the measure. The paper reports the
conclusions with that row excluded as well as included.

Text normalisation differs by system on purpose. Dense encoders receive diacritic and tatweel
stripping only. BM25 and the lexical overlap measures receive the fuller normalisation that also
unifies alef, ya and ta-marbuta, which is standard for Arabic lexical matching but costs dense
encoders accuracy. `norm_ablation.py` measures the difference.

In Diwan, 274 of the 3,500 poems share their text with another poem. Counting a duplicate of the
gold poem as correct changes verse-mode Recall@5 by at most 0.4 points (`rb_corpus_checks.py`).

## Licence

The code is released under the MIT licence (`LICENSE`). Our own data (the queries, the edited
queries, the generated explanations, the judge outputs, the ranked lists and the human judgments)
is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The poems come from two existing corpora and are redistributed with credit to their creators.
Please cite them if you use the data:

- Fann or Flop (Al Ghallabi et al., EMNLP 2025), released under the Apache 2.0 licence at
  [omkarthawakar/FannOrFlop](https://huggingface.co/datasets/omkarthawakar/FannOrFlop). Its
  reference explanations are part of that release.
- Diwan (Al-Onazi, Nashir and Al-Shargabi, IEEE Access 2025,
  [doi:10.1109/ACCESS.2025.3551161](https://doi.org/10.1109/ACCESS.2025.3551161)). We do not
  relicense the Diwan text; its original terms apply.
