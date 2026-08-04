# What the Query Remembers

Code, per-query outputs and human judgments for our study of retrieval over Classical Arabic
poetry. This repository accompanies an anonymous submission, so it carries no author or
institution information.

The study benchmarks nine dense encoders and BM25 over two poetry corpora, three indexing modes and
two evaluation protocols. Its central claim is that which retriever wins is governed by how much of
a poem's wording the query still carries, and that a corpus-level ranking therefore reports its
query mix as much as its systems.

Every number in the paper is recomputed from the files in this repository by the scripts in
`analysis/`. Nothing is typed in by hand. Running `python3 analysis/verify_paper.py` checks all 76
numeric claims against the released artifacts and exits non-zero if any of them disagree.

## Layout

```
analysis/            all scripts (see the table below)
analysis/runs/       per-query ranked lists, one gzipped JSON per
                     (system, corpus, indexing mode, query set)
results/derived/     every table and figure in the paper is built from these CSVs
data/                the released query benchmark
annotation/          thematic relevance annotation: guidelines, blinded task, judgments
spotcheck/           known-item spot check: guidelines, blinded task, judgments
paper/               the paper source, so that the table files can be regenerated in place
```

`paper/` holds the LaTeX body, the bibliography and the generated table files, but not the ACL
style files, which come from the official template. Every `tab_*.tex` in it is written by
`make_tables.py` and `appendix_tables.py` from the CSVs in `results/derived/`, so the claim that no
number is typed in by hand can be checked by deleting them and regenerating.

The two poetry corpora are distributed separately as a data archive, because of their size. Unpack
it into `data/` before running anything that re-embeds text. The analysis scripts that only read
`analysis/runs/` and `results/derived/` work without it.

## Query benchmark

`data/` holds the queries we release, in three sets that sit at different points on the
query-abstraction spectrum.

| File | n | Register |
|---|---|---|
| `accuracy_prompts.csv` | 244 | tip of the tongue, written from the poem, may quote it |
| `diwan_prompts_final.csv` | 258 | abstract description of what a poem is about, no quotation |
| `gemini_prompts.csv` | 200 | thematic topics, written before the corpora were inspected |

The first two are known-item sets and carry a gold poem id. The thematic set has no target poem and
is used only for the judged track. All queries were written by the authors from the poem text alone.
Writers saw neither the reference explanations distributed with one corpus nor the model-generated
explanations, which did not exist until after the queries were written.

## Per-query outputs

`analysis/runs/` contains one file per cell of the grid, named
`{system}__{corpus}__{mode}__{queryset}.json.gz`. Each file is a list of records:

```json
{"qid": "g0", "prompt": "...", "gold_pid": "poem_2774", "rank_of_gold": 1,
 "top_pids": ["poem_2774", "..."], "top_scores": [0.83, "..."]}
```

`rank_of_gold` is null for thematic queries, which have no target. Systems are `bm25`, `gte`,
`jina`, `silma`, `labse`, `muffakir`, `qwen3`, `qwen3_8b`, `bgem3`, `e5`. Modes are `verses`,
`expl_ai` and `expl_ff`.

## Judge outputs and human judgments

`results/derived/trackB_rejudged.csv` holds all 4,000 judgments from the primary judge, including
the free-text justification for each of the seven rubric aspects. `trackB_judge2.csv` holds 2,000
judgments from a second, open-weights judge run locally.

`annotation/` and `spotcheck/` contain two blinded human tasks. In both, system identity was
stripped and items were shuffled before annotation. `key.csv` records which system produced each
item and was not shown to the annotator. The spot check additionally mixes in control items where
the labelled gold poem is the one displayed, indistinguishable from the rest, so that annotator
leniency can be measured rather than assumed.

## Scripts

Ordered roughly as the pipeline runs.

| Script | What it does |
|---|---|
| `unified_run.py` | the retrieval pipeline. One harness for BM25 and every dense encoder, both corpora, all modes and query sets. Writes `analysis/runs/`. |
| `query_features.py` | per-query lexical statistics: containment, Jaccard, longest shared verbatim n-gram |
| `eacl_metrics.py` | Recall@1/@5, MRR, nDCG@10 with bootstrap confidence intervals, and paired exact McNemar tests |
| `mechanism.py` | dose-response curves and the within-query correlation between overlap change and retrieval change |
| `interaction.py` | the quotation-slope interaction and its robustness to the split threshold |
| `fusion.py` | reciprocal rank fusion over index configurations |
| `rejudge.py` | the primary LLM judge over the thematic track |
| `judge2.py` | the second judge, open weights, run locally |
| `judge_agreement2.py` | agreement between the two judges, and between each judge and the human data |
| `trackb_analysis.py` | thematic track scores and the dissociation between the two protocols |
| `build_annotation.py`, `annotation_analysis.py` | build and score the blinded relevance task |
| `build_spotcheck.py`, `spotcheck_analysis.py` | build and score the blinded known-item spot check |
| `multiplicity.py` | Holm correction within each family of tests |
| `norm_ablation.py` | the text normalisation ablation |
| `make_tables.py`, `appendix_tables.py`, `eacl_figures.py` | emit every table and figure |
| `verify_paper.py` | recompute every numeric claim in the paper and fail on any mismatch |

## Reproducing

```bash
pip install -r requirements.txt

python3 analysis/verify_paper.py        # checks all released numbers, no GPU needed
python3 analysis/make_tables.py         # rebuild the paper tables
python3 analysis/appendix_tables.py
python3 analysis/eacl_figures.py
```

Those four read only the released CSVs and rank dumps, so they run on CPU in under a minute.

Re-running retrieval needs the data archive and a GPU:

```bash
python3 analysis/unified_run.py --systems all --corpora all
```

The full grid takes about half an hour on a single 32 GB consumer GPU. Re-running the second judge
downloads a 4-bit 27B model and takes about 70 minutes for 2,000 cells:

```bash
python3 analysis/judge2.py --nqueries 100
```

The primary judge calls a hosted API and needs credentials. Set `VERTEX_PROJECT` in the environment.
Note that the model used to generate the explanations was withdrawn from that API during the work,
so that step is no longer reproducible with the model that produced it. This is one reason all
per-query judge outputs are released here rather than only aggregate scores.

## Notes and caveats

Two of the ten systems deserve a note. The embedding-based overlap measure in `mechanism.py` uses
one fixed encoder for every system so that all systems are scored on the same axis, and that encoder
is `gte`, which is itself one of the ten. Its own correlation is therefore not independent of the
measure. Conclusions in the paper are reported with that row excluded as well as included.

The human judgments come from a single annotator, who is one of the authors. The spot check includes
control items to show that the annotator discriminated, but the study reports no inter-annotator
agreement.

Text normalisation differs by system on purpose. Dense encoders receive diacritic and tatweel
stripping only. BM25 and the lexical overlap measures receive the fuller normalisation that also
unifies alef, ya and ta-marbuta, which is standard for Arabic lexical matching but costs dense
encoders accuracy. `norm_ablation.py` measures the difference.

## Licence

To be added on de-anonymisation. The two poetry corpora are redistributed under the terms set by
their original publishers, which are cited in the paper.
