# Data

The query benchmark used in the paper, and a description of the two poetry corpora. The corpora are
distributed as `corpora.zip` on the repository's release page because of their size. Unzip it into
this directory, so that `flipp_flop_ai.csv` and `diwan_with_explanations.csv` sit next to the query
files.

Both corpora are redistributed from their original publishers, cited in the paper, with one column
added by us. Please cite the original corpus papers if you use them.

## Corpora

### diwan_with_explanations.csv

3,500 poems, 70 from each of the 50 most frequent poets in the Diwan corpus. 23 columns, of which
the pipeline reads three:

| Column | Meaning |
|---|---|
| `poem_id` | poem identifier, used as the gold label for the Diwan known-item queries |
| `verses_list` | the poem's verses, a Python list literal |
| `verses_ai_explanations` | our generated explanation of each verse, aligned to `verses_list` |

The remaining columns are metadata carried over from the source corpus: poet name, era, country,
meter, rhyme, genre and so on. We did not use them for retrieval.

### flipp_flop_ai.csv

4,687 poems from the Fann or Flop benchmark. The columns the pipeline reads:

| Column | Meaning |
|---|---|
| `id` | poem identifier, used as the gold label for the Fann or Flop known-item queries |
| `poem_verses` | the poem text, newline separated |
| `explanation`, `raw_explanation` | the reference explanations distributed with the source corpus |
| `verses_ai_explanations` | our generated explanation of each verse |

A note on the reference explanations. The source paper describes them as human written. We make no
independent claim about their authorship, and we report them as the explanations distributed with
the corpus. Their surface form does not look hand written to us: 91.7 percent contain the phrase
meaning "each one separately" and 35.9 percent contain "I will explain". Nothing in our analysis
depends on who wrote them. Throughout the paper they are called FoF explanations and are treated as
a second explanation source rather than as a human baseline.

`Unnamed: 0` is a leftover row index from the source file. It is ignored.

## Queries (in this directory)

The released benchmark. 502 known-item queries with a gold poem, and 200 thematic queries without.

| File | Rows | Columns | Register |
|---|---|---|---|
| `accuracy_prompts.csv` | 245 | `prompt`, `poem_id` | tip of the tongue, over Fann or Flop |
| `diwan_prompts_final.csv` | 258 | `poem_id`, `prompt` | abstract description, over Diwan |
| `gemini_prompts.csv` | 200 | `id`, `prompt` | thematic topics, no target poem |

`accuracy_prompts.csv` contains 245 rows but only 244 distinct queries. One query is duplicated in
the source file. The pipeline removes it at load time, and every count in the paper is 244. We ship
the file as it was actually used rather than silently editing it.

## How the queries were written

All known-item queries were written by the authors from the poem text alone. Writers saw the verses
of the target poem and nothing else. They did not see the reference explanations distributed with
Fann or Flop, and they did not see the generated explanations, which did not exist until after the
queries were written.

The two known-item sets were written under deliberately different instructions. Fann or Flop queries
simulate someone who half remembers a poem and may include remembered wording, as a real tip of the
tongue searcher would. Diwan queries were written as abstract descriptions of what a poem is about,
with no quotation. That difference in instruction is the variable the paper manipulates, and it is
measurable: 32.8 percent of Fann or Flop queries share a verbatim three token span with their target
poem, against 2.7 percent for Diwan.

The 200 thematic queries name a topic only and were written before either corpus was inspected, so
they have no target poem and cannot have been shaped by the collections.

## Generated explanations

The `verses_ai_explanations` column in both corpora was produced with Gemini 2.0 Flash, one call per
verse, using a fixed instruction asking for a concise and faithful prose paraphrase of the verse in
Arabic and forbidding any information beyond the verse. The explanations were generated
independently of the queries, and no retriever sees a query at indexing time.

That model was withdrawn from the API during this work, so the generation step cannot be reproduced
with the model that produced it. The generated text itself is included here, which is why.

## Edited queries

`rebuttal/` holds the edited copies of the Fann or Flop queries used in section 5.2 of the paper:

| File | What it holds |
|---|---|
| `fof_counterfactual_queries.csv` | each query as released (`orig`), with the quoted tokens deleted (`del_span`), and with one verbatim half-verse of the target appended (`inj_open`, `inj_mid`, `inj_final`) |
| `fof_paraphrase_queries.csv`, `fof_paraphrase_queries_v2.csv` | LLM rewrites of the quoted words (two prompt versions) |
| `fof_hand_paraphrase_queries.csv` | hand rewrites of the quoted words, 79 of the 80 quoting queries |
