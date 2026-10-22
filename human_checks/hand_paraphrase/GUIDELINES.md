# Hand paraphrase: guidelines

In 80 of the Fann or Flop queries, the writer copied words straight from the poem, for example
"مطلعها أكثري أو فأقلي...". The reviewers asked whether BM25's advantage on these queries comes from
the exact wording of the memory or from its content. To test this, we need the same 80 queries with
the copied words replaced by words that mean the same thing, as if the person remembered what the
verse says but not how it says it. We will then re-run every retrieval system on your versions.

For each item:

1. Read the original query. The highlighted words are copied from the poem.
2. Read the poem under it, only to understand what the highlighted words mean.
3. In the box, which starts with the original query, rewrite only the highlighted part in different
   words with the same meaning. Leave everything else as it is.
4. Check the line under the box. It has to turn green ("OK"). If it still lists shared words, change
   them. Changing one letter or the word order is not enough; use different words.

Rules:

- Keep the same information need. Do not add details and do not drop any.
- Keep proper names (people, tribes, places, battles) exactly as they are. A name is information,
  not wording. If the only shared words left are a name, tick "proper name I kept on purpose".
- Write natural Arabic that someone could type as a search query.
- If the highlighted part cannot be paraphrased without changing what is being asked (for example,
  it is entirely a name or a title), tick "skip" and explain in the note.
- Do not look at retrieval results or at the LLM paraphrases while doing this.

There are 80 items; the whole task takes about an hour. Your work is saved in the browser. At the end, click
"Download CSV" and send the file `hand_paraphrase_answers.csv`.
