# Paraphrase check: guidelines

For each Fann or Flop query that copied words from its poem, an LLM rewrote only the highlighted,
copied words and was told to keep everything else. We checked automatically that no rewrite shares
three consecutive words with the poem. Retrieval with the rewritten queries tests whether it is the
exact wording of a memory, rather than its content, that BM25 depends on. That test only works if
the rewrite asks for the same poem with the same details.

Same information need? Imagine one person typing each version. 2 means they are looking for the
same thing with the same details. 1 means mostly, but a detail was lost, added or blurred. 0 means
the rewrite asks for something different or is much vaguer.

Proper names (people, tribes, places) are meant to stay unchanged, because a name is information
the person remembers, not wording. A kept name is not copied wording. If a rewrite changed or
garbled a name, give "same need" a 0 or 1.

Is the copied wording still there? Is the rewrite still quoting the verse in practice, for example by
changing a single letter or only the word order? 0 means no, it is a real paraphrase.

Natural Arabic? Could a person plausibly type it as a search query?

Most queries appear twice, with two different rewrites, in different places in the list. Rate each
one on its own. Do not look up the poems; judge from the two texts only. There are about 150 short
items; the whole task takes about 20 minutes.
