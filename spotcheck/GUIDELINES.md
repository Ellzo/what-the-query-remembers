# Spot-check guidelines: does the retrieved poem satisfy the request?

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
