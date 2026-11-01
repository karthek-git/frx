# Rulebook: the concept

## The problem
Our AI drafts freight quotes from RFQ emails. Forwarders review every draft and often change a value or two
(trucking price, weight, margin, a surcharge). The rules behind those changes ("this customer gets 12% margin",
"trucking from Frankfurt always needs +40") are written down nowhere, they change without notice, and our engineers
aren't forwarders, so they can't tell what the AI is getting wrong or when something has gone stale.

## The idea
Every correction a forwarder makes is free expert feedback. Compare **what the AI proposed** with **what the human
finally sent**, and learn from the differences:

1. **Turn repeated corrections into written rules.** If reps make the same correction under the same condition
   again and again ("departure = FRA -> trucking +40, 31 times"), propose it as a candidate rule with its evidence.
   A human accepts or rejects each one. Accepted rules form a living rulebook: plain-language, useful as context
   for the AI and as onboarding material for new hires.

2. **Watch those rules for drift.** Keep checking each rule against recent behaviour. If the correction changes
   (+40 becomes +55) or stops happening, raise an alert with the estimated date it changed. This catches silent
   changes: a carrier raises prices, the rate sheet is stale, and reps quietly patch it every time.

3. **Spot new kinds of requests.** Group incoming RFQs by meaning. If new requests look unlike anything seen
   before (a new customer, a new format, a new unit such as weights in tonnes) and are being heavily edited, flag
   them before they become a pile of complaints.

4. **Catch groups of requests that start going wrong.** If one group of RFQs suddenly has far more edits that no
   known rule explains, flag it.

5. **Explain, don't just flag.** Each alert comes with the example requests, what the rep changed, the linked trace
   and session replay, any user bug reports, a likely cause and a suggested fix. The goal is that an engineer who
   isn't a forwarder can understand the problem in a minute.

Alerts can also be posted to Slack as a short message with a link back to the evidence.

## What kind of edit is it?
Not every edit is an AI mistake, and mixing them up gives wrong fixes. Each rule and alert is labelled:
- **AI mistake** (misread weight, missed dangerous-goods surcharge)
- **Business rule** (a customer's margin policy the AI doesn't know)
- **Data gap** (a missing or stale rate)
- **Integration issue** (e.g. a wrong value only for customers on one TMS)
One-off changes such as price negotiations should not become rules; they don't repeat.

## Data it needs (conceptually)
For each quote the AI drafted:
- the **RFQ email text** and a few attributes (customer, airport, destination, cargo type, TMS, unit used)
- the **AI's proposed values** and the **final values the rep sent**
- links to the **trace** (what the AI did internally) and the **session replay** (what the user did)
- optionally, the **text from the in-app bug-report button**

Time is split into a **baseline period** (what normal looks like) and a **recent window** (what is checked for
change). The usual sources are product analytics for user actions and replays, and tracing for the AI's internals,
joined by a shared quote or trace id.

## How it decides, in plain terms
- **Rules:** a correction counts as a rule when it happens often, under a clear condition, and with a consistent
  size. A bigger rule absorbs overlapping smaller ones.
- **Drift:** compare the typical correction now with the baseline; find the day where behaviour shifted.
- **New request types:** turn each email into a vector with a text-embedding model, cluster the baseline, and
  flag recent emails that are far from every cluster. The "too far" cut-off is learned from held-out normal data.
- **Group shifts:** a statistical test on the share of edits no rule explains.
Everything is deliberately simple and explainable, with no trained model beyond the embeddings.

## What the human does
The tool proposes; people decide. A forwarder or engineer accepts or rejects candidate rules, confirms drift, and
decides on fixes. A repeated correction is a strong hint, not proof.

## Why it's useful
- Engineers see where real users struggle without needing freight expertise.
- Unwritten knowledge becomes written, reviewable and usable by the AI.
- Problems surface when behaviour changes, not after customers complain.

## Caveats for integration
- Built and tested only on synthetic data with planted rules; thresholds need re-tuning on real data.
- Embedding emails means sending text to an embedding provider; get approval before using real customer data.
- It only covers quoting. The same approach fits other steps (invoice checks, export and import documents)
  wherever there is an "AI proposal" followed by a "human final decision".
- Cause and fix text is currently template-based; an LLM-written explanation per alert is the obvious next step.
