# Nemox Lens

Mines the unwritten rules out of how forwarders correct AI-drafted quotes, and alerts when those rules drift
or new kinds of RFQs appear.

```
../.pyi/bin/pip install -r requirements.txt     # scikit-learn, numpy (once)
../.pyi/bin/python server.py                     # http://localhost:8000
../.pyi/bin/python gen.py                        # print stats for the synthetic data
```

### Embeddings
RFQ text is embedded with Google's embeddings API when a key is present, otherwise a local scikit-learn TF-IDF is used
(the dashboard's "RFQ groups" tab shows which one ran). The key is read from the `GOOGLE_API_KEY` environment
variable (`GEMINI_API_KEY` also works, and a gitignored `rulemine/.env` is an optional alternative):
```
export GOOGLE_API_KEY=your-key
export EMBED_MODEL=gemini-embedding-2     # optional; default is gemini-embedding-001
```
Vectors are cached in `.embed_cache.json`, so re-runs need no network. If the API call fails, the server falls back to
TF-IDF and says so. Note: with real data, this sends RFQ email text to Google; the demo data is synthetic.

## What it does
1. **Rule discovery**: repeated, consistent corrections (same field, same condition) become candidate rules
   (`departure = FRA -> trucking +40`). A human accepts or rejects each; accepted rules form a rulebook
   (`/api/rulebook.md`) usable as AI context or onboarding material.
2. **Drift detection**: each rule is watched in a recent window; a changed correction (+40 -> +55) or a rule that
   stops applying raises an alert, with an estimated change date.
3. **Novel inputs**: embeddings of the RFQ text; Local Outlier Factor against the baseline scores each recent RFQ
   (1.0 = typical), with a conformal cut-off from held-out baseline data (~1% of normal RFQs may exceed it).
   Flagged RFQs are grouped with HDBSCAN and investigated (here: a customer sending weights in metric tonnes, and
   live-animal requests the AI has no knowledge of). No hand-picked cluster count.
4. **Cluster shift**: per-group rate of edits no rule explains, z-tested vs baseline.
5. **Cause labels**: AI mistake / business rule / data gap / integration issue (e.g. currency wrong only on one TMS).
   A fixable error in a new kind of RFQ (a consistent correction, e.g. tonnes read as kg, x1000) also produces a candidate customer
   rule, with a "Create customer rule" button on the alert. A capability gap (varied rewrites, no consistent fix) does not.
6. **Formula rules**: some corrections are a calculation, not a constant. The engine tests candidate weight bases against
   final prices (`price = rate x chargeable weight`, where chargeable weight is the larger of actual and volume weight,
   L x W x H / 6000 per piece) and reports "price on chargeable weight, not actual weight" with how many edits it explains.
7. **Unreported cases** (the "Unreported cases" tab): edited quotes are clustered by journey (email embedding plus what the rep
   changed). A group containing a bug report about the same field, but mostly unreported quotes, sizes the reported problem.
   These are look-alikes: leads to check, not proof of a problem.
8. **Slack**: every alert renders as a Slack message. It is only sent if `SLACK_WEBHOOK_URL` is set and
   "Send to Slack" is clicked; otherwise it is preview-only.

## Data
`gen.py` produces synthetic events shaped like PostHog events joined to Logfire traces
(`trace_id`, `replay_id` are placeholders). Ground truth planted: 6 rules (one is a chargeable-weight formula), 1 drift (day 45), 2 novel input types
(day 50+: a customer sending weights in tonnes, a fixable extraction error; day 52+: live-animal RFQs, a capability gap the AI
has no knowledge of, where reps rewrite 3-4 values per quote with no consistent pattern), ~5% negotiation noise. Swap `gen.make_events()` for a PostHog/Logfire loader to use real data.

## Honest limits
- With the local TF-IDF fallback the two new-input groups merge into one alert; with Google embeddings they separate.
- Synthetic data: results show the method works, not that it works on nemox's data.
- A repeated correction is a hypothesis, not a proven rule; that is why a human accepts each one.
- Baseline unedited rate in the demo is ~31%, lower than nemox's reported ~52%.
- Quote-only for now; the core (`engine.py`) is field/attribute driven and would take other modules.
