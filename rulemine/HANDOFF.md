# Rulebook: handoff for integration

## Paste this into Claude Code (in your own repo)

> I'm integrating a hackathon prototype called "Rulebook" into our codebase. Its source is in `<path-to>/rulemine/`;
> read `rulemine/HANDOFF.md` first, then `engine.py`, `embed.py`, `gen.py`, `server.py`.
> The core is `engine.analyse(events, split, embedder)`: it takes quote events (AI proposal vs. final human-sent
> values) and returns candidate rules, drift/novelty/cluster-shift alerts and RFQ clusters.
> Your job: (1) replace `gen.make_events()` with a loader that builds the same event dicts from our real
> PostHog events joined to Logfire traces (see "Event contract"); (2) keep `engine.py` unchanged unless a real
> field/attribute needs adding to `FIELDS`/`ATTRS`/`FIELD_META`; (3) mount the API and dashboard in our app or run
> `server.py` as is; (4) run it on real data and report which thresholds in "Tuning" need changing.
> Do not commit API keys; read them from the environment. Ask before sending real customer email text to any
> external service (the embedding step does that).

## What it is

A devtool for engineers who aren't domain experts. Forwarders correct the AI's drafted quotes every day; those
corrections encode unwritten business rules. Rulebook:

1. **Discovers rules** from repeated, consistent corrections (e.g. `departure = FRA -> trucking +40`). A human
   accepts or rejects each; accepted rules form a rulebook (`/api/rulebook.md`) usable as AI context / onboarding.
2. **Detects drift**: a rule's correction changes (+40 -> +55) or stops applying, with an estimated change date.
3. **Detects novel inputs**: recent RFQs unlike anything in the baseline, grouped and investigated.
4. **Detects cluster shifts**: a group of RFQs whose "unexplained" edit rate jumps.
5. **Labels causes**: AI mistake / business rule / data gap / integration issue, plus a suggested fix.
6. **Renders Slack messages** per alert (sends only if `SLACK_WEBHOOK_URL` is set and the button is clicked).

Context: built for the nemox (freight-forwarding AI) hackathon, DevTools track. nemox's product drafts quotes inside
Outlook/Gmail and syncs with TMSs; reps edit about half the drafts, usually 1-2 values (trucking price, volume).

## Files (`rulemine/`)

| File | Role |
|---|---|
| `engine.py` | All analysis. Entry point `analyse(events, split, embedder)`. Pure functions, no I/O. |
| `embed.py` | Embedders: `GeminiEmbedder` (Google API, disk cache) and `TfidfEmbedder` (local fallback). `python embed.py` warms the cache. |
| `gen.py` | Synthetic data with planted rules, one drift, two novel input types (tonnes; live animals). Replace with real loader. |
| `server.py` | stdlib HTTP server: JSON API + static dashboard. Runs analysis once at import. |
| `slack.py` | Slack message rendering and optional send. |
| `web/index.html` | Single-file dashboard (Material Web components from CDN, vanilla JS, collapsible sidebar). |
| `requirements.txt` | `scikit-learn`, `numpy`. Everything else is stdlib. |

## Event contract (what a real loader must produce)

One dict per quote, `analyse` mutates them (adds `edits`, and `_sim` on recent ones):

```python
{
  "id": "q00001",                 # unique
  "day": 12,                      # integer day index; baseline = day < split, monitor window = day >= split
  "customer": "Alpina Tools", "tms": "CargoWise", "lang": "en",
  "departure": "FRA", "dest": "SHA", "cargo": "lithium",   # categorical attributes (see ATTRS in engine.py)
  "unit": "lbs",                  # unit used in the email
  "email": "full RFQ text ...",   # what gets embedded
  "proposal": {"margin_pct": 8.0, "trucking_eur": 120.0, "weight_kg": 800.0,
               "dg_surcharge_eur": 0.0, "currency": "EUR",
               "special_handling_eur": 0.0, "routing": "standard", "freight_price_eur": 221.0},   # what the AI drafted
  "final":    { ...same keys... },                              # what the rep actually sent
  "trace_id": "...", "replay_id": "...",                       # Logfire trace / PostHog replay links
  "bug_report": None, "bug_field": None,
  # for formula rules (price = rate x chargeable weight): derived inputs of the quote
  "bulk": "bulky",                                  # "bulky" if volume weight > actual weight, else "standard"
  "actual_kg": 92.0, "volume_kg": 412.4, "rate_per_kg": 2.4,                       # optional: text from the in-app bug button + field it concerns
}
```

Mapping to real systems (assumptions to verify): `proposal` = AI draft values logged in the Logfire trace;
`final` = values at send time (PostHog event or TMS write); `bug_report` = the plugin's bug-button event. The join
key between PostHog and Logfire (trace id or quote id) must exist as an event property; if not, add it in code.
`FIELDS`, `ATTRS`, `FIELD_META` in `engine.py` are the only quote-specific parts. Other modules (Accounting,
Export, Import) would be new field/attr sets, same engine.

## Algorithms

- **Rule discovery** (`discover_rules`): for each field x single attribute value, take matching quotes; keep if
  >= 8 edited, >= 60% edit rate, and the correction is consistent (median absolute deviation / median <= 0.12) as
  additive, multiplicative, or categorical "set". Greedy de-duplication drops sub-rules covered >50% by a larger rule.
- **Drift** (`detect_drift`): median recent correction vs baseline; alert if > 20% change (>= 5 edits) or edit rate
  < 30%. Change date via two-segment changepoint (minimise absolute deviation from each side's median).
- **Novelty** (`cluster_and_novelty`): embed emails; `LocalOutlierFactor(n_neighbors=20, cosine, novelty=True)` fit on the
  baseline scores each recent RFQ (1.0 = typical, higher = more isolated). The cut-off is a conformal quantile (alpha=0.01) of
  scores on a held-out 30% of the baseline, so about 1% of normal RFQs would exceed it. No cluster count to choose.
- **Promoting a fixable error to a rule** (`_candidate_rule`): if a flagged group has a consistent correction on one field (same
  dispersion test as rule discovery), the alert carries a candidate customer rule built from all of that customer's quotes. Groups
  with varied rewrites (capability gaps) get none.
- **Grouping flagged RFQs**: `HDBSCAN(min_cluster_size=5)` over the flagged points; a group >= 5 raises an alert.
- **Baseline map** ("RFQ groups" tab): `HDBSCAN(min_cluster_size=40)` on baseline embeddings finds its own number of groups;
  recent RFQs are assigned to the nearest group centre. Group labels = top TF-IDF terms (separate from the embedding).
- **Formula rules** (`discover_formula_rules`): for a condition with a high edit rate on `freight_price_eur`, test whether the final
  price / rate equals the larger of actual and volume weight (within 2%, on >= 85% of edits). Reported as a rule with the AI's basis
  (actual weight) vs the reps' basis. Needs the derived inputs above.
- **Unreported cases** (`find_unreported`): edited quotes are clustered with HDBSCAN on [email embedding scaled to the edit
  signature's range, edit signature (per field: edited flag + clipped log ratio)]. Clusters with the same changed fields and top
  customer are merged. A group counts a bug report only if it is about a field the group corrects. Output per group: quotes, reported,
  never reported, example ids.
- **Cluster shift**: two-proportion z-test on "unexplained" edit rate (edits no discovered rule covers); alert z > 3.

## Run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
export GEMINI_API_KEY=...        # or GOOGLE_API_KEY; optional. Without it: local TF-IDF embeddings
.venv/bin/python embed.py        # optional: warm the embedding cache (free tier ~100 texts/min, resumes)
.venv/bin/python server.py       # http://localhost:8000   (optional arg: port)
```

Env vars: `GEMINI_API_KEY` then `GOOGLE_API_KEY` (second is tried if the first is rejected), `EMBED_MODEL`
(default `gemini-embedding-001`; `gemini-embedding-2` supported via a text prefix, not live-tested),
`SLACK_WEBHOOK_URL`. A gitignored `rulemine/.env` is also read but never overrides real env vars.

API: `GET /api/analysis`, `GET /api/events/<id,id>`, `GET /api/rulebook.md`, `GET /api/slack/<alert id>`,
`POST /api/decision {id, decision}`, `POST /api/slack/send {id}`. Accept/reject state is in `decisions.json`.

## Status: what was verified

- On synthetic data the engine recovers all 5 planted rules and raises exactly the 2 planted alerts (FRA trucking
  drift dated to day 45; a new customer sending weights in metric tonnes, all x1000), with 0 false alerts, using both
  TF-IDF and live Google embeddings (all 29 new-input RFQs below the cut-off, 0 of 207 normal ones).
- Gemini request format, batching, caching, key rotation and error handling were tested against a fake API and
  live for `gemini-embedding-001`.

## Not verified / known limits

- **Dashboard never checked in a browser** (JS syntax only). Expect small UI bugs.
- Real Slack send untested. Trace/replay ids in the demo are placeholders, not working links.
- All thresholds (0.12 dispersion, 20% drift, z > 3, LOF k=20, alpha=0.01, HDBSCAN sizes) were tuned on synthetic data.
- Rules are single-attribute conditions only; no interactions (e.g. customer x lane).
- Demo unedited rate is ~31% vs nemox's published ~52%.
- A repeated correction is a hypothesis, not proof of a rule; the human accept step is intentional.
- Cause/fix text is template-based (no LLM generation yet). The natural next step is an LLM summary per alert.
- Embedding RFQs sends email text to Google. Fine for synthetic data; needs sign-off for real customer data.
- With real embeddings, cluster labels come out as route words ("shanghai, air, cm"); consider LLM-written labels.

## Suggested integration order

1. Write `load_events()` from PostHog + Logfire in the contract above; keep `gen.py` for tests.
2. Run `analyse` on a real window; inspect discovered rules for sanity before trusting alerts.
3. Re-tune thresholds on real data; decide how `day`/`split` map to real time (e.g. rolling 30-day baseline).
4. Wire accept/reject into wherever the team keeps AI context; feed accepted rules back as prompt context.
5. Slack: use a private test channel first.
