"""Rule discovery, drift detection and novelty detection over (proposal -> final) quote events.

Module-agnostic core: an event is {proposal, final, attrs...}. Nothing below is quote-specific
except FIELD_META (which maps a field to a human label / cause class).
Standard library only.
"""
import math
import statistics as st
from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

NUMERIC = ["margin_pct", "trucking_eur", "weight_kg", "dg_surcharge_eur", "special_handling_eur", "freight_price_eur"]
CATEG = ["currency", "routing"]
FIELDS = NUMERIC + CATEG
ATTRS = ["customer", "departure", "dest", "cargo", "unit", "tms", "lang", "bulk"]

MIN_EDITED = 8          # evidence needed before proposing a rule
MIN_RATE = 0.6          # share of matching quotes that get this field edited
MAX_DISP = 0.12         # how consistent the correction must be
LOF_NEIGHBORS = 20      # Local Outlier Factor: how many nearest baseline RFQs define "normal" around a point
ALPHA = 0.01            # conformal level: ~1% of normal RFQs may exceed the novelty cut-off
HDB_MIN_CLUSTER = 40    # smallest RFQ group worth naming in the baseline map

# field -> (human label, cause class)
FIELD_META = {
    "margin_pct": ("customer margin", "business rule"),
    "trucking_eur": ("trucking / pre-carriage price", "data gap"),
    "weight_kg": ("weight extraction", "AI mistake"),
    "dg_surcharge_eur": ("dangerous-goods surcharge", "AI mistake"),
    "currency": ("currency", "integration issue"),
    "special_handling_eur": ("special handling charge", "data gap"),
    "routing": ("routing", "business rule"),
    "freight_price_eur": ("freight price", "AI mistake"),
}


# ---------------------------------------------------------------- diffs
def annotate(events):
    for ev in events:
        ed = {}
        for f in FIELDS:
            p, q = ev["proposal"][f], ev["final"][f]
            if p != q:
                ed[f] = {"from": p, "to": q}
        ev["edits"] = ed
    return events


def _matches(ev, attr, value):
    return ev[attr] == value


def _corr_value(ev, field, kind):
    e = ev["edits"][field]
    if kind == "add":
        return e["to"] - e["from"]
    if kind == "mul":
        return e["to"] / e["from"] if e["from"] else None
    return e["to"]


def _disp(vals):
    m = st.median(vals)
    mad = st.median([abs(v - m) for v in vals])
    return (mad / abs(m) if m else (0.0 if mad == 0 else float("inf"))), m


def _fmt(v):
    if isinstance(v, str):
        return v
    return f"{v:+.4g}" if abs(v) < 1e6 else f"{v:.3g}"


def _describe(rule):
    f, k, c = rule["field"], rule["kind"], rule["correction"]
    cond = f"{rule['attr']} = {rule['value']}"
    label = FIELD_META[f][0]
    if k == "add":
        return f"When {cond}: correct {label} by {c:+.4g}"
    if k == "mul":
        return f"When {cond}: multiply {label} by {c:.4g}"
    return f"When {cond}: set {label} to {c}"


def _cause(rule):
    """Heuristic classification of why reps make this edit."""
    f = rule["field"]
    label, cls = FIELD_META[f]
    if f == "weight_kg":
        c = rule["correction"]
        if abs(c - 0.4536) < 0.02:
            return "AI mistake", "Pounds treated as kilograms (unit conversion not applied)."
        if abs(c - 1000) < 50:
            return "AI mistake", "Weights given in metric tonnes are read as kilograms (1.873 t parsed as 1.873 kg)."
        return "AI mistake", "Weight is systematically misread from the email or attachment."
    if f == "currency":
        return "integration issue", (
            f"Currency differs only for TMS '{rule['value']}'; likely an adapter mapping problem, not a model error."
            if rule["attr"] == "tms" else "Currency is corrected consistently; check the master-data mapping.")
    if f == "dg_surcharge_eur":
        return "AI mistake", "Special cargo is not recognised, so the dangerous-goods surcharge is never added."
    if f == "trucking_eur":
        return "data gap", "Trucking rate is missing a recurring component; reps add it by hand every time."
    if f == "margin_pct":
        return "business rule", "Customer-specific margin policy that exists only in the reps' heads."
    return cls, ""


def _fix(rule):
    f = rule["field"]
    cond = f"{rule['attr']} = {rule['value']}"
    if f == "weight_kg":
        return f"Add a unit-normalisation step before pricing for {cond}, and add regression cases from the linked traces."
    if f == "currency":
        return f"Inspect the TMS adapter's currency mapping for {cond}; add an integration test."
    if f == "dg_surcharge_eur":
        return f"Add cargo-class detection (UN numbers / 'lithium') for {cond} and apply the surcharge automatically."
    if f == "trucking_eur":
        return f"Update the rate sheet or add a surcharge rule for {cond}."
    if f == "margin_pct":
        return f"Store the margin for {cond} in the customer profile so the AI applies it."
    return "Review and codify this correction."


# ---------------------------------------------------------------- rule discovery
def discover_rules(events):
    """Propose rules from repeated, consistent corrections in `events` (the baseline window)."""
    cands = []
    for f in FIELDS:
        for attr in ATTRS:
            groups = defaultdict(list)
            for ev in events:
                groups[ev[attr]].append(ev)
            for value, S in groups.items():
                if len(S) < 10:
                    continue
                ed = [e for e in S if f in e["edits"]]
                rate = len(ed) / len(S)
                if len(ed) < MIN_EDITED or rate < MIN_RATE:
                    continue
                if f in CATEG:
                    c = Counter(e["edits"][f]["to"] for e in ed)
                    top, cnt = c.most_common(1)[0]
                    if cnt / len(ed) < 0.8:
                        continue
                    kind, corr, disp = "set", top, 0.0
                else:
                    options = []
                    for kind_ in ("add", "mul"):
                        vals = [v for v in (_corr_value(e, f, kind_) for e in ed) if v is not None]
                        if len(vals) >= MIN_EDITED:
                            d, m = _disp(vals)
                            options.append((d, 0 if kind_ == "add" else 1, kind_, m))
                    if not options:
                        continue
                    disp, _, kind, corr = min(options)
                    if disp > MAX_DISP:
                        continue
                out = [e for e in events if e[attr] != value]
                out_rate = sum(1 for e in out if f in e["edits"]) / max(1, len(out))
                cands.append({
                    "id": f"{f}|{attr}={value}", "field": f, "attr": attr, "value": value,
                    "kind": kind, "correction": corr, "dispersion": round(disp, 4),
                    "n": len(S), "n_edited": len(ed), "rate": round(rate, 3),
                    "outside_rate": round(out_rate, 3),
                    "event_ids": [e["id"] for e in ed],
                })
    # greedy de-duplication: prefer the rule explaining the most edits; drop overlapping sub-rules
    cands.sort(key=lambda r: (-r["n_edited"], -r["rate"]))
    chosen = []
    for r in cands:
        s = set(r["event_ids"])
        dup = False
        for c in chosen:
            if c["field"] == r["field"] and len(s & set(c["event_ids"])) / len(s) > 0.5:
                dup = True
                break
        if not dup:
            chosen.append(r)
    byid = {e["id"]: e for e in events}
    for r in chosen:
        r["statement"] = _describe(r)
        r["cause_class"], r["cause"] = _cause(r)
        r["fix"] = _fix(r)
        r["label"] = FIELD_META[r["field"]][0]
        if r["kind"] != "set":
            r["from_med"] = st.median(byid[i]["edits"][r["field"]]["from"] for i in r["event_ids"])
            r["to_med"] = st.median(byid[i]["edits"][r["field"]]["to"] for i in r["event_ids"])
        r["action"] = _recommend_rule(r)
    return chosen


# ---------------------------------------------------------------- series / drift
def _series(rule, events, bucket=5):
    if rule["kind"] == "formula":
        return []
    pts = defaultdict(list)
    for e in events:
        if _matches(e, rule["attr"], rule["value"]) and rule["field"] in e["edits"]:
            v = _corr_value(e, rule["field"], rule["kind"])
            if v is not None and not isinstance(v, str):
                pts[e["day"] // bucket].append(v)
    return [{"day": b * bucket, "value": round(st.median(v), 4), "n": len(v)} for b, v in sorted(pts.items())]


def _changepoint(rule, events):
    xs = sorted((e["day"], _corr_value(e, rule["field"], rule["kind"]))
                for e in events if _matches(e, rule["attr"], rule["value"]) and rule["field"] in e["edits"])
    xs = [(d, v) for d, v in xs if v is not None and not isinstance(v, str)]
    best = None
    for k in range(4, len(xs) - 3):
        l, r = [v for _, v in xs[:k]], [v for _, v in xs[k:]]
        ml, mr = st.median(l), st.median(r)
        # two-segment fit: minimise absolute deviation from each segment's median
        cost = sum(abs(v - ml) for v in l) + sum(abs(v - mr) for v in r)
        if best is None or cost < best[0]:
            best = (cost, xs[k][0], ml, mr)
    return best


def detect_drift(rules, events, split):
    alerts = []
    mon = [e for e in events if e["day"] >= split]
    for r in rules:
        M = [e for e in mon if _matches(e, r["attr"], r["value"])]
        if len(M) < 6:
            continue
        ed = [e for e in M if r["field"] in e["edits"]]
        rate = len(ed) / len(M)
        r["monitor"] = {"n": len(M), "n_edited": len(ed), "rate": round(rate, 3)}
        if r["kind"] in ("set", "formula"):
            continue
        drifted = None
        if len(ed) >= 5:
            vals = [v for v in (_corr_value(e, r["field"], r["kind"]) for e in ed) if v is not None]
            now = st.median(vals)
            rel = abs(now - r["correction"]) / max(abs(r["correction"]), 1e-9)
            if rel > 0.2:
                drifted = ("shift", now)
        if drifted is None and rate < 0.3 and r["rate"] > 0.6:
            drifted = ("stopped", None)
        if drifted:
            r["drift"] = True
            cp = _changepoint(r, events)
            since = cp[1] if cp else split
            alerts.append(_drift_alert(r, drifted, since, ed))
    return alerts


def _drift_alert(r, drifted, since, ed):
    kind, now = drifted
    label = r["label"]
    cond = f"{r['attr']} = {r['value']}"
    if kind == "shift":
        title = f"Rule drift: {label} when {cond}"
        old, new = r["correction"], now
        summary = (f"Reps used to correct this by {_fmt(old)}; since about day {since} they correct it by "
                   f"{_fmt(new)} ({len(ed)} recent edits, {int(r['monitor']['rate'] * 100)}% of matching quotes).")
        hyp = ("The underlying cost changed (carrier or trucker price update) and the rate sheet / rule the AI uses "
               "is stale. Reps are silently patching the difference.")
        fix = f"Update the source rate for {cond} to include the new component ({_fmt(new)}); re-baseline this rule."
    else:
        title = f"Rule no longer applies: {label} when {cond}"
        summary = (f"This correction used to happen in {int(r['rate'] * 100)}% of matching quotes; "
                   f"recently only {int(r['monitor']['rate'] * 100)}%.")
        hyp = "Either the AI now handles this case, or the business rule changed. Confirm before deleting the rule."
        fix = "Review the rule with a forwarder; retire it if it no longer reflects practice."
    return {
        "id": f"drift|{r['id']}", "type": "rule_drift", "severity": "high",
        "title": title, "summary": summary, "hypothesis": hyp, "fix": fix,
        "rule_id": r["id"], "since_day": since, "field": r["field"],
        "event_ids": [e["id"] for e in ed][-8:],
        "series": r.get("series", []),
        "action": _rec(
            "confirm_with_user", "Confirm with the account owner",
            "Developer, then the account owner or a forwarder",
            (f"Reps now correct {label} for {cond} by {_fmt(now)} instead of {_fmt(r['correction'])} "
             f"(since about day {since}). Is this a real change, such as a new carrier or trucker price? "
             "If yes, we will update the rule and the rate source.") if kind == "shift" else
            (f"The usual correction of {label} for {cond} has nearly stopped. Did the AI start handling it, "
             "or did the business rule change? Please confirm before we retire the rule."),
            "fact_check"),
    }


# ---------------------------------------------------------------- text clustering / novelty
def _label_terms(texts, assign, ids, n=5):
    """Human-readable cluster labels: top TF-IDF terms of each cluster's emails (independent of the embedder)."""
    vec = TfidfVectorizer(token_pattern=r"[A-Za-z][A-Za-z]+", stop_words="english", sublinear_tf=True)
    X = vec.fit_transform(texts)
    terms = np.array(vec.get_feature_names_out())
    out = []
    for j in ids:
        idx = np.where(assign == j)[0]
        if len(idx) == 0:
            out.append([])
            continue
        m = np.asarray(X[idx].mean(axis=0)).ravel()
        out.append([str(t) for t in terms[np.argsort(-m)[:n]]])
    return out


def _unit(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def cluster_and_novelty(events, split, rules, embedder):
    base = [e for e in events if e["day"] < split]
    mon = [e for e in events if e["day"] >= split]
    embedder.fit([e["email"] for e in base])
    Xb = embedder.transform([e["email"] for e in base])
    Xm = embedder.transform([e["email"] for e in mon])

    # --- novelty detection: Local Outlier Factor with a conformal cut-off -------------------------------------
    # LOF scores a point by how isolated it is compared with its nearest baseline neighbours (1.0 = typical).
    # The cut-off is the conformal quantile of scores on held-out baseline RFQs: under exchangeability, at most
    # ALPHA of normal future RFQs exceed it. No cluster count, no hand-tuned distance.
    def lof(X):
        return LocalOutlierFactor(n_neighbors=LOF_NEIGHBORS, novelty=True, metric="cosine").fit(X)

    perm = np.random.RandomState(0).permutation(len(base))
    cut = int(len(base) * 0.7)
    held = np.sort(-lof(Xb[perm[:cut]]).score_samples(Xb[perm[cut:]]))
    k = min(len(held) - 1, int(math.ceil((len(held) + 1) * (1 - ALPHA))) - 1)
    threshold = float(held[k])
    mon_score = -lof(Xb).score_samples(Xm)

    # --- descriptive map of normal RFQs: HDBSCAN (finds its own number of groups; outliers become noise) ----
    labels = HDBSCAN(min_cluster_size=HDB_MIN_CLUSTER, min_samples=5, copy=True).fit_predict(Xb)
    ids = sorted(set(labels.tolist()) - {-1})
    if not ids:                       # nothing dense enough: treat the baseline as one group
        labels, ids = np.zeros(len(base), dtype=int), [0]
    cents = _unit(np.vstack([Xb[labels == j].mean(axis=0) for j in ids]))
    mon_assign = (Xm @ cents.T).argmax(axis=1)
    terms = _label_terms([e["email"] for e in base], labels, ids)

    def covered(e, field):
        return any(r["field"] == field and _matches(e, r["attr"], r["value"]) for r in rules)

    def unexplained(e):
        return any(not covered(e, f) for f in e["edits"])

    clusters = [{"id": int(j), "terms": terms[n],
                 "base": [base[i] for i in range(len(base)) if labels[i] == j], "mon": []}
                for n, j in enumerate(ids)]
    edited_base = [len(e["edits"]) for e in base if e["edits"]]
    base_avg = sum(edited_base) / max(1, len(edited_base))
    novel_idx = []
    for i, e in enumerate(mon):
        e["_score"] = float(mon_score[i])
        if mon_score[i] > threshold:
            novel_idx.append(i)
        else:
            clusters[int(mon_assign[i])]["mon"].append(e)

    alerts = []
    # cluster-level shift in *unexplained* edit rate (two-proportion z-test)
    out = []
    for c in clusters:
        B, M = c["base"], c["mon"]
        bu = sum(1 for e in B if unexplained(e))
        mu = sum(1 for e in M if unexplained(e))
        b_rate = bu / max(1, len(B))
        m_rate = mu / max(1, len(M))
        z = 0.0
        if len(M) >= 10 and len(B) >= 10:
            p = (bu + mu) / (len(B) + len(M))
            se = math.sqrt(max(p * (1 - p), 1e-9) * (1 / len(B) + 1 / len(M)))
            z = (m_rate - b_rate) / se
        out.append({
            "id": c["id"], "terms": c["terms"], "n_base": len(B), "n_mon": len(M),
            "unedited_base": round(sum(1 for e in B if not e["edits"]) / max(1, len(B)), 3),
            "unedited_mon": round(sum(1 for e in M if not e["edits"]) / max(1, len(M)), 3),
            "unexplained_base": round(b_rate, 3), "unexplained_mon": round(m_rate, 3), "z": round(z, 2),
        })
        if z > 3.0:
            alerts.append({
                "id": f"shift|{c['id']}", "type": "cluster_shift", "severity": "medium",
                "title": f"Edit rate jump in RFQ group '{' '.join(c['terms'][:3])}'",
                "summary": f"Unexplained edits rose from {b_rate:.0%} to {m_rate:.0%} (z={z:.1f}) in this group of RFQs.",
                "hypothesis": "Something changed for this kind of request that no known rule explains.",
                "fix": "Inspect the sampled quotes and traces; consider running rule discovery on this group alone.",
                "action": _rec("investigate", "Investigate this RFQ group", "Developer (via Slack)",
                               f"Edits nobody has a rule for rose from {b_rate:.0%} to {m_rate:.0%} in this group of RFQs. "
                               "Open the linked traces and replays, and ask a forwarder what changed.", "search"),
                "event_ids": [e["id"] for e in M if unexplained(e)][:8],
            })

    # novel inputs: group the flagged RFQs by density (HDBSCAN; isolated points are ignored)
    groups = defaultdict(list)
    if len(novel_idx) >= 5:
        lab = HDBSCAN(min_cluster_size=5, min_samples=3, copy=True).fit_predict(Xm[novel_idx])
        for i, l in zip(novel_idx, lab):
            if l >= 0:
                groups[l].append(mon[i])
    for E in groups.values():
        if len(E) < 5:
            continue
        edited = [e for e in E if e["edits"]]
        fcount = Counter(f for e in edited for f in e["edits"])
        field = fcount.most_common(1)[0][0] if fcount else None
        finding, consistent = "", None
        if field and field in NUMERIC:
            for kind in ("mul", "add"):
                vals = [v for v in (_corr_value(e, field, kind) for e in edited if field in e["edits"]) if v is not None]
                if len(vals) >= 5:
                    d, m = _disp(vals)
                    if d <= MAX_DISP:
                        finding = (f"{len(vals)} edits to {FIELD_META[field][0]}, always "
                                   f"{'x' + format(m, '.4g') if kind == 'mul' else format(m, '+.4g')}.")
                        consistent = (field, kind, m, d)
                        break
        custs = Counter(e["customer"] for e in E).most_common(2)
        n_fields = len({f for e in edited for f in e["edits"]})
        avg_edits = sum(len(e["edits"]) for e in edited) / max(1, len(edited))
        live, tonnes = is_live_animals(E), uses_tonnes(E)
        if not finding and edited:
            finding = (f"Reps rewrote {len(edited)} of them by hand: {avg_edits:.1f} values changed per quote across "
                       f"{n_fields} fields (normal quotes: {base_avg:.1f}), with no consistent pattern.")
        title = f"New kind of RFQ not seen before ({len(E)} requests, e.g. {custs[0][0]})"
        hyp = "Inputs of a new type reach the model with no matching rule or test."
        fix = "Add examples of this input type to the eval set."
        action = _rec("investigate", "Investigate this new input type", "Developer (via Slack)",
                      f"{len(E)} RFQs look unlike anything seen before and {len(edited)} were corrected by hand. "
                      "Open the linked traces and ask a forwarder what is different.", "search")
        if tonnes:
            hyp = ("A new customer sends weights in metric tonnes ('1.873 t') in a consignment-note style the "
                   "extraction step has not seen; the number is read as kilograms.")
            fix = ("Normalise units in extraction (t, kg, lbs) and add regression cases for this customer's format "
                   "from the linked traces.")
            action = _rec("onboard_customer", "Set up the new customer / format",
                          "Developer + customer success (via Slack)",
                          f"{custs[0][0]} sends RFQs in a format the AI has not seen, and {len(edited)} of {len(E)} were "
                          "corrected by hand. Add this format to the test set, fix the extraction, and create a customer "
                          "profile so the next requests come out right first time.", "person_search")
        if live:
            title = f"Capability gap: live-animal shipments ({len(E)} RFQs the AI has never seen)"
            hyp = ("Live-animal transport needs specialist knowledge the AI does not have: which airlines accept animals, "
                   "direct-routing and embargo constraints, crate and health-certificate requirements, and handling fees. "
                   "The AI quotes it like ordinary cargo and reps rewrite the price and routing by hand.")
            fix = ("Do not patch this with a rule. Capture a live-animals playbook with an experienced forwarder "
                   "(acceptance by airline, routing constraints, required documents, surcharge bands). Until then, warn "
                   "the rep and route these RFQs to a specialist.")
            action = _rec("capability_gap", "Escalate: build a live-animals playbook",
                          "Developer + a forwarder with live-animal experience",
                          f"{len(E)} RFQs ask for live-animal transport, which the AI has no knowledge of; reps rewrote "
                          f"{len(edited)} of them. Two steps: (1) show reps a banner in the plugin, 'Unusual request: the AI "
                          "has no experience with this, review everything', triggered by this detector; (2) sit with a "
                          "forwarder to write the playbook (airline acceptance, direct-only routing, documents, handling fees).",
                          "pets")
        cand = _candidate_rule(events, custs[0][0], consistent) if consistent else None
        alerts.append({
            "id": f"novel|{len(alerts)}", "type": "novel_inputs", "severity": "high", "title": title,
            "summary": (f"{len(E)} recent RFQs are unlike anything in the baseline (novelty score "
                        f"{min(e['_score'] for e in E):.1f}+ where 1.0 is typical; cut-off {threshold:.2f}); "
                        f"{len(edited)} of them were edited. " + (finding or "")),
            "hypothesis": hyp, "fix": fix, "action": action,
            "event_ids": [e["id"] for e in E][:8], "novel_count": len(E), "field": field,
            "candidate_rule": cand,
        })
    return out, alerts, {"backend": embedder.name, "threshold": round(threshold, 2), "clusters": len(ids),
                         "method": f"LOF(k={LOF_NEIGHBORS}) + conformal cut-off (alpha={ALPHA}); HDBSCAN"}


def is_live_animals(E):
    return sum(1 for e in E if "live animal" in e["email"]) > len(E) / 2


def uses_tonnes(E):
    return sum(1 for e in E if "metric tonnes" in e["email"]) > len(E) / 2


# ---------------------------------------------------------------- recommended actions + profiles
def _rec(kind, label, audience, message, icon):
    return {"kind": kind, "label": label, "audience": audience, "message": message, "icon": icon}


def _recommend_rule(r):
    """Route a discovered rule to the person who can actually remove the need for the manual edit."""
    cls, attr, val, f = r["cause_class"], r["attr"], r["value"], r["field"]
    n, N, pct, label = r["n_edited"], r["n"], int(r["rate"] * 100), r["label"]
    if cls == "business rule" and attr == "customer" and "to_med" in r:
        to, fr = r["to_med"], r["from_med"]
        unit = "%" if f == "margin_pct" else ""
        return _rec("add_to_profile", "Add to customer profile", "Rep, as a prompt in the email plugin",
                    f"You changed the {label} for {val} on {n} of the last {N} quotes ({pct}%), always to {to:g}{unit} "
                    f"(the AI proposed {fr:g}{unit}). Save {to:g}{unit} as the default {label} for {val} so it is right next time?",
                    "person_add")
    if cls in ("data gap", "business rule"):
        what = r["statement"].split(": ", 1)[1]
        return _rec("add_to_rate_sheet", "Add to rate sheet", "Operations (rate-sheet owner)",
                    f"On {pct}% of quotes where {attr} = {val}, reps make the same adjustment ({what}). "
                    "Add it as a standing charge so quotes come out right first time?", "table_chart")
    if cls == "integration issue":
        return _rec("check_integration", "Check the TMS adapter", "Developer (via Slack)",
                    f"{r['cause']} Seen on {n} of {N} quotes for {attr} = {val}. Please check the adapter's mapping.",
                    "settings_ethernet")
    return _rec("fix_ai", "Fix in the AI pipeline", "Developer (via Slack)",
                f"{r['cause']} {n} of {N} quotes with {attr} = {val} were corrected the same way. "
                f"Suggested fix: {r['fix']}", "build")


def build_profiles(events, rules, split):
    """Per-customer view (deliberately not per-rep): which edits are codifiable, which are judgement calls."""
    by_c = defaultdict(list)
    for e in events:
        by_c[e["customer"]].append(e)
    profiles = []
    for cust, E in by_c.items():
        base_n = sum(1 for e in E if e["day"] < split)
        total_e = fixable = 0
        for e in E:
            for f in e["edits"]:
                total_e += 1
                if any(r["field"] == f and _matches(e, r["attr"], r["value"]) for r in rules):
                    fixable += 1
        items = []
        for r in rules:
            M = [e for e in E if _matches(e, r["attr"], r["value"])]
            ed = [e for e in M if r["field"] in e["edits"]]
            if len(M) >= 5 and len(ed) >= 3 and len(ed) / len(M) >= 0.5:
                items.append({"rule_id": r["id"], "statement": r["statement"], "label": r["label"],
                              "n": len(M), "n_edited": len(ed), "action": r["action"]})
        items.sort(key=lambda x: -x["n_edited"])
        edited_q = sum(1 for e in E if e["edits"])
        if base_n == 0:
            status, note = "new", "New customer: no history yet. See the new-input alert."
        elif fixable >= 10:
            status, note = "gaps", f"{fixable} manual edits could be avoided by saving rules to this customer's profile."
        else:
            status, note = "healthy", "Edits here are mostly judgement calls; nothing to codify."
        profiles.append({
            "customer": cust, "tms": E[0]["tms"], "n": len(E), "edit_rate": round(edited_q / len(E), 3),
            "n_edits": total_e, "fixable_edits": fixable, "judgement_edits": total_e - fixable,
            "status": status, "note": note, "rules": items,
        })
    profiles.sort(key=lambda p: (-p["fixable_edits"], p["customer"]))
    return profiles


def _candidate_rule(events, cust, consistent):
    """A fixable error found in a brand-new kind of RFQ: the correction is consistent, so it can become a customer rule."""
    field, kind, corr, disp = consistent
    S = [x for x in events if x["customer"] == cust]
    ed = [x for x in S if field in x["edits"]]
    if len(ed) < 5:
        return None
    out = [x for x in events if x["customer"] != cust]
    r = {"id": f"{field}|customer={cust}", "field": field, "attr": "customer", "value": cust, "kind": kind,
         "correction": corr, "dispersion": round(disp, 4), "n": len(S), "n_edited": len(ed),
         "rate": round(len(ed) / len(S), 3),
         "outside_rate": round(sum(1 for x in out if field in x["edits"]) / max(1, len(out)), 3),
         "event_ids": [x["id"] for x in ed], "source": "From a new-RFQ alert", "label": FIELD_META[field][0]}
    r["statement"] = _describe(r)
    r["cause_class"], r["cause"] = _cause(r)
    r["fix"] = _fix(r)
    r["action"] = _rec("add_to_profile", "Save as a customer rule", "Developer, as a rule the AI applies for this customer",
                       f"Reps correct the {r['label']} for {cust} the same way on {len(ed)} of {len(S)} quotes. "
                       f"Save \u201c{r['statement']}\u201d as a rule, so the AI applies it until the extraction is fixed?",
                       "person_add")
    return r


# ---------------------------------------------------------------- formula rules (the correction is a calculation)
FORMULA_TOL = 0.02          # a final price "matches" a weight basis if it is within 2% of rate x basis
FORMULA_MIN_MATCH = 0.85    # share of edited quotes a basis must explain


def discover_formula_rules(events):
    """Some corrections are not a constant but a calculation: price = rate x chargeable weight, where chargeable weight
    is the larger of actual and volume weight. Test candidate weight bases against the final prices of edited quotes."""
    f = "freight_price_eur"
    bases = [("the larger of actual and volume weight", lambda e: max(e["actual_kg"], e["volume_kg"])),
             ("volume weight", lambda e: e["volume_kg"]),
             ("actual weight", lambda e: e["actual_kg"])]
    cands = []
    for attr in ATTRS:
        groups = defaultdict(list)
        for ev in events:
            groups[ev[attr]].append(ev)
        for value, S in groups.items():
            if len(S) < 10:
                continue
            ed = [x for x in S if f in x["edits"] and x.get("rate_per_kg") and x.get("actual_kg", 0) > 0]
            rate = len(ed) / len(S)
            if len(ed) < MIN_EDITED or rate < MIN_RATE:
                continue

            def share(fn, side="to"):
                ok = [x for x in ed if fn(x) > 0 and abs(x["edits"][f][side] / x["rate_per_kg"] - fn(x)) <= FORMULA_TOL * fn(x)]
                return len(ok) / len(ed)

            best_share, best_name = max(((share(fn), name) for name, fn in bases), key=lambda t: t[0])
            if best_share < FORMULA_MIN_MATCH or best_name == "actual weight":
                continue
            ai_actual = share(bases[2][1], "from")
            out = [x for x in events if x[attr] != value]
            out_rate = sum(1 for x in out if f in x["edits"]) / max(1, len(out))
            cands.append({
                "id": f"{f}|{attr}={value}", "field": f, "attr": attr, "value": value, "kind": "formula",
                "correction": best_name, "dispersion": round(1 - best_share, 4),
                "n": len(S), "n_edited": len(ed), "rate": round(rate, 3), "outside_rate": round(out_rate, 3),
                "event_ids": [x["id"] for x in ed],
                "statement": f"When {attr} = {value}: price on {best_name}, not on actual weight",
                "cause_class": "AI mistake",
                "cause": (f"The AI prices on actual weight ({ai_actual:.0%} of its drafts), but reps re-price on {best_name} "
                          f"in {best_share:.0%} of edits (volume weight = L\u00d7W\u00d7H \u00f7 6000 per piece)."),
                "fix": "Compute chargeable weight = max(actual, volume weight) before pricing, and add test cases for bulky cargo.",
                "label": FIELD_META[f][0], "formula_match": round(best_share, 3),
            })
    cands.sort(key=lambda r: (-r["n_edited"], -r["rate"]))
    chosen = []
    for r in cands:
        s = set(r["event_ids"])
        if not any(len(s & set(c["event_ids"])) / len(s) > 0.5 for c in chosen):
            chosen.append(r)
    for r in chosen:
        r["action"] = _recommend_rule(r)
    return chosen


# ---------------------------------------------------------------- feature 3: cases nobody reported
JOURNEY_MIN_CLUSTER = 8


def _edit_signature(ev):
    """What the rep did, as numbers: per field, was it edited, and by how much (log ratio, clipped to +-1)."""
    sig = []
    for f in FIELDS:
        ed = ev["edits"].get(f)
        if not ed:
            sig += [0.0, 0.0]
        elif f in CATEG:
            sig += [1.0, 0.0]
        else:
            a, b = ed["from"], ed["to"]
            mag = 1.0 if not a else (-1.0 if b <= 0 else max(-1.0, min(1.0, math.log(b / a) / math.log(3))))
            sig += [1.0, mag]
    return sig


def find_unreported(events, embedder):
    """Cluster edited quotes by journey (what the request looked like + what the rep changed). A cluster that contains a
    bug report but mostly unreported quotes shows how big the reported problem really is."""
    edited = [ev for ev in events if ev["edits"]]
    if len(edited) < 3 * JOURNEY_MIN_CLUSTER:
        return []
    X = embedder.transform([ev["email"] for ev in edited])
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    d1 = float(np.median(nn.kneighbors(X)[0][:, 1]))
    scale = 0.5 / max(d1, 1e-6)                 # put the email part on the same footing as the edit signature
    Z = np.hstack([X * scale, np.array([_edit_signature(ev) for ev in edited])])
    labels = HDBSCAN(min_cluster_size=JOURNEY_MIN_CLUSTER, min_samples=3, copy=True).fit_predict(Z)
    # one journey can span several clusters (e.g. a customer's edits in different routes): merge by what changed + who
    buckets = defaultdict(list)
    for lab in sorted(set(labels.tolist()) - {-1}):
        M = [edited[i] for i in range(len(edited)) if labels[i] == lab]
        fc = Counter(f for m in M for f in m["edits"])
        main = tuple([f for f, c in fc.most_common() if c / len(M) >= 0.6][:3] or [fc.most_common(1)[0][0]])
        buckets[(main, Counter(m["customer"] for m in M).most_common(1)[0][0])].extend(M)
    out = []
    for (main, _top), M in buckets.items():
        # a report counts only if it is about the field this journey corrects
        reported = [m for m in M if m.get("bug_report") and m.get("bug_field") in main]
        rep_ids = {m["id"] for m in reported}
        unreported = [m for m in M if m["id"] not in rep_ids]
        if not reported or len(unreported) < 3:
            continue
        names = [FIELD_META[f][0] for f in main]
        cust = Counter(m["customer"] for m in M).most_common(3)
        reps = list(dict.fromkeys(m["bug_report"] for m in reported))[:3]
        out.append({
            "id": f"journey|{'+'.join(main)}|{cust[0][0]}", "title": f"{' + '.join(names).capitalize()} edits, mostly {cust[0][0]}",
            "fields": names, "customers": [{"name": c, "n": n} for c, n in cust],
            "n": len(M), "n_reported": len(reported), "n_unreported": len(unreported),
            "reports": reps, "event_ids": [m["id"] for m in unreported][:8],
            "reported_ids": [m["id"] for m in reported][:8],
            "summary": (f"{len(M)} quotes were corrected the same way ({', '.join(names)}). Only {len(reported)} "
                        f"{'was' if len(reported) == 1 else 'were'} reported; {len(unreported)} never were."),
            "action": _rec("review_unreported", "Review the unreported cases", "Developer (via Slack)",
                           f"{len(reported)} user report{'s' if len(reported) > 1 else ''} (\u201c{reps[0]}\u201d) match "
                           f"{len(unreported)} other quotes with the same kind of request and the same correction, which nobody "
                           f"reported. Open the linked traces and replays, and check whether customers are affected.", "travel_explore"),
        })
    out.sort(key=lambda j: -j["n_unreported"])
    return out


# ---------------------------------------------------------------- top level
def analyse(events, split, embedder):
    annotate(events)
    base = [e for e in events if e["day"] < split]
    mon = [e for e in events if e["day"] >= split]
    rules = discover_rules(base) + discover_formula_rules(base)
    for r in rules:
        r["series"] = _series(r, events)
        r["monitor"] = {"n": 0, "n_edited": 0, "rate": 0}
    drift = detect_drift(rules, events, split)
    for a in drift:
        a["series"] = next(r["series"] for r in rules if r["id"] == a["rule_id"])
    clusters, other, emb_info = cluster_and_novelty(events, split, rules, embedder)
    for a in other:                       # a fixable error in a new kind of RFQ becomes a candidate customer rule
        cr = a.pop("candidate_rule", None)
        if cr:
            if not any(r["id"] == cr["id"] for r in rules):
                cr["series"] = _series(cr, events)
                cr["monitor"] = {"n": cr["n"], "n_edited": cr["n_edited"], "rate": cr["rate"]}
                rules.append(cr)
            a["candidate_rule_id"] = cr["id"]
    journeys = find_unreported(events, embedder)
    alerts = sorted(drift + other, key=lambda a: (a["severity"] != "high", a["type"]))
    # bug reports attached to rules / alerts
    by_id = {e["id"]: e for e in events}
    def reports(ids, field):
        return [by_id[i]["bug_report"] for i in ids
                if by_id[i]["bug_report"] and (field is None or by_id[i].get("bug_field") == field)][:3]

    for r in rules:
        r["bug_reports"] = reports(r["event_ids"], r["field"])
    for a in alerts:
        a["bug_reports"] = reports(a["event_ids"], a.get("field"))
    profiles = build_profiles(events, rules, split)
    all_edits = sum(p["n_edits"] for p in profiles)
    avoidable = sum(p["fixable_edits"] for p in profiles)
    kpi = {
        "avoidable_edits": round(avoidable / max(1, all_edits), 3),
        "unreported": sum(j["n_unreported"] for j in journeys),
        "events": len(events),
        "unedited_base": round(sum(1 for e in base if not e["edits"]) / max(1, len(base)), 3),
        "unedited_mon": round(sum(1 for e in mon if not e["edits"]) / max(1, len(mon)), 3),
        "rules": len(rules), "alerts": len(alerts), "split": split,
    }
    return {"kpi": kpi, "rules": rules, "alerts": alerts, "clusters": clusters, "embedding": emb_info,
            "profiles": profiles, "journeys": journeys}
