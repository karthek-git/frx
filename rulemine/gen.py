"""Synthetic quoting data shaped like PostHog events joined with Logfire traces.

Each event = one RFQ, the AI's drafted quote (proposal), and what the rep finally sent (final).
Planted ground truth (so we can measure what the engine rediscovers):

  R1  customer = Alpina Tools      -> margin_pct 8 -> 12            (customer margin policy)
  R2  departure = FRA              -> trucking_eur +40 (days 0-44), +55 (day 45+)   <- DRIFT
  R3  unit = lb                    -> weight_kg x0.4536             (lbs read as kg)
  R4  cargo = lithium              -> dg_surcharge_eur 0 -> 85      (special cargo missed)
  R5  tms = CarLo                  -> currency EUR -> USD           (TMS adapter mapping)
  N1  Northwind Logistics (weights in metric tonnes, from day 50) -> weight x1000   <- NOVEL INPUT
  N2  PetAir Relocations (live animals, from day 52) -> reps rewrite price, routing and add a handling fee by hand;
      values vary every time, so there is no rule to learn. The AI has no knowledge of this cargo.   <- CAPABILITY GAP
  R6  bulk = bulky (Rhine Furniture, from day 10): the AI prices on ACTUAL weight, reps re-price on CHARGEABLE weight =
      larger of actual and volume weight (L x W x H / 6000 per piece). The correction differs every time (it is a formula,
      not a constant), so it is found by testing candidate formulas against the final prices.
  noise: ~5% random margin tweaks (negotiation, not learnable)
"""
import random

DAYS = 60
SPLIT_DAY = 45          # baseline = days < 45, monitoring window = days >= 45
NOVEL_FROM = 50
LIVE_FROM = 52
FURNITURE_FROM = 10
RATE_PER_KG = {"SHA": 2.8, "JFK": 2.4, "SIN": 2.9, "DXB": 2.2, "BOM": 2.5}   # EUR per chargeable kg

CUSTOMERS = [
    # name, tms, weight, typical cargo
    ("Alpina Tools", "CargoWise", 15, ["power tools", "spare parts", "machine parts"]),
    ("Bavaria Motors Parts", "CargoWise", 18, ["brake components", "engine parts", "spare parts"]),
    ("Rhein Pharma", "Riege Scope", 14, ["temperature controlled pharmaceuticals"]),
    ("Hansa Electronics", "CarLo", 9, ["circuit boards", "sensors", "electronic modules"]),
    ("Sunrise Textiles", "FM3", 16, ["garments in cartons", "fabric rolls"]),
    ("Kraft Machinery", "CarLo", 9, ["machine components", "hydraulic pumps"]),
]
NOVEL_CUSTOMER = ("Northwind Logistics", "CargoSoft", 0, ["industrial valves", "compressor units"])

DEPARTURES = [("FRA", "Frankfurt", 20), ("MUC", "Munich", 45), ("HAM", "Hamburg", 35)]
DESTS = [("SHA", "Shanghai"), ("JFK", "New York"), ("SIN", "Singapore"), ("DXB", "Dubai"), ("BOM", "Mumbai")]
INCOTERMS = ["EXW", "FCA", "DAP", "FOB"]
CONTACTS = ["Anna", "Markus", "Julia", "Stefan", "Petra", "Tom"]

BUG_TEXT = {
    "weight_kg": ["weight looks wrong, kg vs lbs?", "weight is off", "wrong chargeable weight"],
    "trucking_eur": ["trucking too low", "pre-carriage price is wrong for Frankfurt", "trucking missing something"],
    "margin_pct": ["wrong margin for this customer", "margin too low"],
    "dg_surcharge_eur": ["forgot the battery surcharge", "DG fee missing"],
    "currency": ["currency is wrong in TMS", "TMS shows USD but quote says EUR"],
    "freight_price_eur": ["price too low for bulky goods", "AI priced the actual weight, not the volume", "volume weight missing from the price"],
}
LIVE_CUSTOMER = ("PetAir Relocations", "FM3")
LIVE_ANIMALS = [   # description, container, weight range (kg)
    ("2 dogs and 1 cat", "IATA-approved kennels", (60, 140)),
    ("1 horse", "an air stall", (900, 1300)),
    ("3 racehorses", "air stalls", (2400, 3600)),
    ("4 parrots", "travel cages", (20, 60)),
    ("12 racing pigeons", "ventilated baskets", (15, 40)),
    ("2 alpacas", "ventilated crates", (180, 320)),
]
BUG_TEXT_LIVE = ["AI has no idea about live animals, I rewrote the whole quote",
                 "needs direct flights only and airline acceptance, none of that is in the draft",
                 "no live animal handling fee in the quote"]
BUG_TEXT_NOVEL = ["weight is 1000x too low, it was in tonnes", "weight looks wrong, I sent it in metric tons"]


def _vol_kg(pieces, dims):
    """Volume weight in kg for air freight: pieces x (L x W x H in cm) / 6000."""
    l, w, h = (int(x) for x in dims.split("x"))
    return float(pieces * l * w * h) / 6000.0


def _pick(rng, weighted, widx=-1):
    total = sum(item[widx] for item in weighted)
    r = rng.uniform(0, total)
    acc = 0
    for item in weighted:
        acc += item[widx]
        if r <= acc:
            return item
    return weighted[-1]


def _email_en(rng, cust, dep, dest, pieces, stated, unit, dims, cargo_desc, inco):
    return (
        f"Hello, could you quote air freight from {dep[1]} to {dest[1]}? "
        f"{pieces} pieces, total {stated} {unit}. Dimensions {dims} cm per piece. "
        f"Cargo: {cargo_desc}. Incoterm {inco}. Best regards, {rng.choice(CONTACTS)}, {cust}"
    )


def _email_tonnes(rng, cust, dep, dest, pieces, tonnes_str, dims, cargo_desc, inco):
    # a different house style: consignment-note wording, gross mass in metric tonnes
    return (
        f"Consignment request. Route: {dep[1]} to {dest[1]}. "
        f"Consignment details: {pieces} packages, gross mass {tonnes_str} t (metric tonnes), package size {dims} cm. "
        f"Commodity: {cargo_desc}. Delivery terms: {inco}. Kind regards, {rng.choice(CONTACTS)}, {cust}"
    )


def _email_live(rng, cust, dep, dest, desc, crate, kg):
    return (
        f"Hi, we are relocating {desc} from {dep[1]} to {dest[1]} and need a price for live animal transport by air. "
        f"The animals travel in {crate}, total weight with crates {kg} kg. They have vet health certificates and "
        f"microchips, and some need a CITES permit. Please advise on routing, airline acceptance and any live animal "
        f"handling fees. Regards, {rng.choice(CONTACTS)}, {cust}"
    )


def make_events(seed=7):
    rng = random.Random(seed)
    events = []
    eid = 0
    for day in range(DAYS):
        plan = [_pick(rng, CUSTOMERS, 2) for _ in range(rng.randint(11, 17))]
        if day >= NOVEL_FROM:
            plan += [NOVEL_CUSTOMER] * rng.randint(2, 4)
        for cust_row in plan:
            cust, tms, _w, cargos = cust_row[0], cust_row[1], cust_row[2], cust_row[3]
            novel = cust == NOVEL_CUSTOMER[0]
            dep = _pick(rng, DEPARTURES)
            dest = rng.choice(DESTS)
            inco = rng.choice(INCOTERMS)
            pieces = rng.randint(1, 12)
            dims = f"{rng.randint(40, 120)}x{rng.randint(30, 100)}x{rng.randint(30, 110)}"

            # cargo type
            if cust == "Hansa Electronics" and rng.random() < 0.35 or (not novel and rng.random() < 0.08):
                cargo, cargo_desc = "lithium", "lithium-ion battery packs (UN3481)"
            elif cust == "Rhein Pharma":
                cargo, cargo_desc = "perishable", cargos[0]
            else:
                cargo, cargo_desc = "general", rng.choice(cargos)

            # unit
            lb = (cust == "Sunrise Textiles" and rng.random() < 0.9) or (not novel and rng.random() < 0.04)
            unit = "lbs" if lb else "kg"

            true_kg = rng.randint(1000, 4000) if novel else rng.randint(40, 1800)
            if novel:
                tonnes_str = f"{true_kg / 1000:.3f}"
                ai_weight = float(tonnes_str)            # "1.873 t" read as 1.873 kg
                email = _email_tonnes(rng, cust, dep, dest, pieces, tonnes_str, dims, cargo_desc, inco)
            else:
                stated = round(true_kg / 0.4536) if lb else true_kg
                ai_weight = float(stated)                # lbs read as kg
                email = _email_en(rng, cust, dep, dest, pieces, stated, unit, dims, cargo_desc, inco)

            ai_trucking = float(rng.choice([90, 110, 120, 140, 160]))
            proposal = {
                "margin_pct": 8.0,
                "trucking_eur": ai_trucking,
                "weight_kg": ai_weight,
                "dg_surcharge_eur": 0.0,
                "currency": "EUR",
                "special_handling_eur": 0.0,
                "routing": "standard",
                "freight_price_eur": float(round(RATE_PER_KG[dest[0]] * ai_weight)),
            }
            final = dict(proposal)

            def applies(p):
                return rng.random() < p

            # planted rules (reps fix ~90% of the time)
            if cust == "Alpina Tools" and applies(0.92):
                final["margin_pct"] = 12.0
            elif not novel and rng.random() < 0.05:
                final["margin_pct"] = 8.0 + rng.choice([-2.0, -1.0, 1.0, 2.0])   # negotiation noise
            if dep[0] == "FRA" and applies(0.9):
                final["trucking_eur"] = ai_trucking + (40.0 if day < SPLIT_DAY else 55.0)
            if lb and applies(0.9):
                final["weight_kg"] = float(true_kg)
            if cargo == "lithium" and applies(0.9):
                final["dg_surcharge_eur"] = 85.0
            if tms == "CarLo" and applies(0.85):
                final["currency"] = "USD"
            if novel and applies(0.95):
                final["weight_kg"] = float(true_kg)

            attrs_dep = dep[0]
            ev = {
                "id": f"q{eid:05d}",
                "day": day,
                "customer": cust,
                "tms": tms,
                "lang": "en",
                "departure": attrs_dep,
                "dest": dest[0],
                "cargo": cargo,
                "unit": unit if not novel else "kg",
                "email": email,
                "proposal": proposal,
                "final": final,
                "trace_id": f"logfire://trace/{rng.getrandbits(48):012x}",
                "replay_id": f"posthog://replay/{rng.getrandbits(40):010x}",
                "bug_report": None,
                "bulk": "standard", "actual_kg": float(true_kg), "volume_kg": _vol_kg(pieces, dims),
                "rate_per_kg": RATE_PER_KG[dest[0]],
            }
            eid += 1
            events.append(ev)

    # a few bug reports on edited events (explicit, rare signal)
    for ev in events:
        edits = [f for f in proposal_fields() if ev["proposal"][f] != ev["final"][f]]
        if edits and rng.random() < 0.05:
            f = rng.choice(edits)
            ev["bug_report"] = rng.choice(BUG_TEXT_NOVEL if ev["customer"] == NOVEL_CUSTOMER[0] and f == "weight_kg" else BUG_TEXT[f])
            ev["bug_field"] = f

    # live-animal RFQs: appended with their own random stream so the rest of the dataset (and its cached embeddings)
    # is unchanged. The AI knows nothing about this cargo; reps rewrite the quote and the values differ every time.
    lrng = random.Random(seed + 101)
    for day in range(LIVE_FROM, DAYS):
        for _ in range(lrng.randint(1, 3)):
            desc, crate, (lo, hi) = lrng.choice(LIVE_ANIMALS)
            dep, dest = _pick(lrng, DEPARTURES), lrng.choice(DESTS)
            kg = lrng.randint(lo, hi)
            ai_trucking = float(lrng.choice([90, 110, 120, 140, 160]))
            proposal = {"margin_pct": 8.0, "trucking_eur": ai_trucking, "weight_kg": float(kg), "dg_surcharge_eur": 0.0,
                        "currency": "EUR", "special_handling_eur": 0.0, "routing": "standard",
                        "freight_price_eur": float(round(RATE_PER_KG[dest[0]] * kg))}
            final = dict(proposal)
            if lrng.random() < 0.95:
                final["special_handling_eur"] = float(lrng.randint(300, 1500))
            if lrng.random() < 0.8:
                final["trucking_eur"] = ai_trucking + lrng.randint(150, 500)
            if lrng.random() < 0.9:
                final["routing"] = "direct only"
            if lrng.random() < 0.7:
                final["margin_pct"] = lrng.choice([15.0, 18.0, 20.0, 25.0])
            ev = {
                "id": f"q{len(events):05d}", "day": day, "customer": LIVE_CUSTOMER[0], "tms": LIVE_CUSTOMER[1],
                "lang": "en", "departure": dep[0], "dest": dest[0], "cargo": "live_animals", "unit": "kg",
                "email": _email_live(lrng, LIVE_CUSTOMER[0], dep, dest, desc, crate, kg),
                "proposal": proposal, "final": final,
                "trace_id": f"logfire://trace/{lrng.getrandbits(48):012x}",
                "replay_id": f"posthog://replay/{lrng.getrandbits(40):010x}",
                "bug_report": None,
                "bulk": "standard", "actual_kg": float(kg), "volume_kg": 0.0, "rate_per_kg": RATE_PER_KG[dest[0]],
            }
            if lrng.random() < 0.4:
                ev["bug_report"], ev["bug_field"] = lrng.choice(BUG_TEXT_LIVE), "special_handling_eur"
            events.append(ev)

    # bulky-goods customer: the AI prices on actual weight, reps re-price on chargeable weight (own random stream)
    frng = random.Random(seed + 202)
    furniture = ["upholstered armchairs", "foam mattresses", "flat-pack wardrobes", "inflatable boats"]
    for day in range(FURNITURE_FROM, DAYS):
        for _ in range(frng.randint(1, 3)):
            dep, dest, inco = _pick(frng, DEPARTURES), frng.choice(DESTS), frng.choice(INCOTERMS)
            pieces = frng.randint(2, 8)
            if frng.random() < 0.85:      # bulky: large, light pieces, so volume weight exceeds actual weight
                dims = f"{frng.randint(120, 200)}x{frng.randint(80, 120)}x{frng.randint(80, 140)}"
                per_piece_kg = frng.randint(25, 90)
            else:                          # dense: small, heavy pieces, so the actual weight is already the larger
                dims = f"{frng.randint(40, 70)}x{frng.randint(30, 50)}x{frng.randint(30, 50)}"
                per_piece_kg = frng.randint(150, 300)
            actual, vol = float(pieces * per_piece_kg), _vol_kg(pieces, dims)
            rate = RATE_PER_KG[dest[0]]
            ai_trucking = float(frng.choice([90, 110, 120, 140, 160]))
            proposal = {"margin_pct": 8.0, "trucking_eur": ai_trucking, "weight_kg": actual, "dg_surcharge_eur": 0.0,
                        "currency": "EUR", "special_handling_eur": 0.0, "routing": "standard",
                        "freight_price_eur": float(round(rate * actual))}
            final = dict(proposal)
            if vol > actual and frng.random() < 0.92:
                final["freight_price_eur"] = float(round(rate * max(actual, vol)))
            if dep[0] == "FRA" and frng.random() < 0.9:
                final["trucking_eur"] = ai_trucking + (40.0 if day < SPLIT_DAY else 55.0)
            ev = {
                "id": f"q{len(events):05d}", "day": day, "customer": "Rhine Furniture GmbH", "tms": "CargoWise",
                "lang": "en", "departure": dep[0], "dest": dest[0], "cargo": "furniture", "unit": "kg",
                "email": _email_en(frng, "Rhine Furniture GmbH", dep, dest, pieces, int(actual), "kg", dims,
                                   frng.choice(furniture), inco),
                "proposal": proposal, "final": final,
                "trace_id": f"logfire://trace/{frng.getrandbits(48):012x}",
                "replay_id": f"posthog://replay/{frng.getrandbits(40):010x}",
                "bug_report": None,
                "bulk": "bulky" if vol > actual else "standard", "actual_kg": actual, "volume_kg": vol, "rate_per_kg": rate,
            }
            if "freight_price_eur" in [f for f in proposal_fields() if proposal[f] != final[f]] and frng.random() < 0.08:
                ev["bug_report"], ev["bug_field"] = frng.choice(BUG_TEXT["freight_price_eur"]), "freight_price_eur"
            events.append(ev)
    return events


def proposal_fields():
    return ["margin_pct", "trucking_eur", "weight_kg", "dg_surcharge_eur", "currency", "special_handling_eur", "routing",
            "freight_price_eur"]


if __name__ == "__main__":
    evs = make_events()
    edited = sum(1 for e in evs if e["proposal"] != e["final"])
    print(f"{len(evs)} events, {edited} edited ({100 * (1 - edited / len(evs)):.0f}% unedited)")
