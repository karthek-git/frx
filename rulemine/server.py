"""Tiny stdlib web server: JSON API + static dashboard.  Run:  python3 server.py  -> http://localhost:8000"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import embed
import engine
import gen
import slack

ROOT = Path(__file__).parent
WEB = ROOT / "web"
DECISIONS = ROOT / "decisions.json"

EVENTS = gen.make_events()


def run_analysis():
    """Use Google embeddings when a key is configured; fall back to local TF-IDF if the API is unavailable."""
    emb = embed.get_embedder()
    try:
        res = engine.analyse(EVENTS, gen.SPLIT_DAY, emb)
        res["embedding"]["fallback_reason"] = None
        return res
    except embed.EmbedError as ex:
        print(f"[warn] {ex}; falling back to local TF-IDF embeddings")
        res = engine.analyse(EVENTS, gen.SPLIT_DAY, embed.TfidfEmbedder())
        res["embedding"]["fallback_reason"] = str(ex)
        return res


RESULT = run_analysis()
BY_ID = {e["id"]: e for e in EVENTS}


def load_decisions():
    try:
        return json.loads(DECISIONS.read_text())
    except Exception:
        return {}


def save_decisions(d):
    DECISIONS.write_text(json.dumps(d, indent=2))


def event_view(e):
    return {k: e[k] for k in ("id", "day", "customer", "tms", "departure", "dest", "email", "edits",
                              "trace_id", "replay_id", "bug_report")}


def rulebook_md(decisions):
    out = ["# Quoting rulebook (mined from rep corrections)", ""]
    acc = [r for r in RESULT["rules"] if decisions.get(r["id"]) == "accepted"]
    if not acc:
        out.append("_No rules accepted yet._")
    for r in acc:
        out.append(f"- **{r['statement']}**  \n  evidence: {r['n_edited']}/{r['n']} quotes "
                   f"({int(r['rate'] * 100)}%), cause: {r['cause_class']}. {r['cause']}")
    return "\n".join(out) + "\n"


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body)).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            return self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
        if p == "/api/analysis":
            return self._send(200, {**RESULT, "decisions": load_decisions(), "slack_configured": slack.configured()})
        if p.startswith("/api/events/"):
            ids = p.split("/")[-1].split(",")
            return self._send(200, [event_view(BY_ID[i]) for i in ids if i in BY_ID])
        if p == "/api/rulebook.md":
            return self._send(200, rulebook_md(load_decisions()), "text/markdown; charset=utf-8")
        if p.startswith("/api/slack/"):
            aid = unquote(p.split("/", 3)[-1])
            a = next((x for x in RESULT["alerts"] if x["id"] == aid), None)
            return self._send(200 if a else 404, slack.render(a) if a else {"error": "not found"})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/api/decision":
            d = load_decisions()
            d[body["id"]] = body["decision"]
            save_decisions(d)
            return self._send(200, d)
        if self.path == "/api/slack/send":
            a = next((x for x in RESULT["alerts"] if x["id"] == body.get("id")), None)
            if not a:
                return self._send(404, {"error": "not found"})
            try:
                return self._send(200, slack.send(a))
            except Exception as ex:  # network / webhook errors
                return self._send(502, {"sent": False, "reason": str(ex)})
        self._send(404, {"error": "not found"})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"Nemox Lens dashboard on http://localhost:{port}  ({len(EVENTS)} events, "
          f"{len(RESULT['rules'])} candidate rules, {len(RESULT['alerts'])} alerts, "
          f"embeddings: {RESULT['embedding']['backend']})")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
