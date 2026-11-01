"""Slack message rendering. Sends for real ONLY when SLACK_WEBHOOK_URL is set AND send() is called."""
import json
import os
import urllib.request

ICON = {"rule_drift": ":chart_with_upwards_trend:", "novel_inputs": ":new:", "cluster_shift": ":bar_chart:"}


def render(alert, dashboard_url="http://localhost:8000"):
    lines = [
        f"{ICON.get(alert['type'], ':warning:')} *{alert['title']}*",
        alert["summary"],
        f"*Likely cause:* {alert['hypothesis']}",
        f"*Suggested fix:* {alert['fix']}",
    ]
    if alert.get("bug_reports"):
        lines.append("*User reports:* " + "; ".join(f"\"{b}\"" for b in alert["bug_reports"]))
    lines.append(f"<{dashboard_url}/#alert={alert['id']}|Open evidence>  (traces + replays linked)")
    text = "\n".join(lines)
    return {"text": text, "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": text}}]}


def configured():
    return bool(os.environ.get("SLACK_WEBHOOK_URL"))


def send(alert, dashboard_url="http://localhost:8000"):
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        return {"sent": False, "reason": "SLACK_WEBHOOK_URL not set; preview only"}
    req = urllib.request.Request(url, data=json.dumps(render(alert, dashboard_url)).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return {"sent": True, "status": r.status}
