"""AWS Lambda entry point for the Seller Support Triage pipeline.

    GET  /                                the operator console (HTML)
    POST /ticket        {"text": "..."}   run a ticket, get the decision
    GET  /ticket/{id}                     replay one decision trail
    GET  /health                          liveness + warm the container

WHY THE CONSOLE IS SERVED FROM HERE
    Same origin as the API, so the page calls `ticket` and `health` as
    relative paths with no CORS involved and nothing else to host. One URL is
    the whole system — paste it anywhere and it works.

`Pipeline.run()` was always the handler body; this module is the thin wrapper
that turns an API Gateway event into a call and the result into JSON. The CLI
in `pipeline/run.py` is the same call with a different front end.

WHY THE PIPELINE IS BUILT AT MODULE SCOPE
    Building it loads the embedding model, the 710 chunks and the Cedar
    policies. That is the ~3s cold start. At module scope it is paid once per
    container and reused by every warm invocation (measured locally: median
    49ms, p90 65ms). Building it inside the handler would pay it per request
    and make the demo unusable.

WHY IMPORT FAILURE IS CAUGHT RATHER THAN RAISED
    An exception at module scope gives Lambda's opaque "Runtime.ImportModuleError"
    with no useful detail. Catching it here means the first request returns a
    readable 500 naming the real cause, which is the difference between five
    minutes and an hour of debugging on the day.
"""

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pipeline"))

_PIPELINE = None
_INIT_ERROR = None

try:
    import settings          # noqa: E402  (bootstraps gate/ and opensearch/ onto sys.path)
    import trail as trail_mod  # noqa: E402
    from orchestrator import Pipeline  # noqa: E402

    _PIPELINE = Pipeline()
except Exception:                       # noqa: BLE001 — see the docstring
    _INIT_ERROR = traceback.format_exc()


# CORS is the Function URL's job, not ours. Its URL config already answers
# preflights and stamps Access-Control-Allow-Origin on every response. When
# this handler added the same header itself, browsers saw two values
# ("*, <origin>") and refused the response — so the console worked when the
# Lambda served it (same origin, no CORS) and failed from anywhere else.
CORS: dict[str, str] = {}


def _reply(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json", **CORS},
        "body": json.dumps(body, default=str),
    }


def _route(event: dict) -> tuple[str, str]:
    """Method and path, for both API Gateway payload formats.

    HTTP API (v2) and Function URLs put them under requestContext.http; REST
    API (v1) puts them at the top level. Supporting both costs three lines and
    removes a whole class of "works in the console, 404s from the browser".
    """
    ctx = (event.get("requestContext") or {}).get("http") or {}
    method = ctx.get("method") or event.get("httpMethod") or "GET"
    path = ctx.get("path") or event.get("rawPath") or event.get("path") or "/"
    return method.upper(), path.rstrip("/") or "/"


def _body(event: dict) -> dict:
    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        import base64
        raw = base64.b64decode(raw).decode("utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def lambda_handler(event, context):
    method, path = _route(event)

    # Preflight. Answer before the init check — a browser must be able to
    # preflight even a broken deployment, or the real error never surfaces.
    if method == "OPTIONS":
        return _reply(200, {"ok": True})

    if _INIT_ERROR is not None:
        return _reply(500, {"error": "pipeline failed to initialise",
                            "detail": _INIT_ERROR})

    try:
        # Liveness, and the warm-up call. Send one of these a minute before
        # demoing: it pays the cold start while nobody is watching, so the
        # first real ticket runs on a warm container.
        if path == "/health":
            return _reply(200, {
                "ok": True,
                "retriever": settings.RETRIEVER,
                "drafter": settings.DRAFTER,
                "trail": settings.TRAIL,
                "precedent": settings.PRECEDENT,
            })

        # The console. Read from disk per request rather than cached at import:
        # the file is ~9KB, the read is microseconds against a multi-second
        # request, and it means a broken console can never take the API down
        # with it at module scope.
        if method == "GET" and path == "/":
            try:
                html = (ROOT / "console.html").read_text(encoding="utf-8")
            except OSError as e:
                return _reply(500, {"error": "console.html missing", "detail": str(e)})
            return {
                "statusCode": 200,
                "headers": {"Content-Type": "text/html; charset=utf-8", **CORS},
                "body": html,
            }

        if method == "POST" and path in ("/ticket", "/"):
            text = (_body(event).get("text") or "").strip()
            if not text:
                return _reply(400, {"error": "body must be {\"text\": \"...\"}"})
            return _reply(200, _PIPELINE.run(text))

        if method == "GET" and path.startswith("/ticket/"):
            ticket_id = path.rsplit("/", 1)[-1]
            rows = _PIPELINE.trail.query(ticket_id)
            if not rows:
                return _reply(404, {"error": "no trail for %s" % ticket_id})
            return _reply(200, {"ticket_id": ticket_id, "trail": rows})

        return _reply(404, {"error": "no route for %s %s" % (method, path)})

    except Exception:                    # noqa: BLE001
        # Never let a stack trace escape as an API Gateway 502. The trail has
        # already recorded whatever completed before the failure.
        return _reply(500, {"error": "pipeline error",
                            "detail": traceback.format_exc()})
