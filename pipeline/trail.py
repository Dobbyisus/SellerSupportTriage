"""The decision trail — every step of every ticket, in DynamoDB's shape.

Records use the single-table layout from the architecture:

    pk = TICKET#<id>     sk = META        the ticket and its outcome
    pk = TICKET#<id>     sk = STEP#<n>    one row per pipeline step
    pk = COVERAGE#<ym>   sk = RUN#<ms>#<id>   one row per run, for the gaps report

One query on the partition key returns the whole trail in order, so the audit
log and the console's data source are the same read. That is deliberate: the
thing shown to a judge is the thing the system actually wrote.

The second partition is the same trick applied to the month instead of the
ticket: one query returns every run in it, which is how coverage.py reports
what the corpus could not answer without ever scanning the table. The shape of
those rows belongs to coverage.py; only the read primitive lives here.

The local store writes the identical records to a JSONL file, so switching to
DynamoDB changes where rows land and nothing about their shape.
"""

import json
import time
from pathlib import Path
from typing import Protocol

import settings


def now_ms() -> int:
    return int(time.time() * 1000)


def meta_item(ticket_id: str, text: str) -> dict:
    return {
        "pk": "TICKET#%s" % ticket_id,
        "sk": "META",
        "ticket_id": ticket_id,
        "text": text,
        "created_ms": now_ms(),
        "backends": settings.summary(),
        "status": "in_progress",
    }


def step_item(ticket_id: str, n: int, step: str, detail: dict) -> dict:
    return {
        "pk": "TICKET#%s" % ticket_id,
        # Zero-padded so lexical sort matches numeric order — DynamoDB sorts
        # sort keys as strings, and STEP#10 must not fall between 1 and 2.
        "sk": "STEP#%03d" % n,
        "ticket_id": ticket_id,
        "n": n,
        "step": step,
        "at_ms": now_ms(),
        **detail,
    }


class Trail(Protocol):
    def put(self, item: dict) -> None: ...
    def query(self, ticket_id: str) -> list[dict]: ...
    def query_pk(self, pk: str) -> list[dict]: ...


class JsonlTrail:
    """Append-only local file. Same records DynamoDB would hold."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or settings.TRAIL_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def put(self, item: dict) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    def query(self, ticket_id: str) -> list[dict]:
        return self.query_pk("TICKET#%s" % ticket_id)

    def query_pk(self, pk: str) -> list[dict]:
        """Last write wins per (pk, sk), mirroring DynamoDB's put_item.

        The file is append-only, so a META row written at the start and again
        at the end appears twice on disk. DynamoDB would have overwritten it,
        and the local store has to behave the same way or the console shows a
        ticket as both in_progress and complete.
        """
        latest: dict[str, dict] = {}
        for r in self.all():
            if r.get("pk") == pk:
                latest[r["sk"]] = r
        return [latest[k] for k in sorted(latest)]

    def all(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def tickets(self) -> list[dict]:
        """Every META row, newest first — the console's queue view."""
        metas = [r for r in self.all() if r.get("sk") == "META"]
        # A ticket re-run overwrites nothing in an append-only file, so keep
        # the most recent META per ticket_id.
        latest: dict[str, dict] = {}
        for m in metas:
            prev = latest.get(m["ticket_id"])
            if prev is None or m["created_ms"] >= prev["created_ms"]:
                latest[m["ticket_id"]] = m
        return sorted(latest.values(), key=lambda m: -m["created_ms"])


class DynamoTrail:
    """Same records, in DynamoDB. Serves every deployed request."""

    def __init__(self, table_name: str | None = None):
        import boto3

        self.table_name = table_name or settings.DYNAMO_TABLE
        self.table = boto3.resource("dynamodb").Table(self.table_name)

    def put(self, item: dict) -> None:
        self.table.put_item(Item=_floats_to_decimal(item))

    def query(self, ticket_id: str) -> list[dict]:
        return self.query_pk("TICKET#%s" % ticket_id)

    def query_pk(self, pk: str) -> list[dict]:
        """Every row in one partition, in sort-key order.

        Paginated, unlike the original single-ticket read: a ticket has eight
        steps and never needs a second page, but a month of runs will.
        """
        from boto3.dynamodb.conditions import Key

        rows: list[dict] = []
        kwargs = {"KeyConditionExpression": Key("pk").eq(pk)}
        while True:
            resp = self.table.query(**kwargs)
            rows.extend(resp.get("Items", []))
            if "LastEvaluatedKey" not in resp:
                break
            kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
        return sorted(rows, key=lambda r: r["sk"])

    def tickets(self) -> list[dict]:
        """Every META row, newest first. A SCAN — developer-side only.

        The function's role deliberately holds PutItem and Query and nothing
        else, so this raises AccessDenied from the Lambda. It exists for
        `pipeline/precedent.py --backfill`, which runs with the developer's
        credentials, and for nothing in the request path.
        """
        from boto3.dynamodb.conditions import Attr

        rows: list[dict] = []
        kwargs = {"FilterExpression": Attr("sk").eq("META")}
        while True:
            resp = self.table.scan(**kwargs)
            rows.extend(resp.get("Items", []))
            if "LastEvaluatedKey" not in resp:
                break
            kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
        return sorted(rows, key=lambda m: -int(m.get("created_ms", 0)))


def _floats_to_decimal(obj):
    """DynamoDB rejects float. Convert on the way in."""
    from decimal import Decimal

    if isinstance(obj, float):
        return Decimal(str(round(obj, 6)))
    if isinstance(obj, dict):
        return {k: _floats_to_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_floats_to_decimal(v) for v in obj]
    return obj


def open_trail() -> Trail:
    return DynamoTrail() if settings.TRAIL == "dynamodb" else JsonlTrail()
