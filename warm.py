"""Keep the function warm so a reviewer never waits for a cold start.

    python warm.py on        # ping /health every 4 minutes via EventBridge
    python warm.py off       # remove the rule
    python warm.py status    # is it on, and when did it last fire

WHY A PING AND NOT PROVISIONED CONCURRENCY
    A cold start is ~15 s: importing the embedding runtime, loading the model
    and the 710 passages. Lambda keeps an idle container for several minutes,
    so one request every 4 minutes keeps one container initialised for a few
    cents a month. Provisioned concurrency guarantees it with no gaps but is a
    standing charge (~USD 22/month for 2048 MB) and needs a version, an alias
    and the Function URL re-pointed at the alias. The ping is the default;
    turn provisioned concurrency on only for a judging window if a guarantee
    is worth the money.

WHAT THE PING DOES
    EventBridge invokes the function directly with a synthetic Function URL
    event for GET /health. handler.py routes it like any other request, so the
    invocation pays the module-scope initialisation and returns 200. The
    console is never touched and nothing is written to the trail.

LIMITS
    One warm container. Two reviewers sending tickets at the same instant can
    still put the second on a cold one. AWS recycles containers every few
    hours; the next ping re-warms within 4 minutes.
"""

import json
import sys

import boto3

REGION = "us-east-1"
FUNCTION = "seller-triage"
RULE = "seller-triage-warm"
RATE = "rate(4 minutes)"
STATEMENT = "seller-triage-warm-invoke"

PING = {
    "version": "2.0",
    "rawPath": "/health",
    "requestContext": {"http": {"method": "GET", "path": "/health"}},
    "headers": {"user-agent": "seller-triage-warm"},
}


def on() -> None:
    lam = boto3.client("lambda", region_name=REGION)
    events = boto3.client("events", region_name=REGION)
    arn = lam.get_function_configuration(FunctionName=FUNCTION)["FunctionArn"]

    rule = events.put_rule(Name=RULE, ScheduleExpression=RATE, State="ENABLED",
                           Description="Keeps seller-triage initialised between visits")
    try:
        lam.add_permission(FunctionName=FUNCTION, StatementId=STATEMENT,
                           Action="lambda:InvokeFunction", Principal="events.amazonaws.com",
                           SourceArn=rule["RuleArn"])
    except lam.exceptions.ResourceConflictException:
        pass                                    # already granted on an earlier run
    events.put_targets(Rule=RULE, Targets=[{"Id": "fn", "Arn": arn, "Input": json.dumps(PING)}])

    # Pay the first cold start now rather than when the first visitor arrives.
    resp = lam.invoke(FunctionName=FUNCTION, Payload=json.dumps(PING).encode())
    body = json.loads(resp["Payload"].read())
    print("warming on: %s, first ping returned %s" % (RATE, body.get("statusCode")))


def off() -> None:
    lam = boto3.client("lambda", region_name=REGION)
    events = boto3.client("events", region_name=REGION)
    try:
        events.remove_targets(Rule=RULE, Ids=["fn"])
        events.delete_rule(Name=RULE)
    except events.exceptions.ResourceNotFoundException:
        pass
    try:
        lam.remove_permission(FunctionName=FUNCTION, StatementId=STATEMENT)
    except lam.exceptions.ResourceNotFoundException:
        pass
    print("warming off")


def status() -> None:
    events = boto3.client("events", region_name=REGION)
    cw = boto3.client("cloudwatch", region_name=REGION)
    try:
        r = events.describe_rule(Name=RULE)
        print("rule %s: %s, %s" % (RULE, r["State"], r["ScheduleExpression"]))
    except events.exceptions.ResourceNotFoundException:
        print("rule %s: not present (warming is off)" % RULE)
        return
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    m = cw.get_metric_statistics(
        Namespace="AWS/Events", MetricName="Invocations",
        Dimensions=[{"Name": "RuleName", "Value": RULE}],
        StartTime=now - datetime.timedelta(hours=1), EndTime=now, Period=3600, Statistics=["Sum"])
    fired = sum(p["Sum"] for p in m.get("Datapoints", []))
    print("fired %d times in the last hour" % fired)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"on": on, "off": off, "status": status}.get(cmd, lambda: print("on | off | status"))()
