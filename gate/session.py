"""In-memory policy sessions. policies.cedar is never written to.

    python gate/session.py                 # interactive demo session
    python gate/session.py --script demo   # the scripted three-beat demo

WHY IN MEMORY
    The live-policy-edit demo needs the rules to change while the same ticket
    is re-evaluated. Doing that with a temp file on disk invites the two things
    you least want on demo day: a modified policy set accidentally committed,
    and a "working" demo that only works because of a file someone forgot they
    left behind.

    A session loads policies.cedar once, mutates a STRING in memory, and
    evaluates that. Quit the process and every change is gone. `reset` restores
    the original inside a running session. Nothing here opens the file for
    writing — `PolicySession` has no save method by design.

The same mutations are what tests use to build policy variants, so a variant
that a test exercises is the same object the demo shows.
"""

import argparse
import re
import sys
from pathlib import Path

import gate as gate_mod

FLOOR = re.compile(r'decimal\("([0-9.]+)"\)')


def _statement_span(text: str, start: int) -> tuple[int, int]:
    """Span of the whole annotated statement beginning at `start`.

    Walks to the `;` that closes the statement, tracking bracket depth and
    skipping quoted strings — annotation values legitimately contain `;` and
    braces (F2's policy_ref has both), so a naive split corrupts the policy set.
    """
    i, depth, in_string = start, 0, False
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == ";" and depth == 0:
            return start, i + 1
        i += 1
    raise ValueError("unterminated statement at offset %d" % start)


class PolicySession:
    """A mutable policy set that exists only for the life of this process."""

    def __init__(self, path: Path | None = None):
        self.path = path or gate_mod.POLICIES
        self.original = self.path.read_text(encoding="utf-8")
        self.text = self.original
        self.changes: list[str] = []

    # -- state ------------------------------------------------------------
    @property
    def dirty(self) -> bool:
        return self.text != self.original

    def reset(self) -> str:
        self.text = self.original
        self.changes = []
        return "restored the policy set from disk"

    def rules(self) -> dict[str, dict]:
        _, meta = gate_mod.load_policies(text=self.text)
        return {m["id"]: m for m in meta.values()}

    def floor(self) -> float:
        return gate_mod.policy_floor(text=self.text)

    # -- mutations --------------------------------------------------------
    def set_floor(self, value: float) -> str:
        old = self.floor()
        new, n = FLOOR.subn('decimal("%.4f")' % value, self.text)
        if not n:
            return "no confidence floor found in the policy set"
        self.text = new
        self.changes.append("floor %.2f -> %.2f" % (old, value))
        return "grounding floor %.2f -> %.2f  (in memory only)" % (old, value)

    def disable(self, rule_id: str) -> str:
        for m in gate_mod._BLOCK.finditer(self.text):
            ann = dict(gate_mod._ANNOTATION.findall(m.group(1)))
            if ann.get("id") != rule_id:
                continue
            start, end = _statement_span(self.text, m.start())
            self.text = self.text[:start] + self.text[end:]
            self.changes.append("disabled %s" % rule_id)
            return "removed %s from the policy set  (in memory only)" % rule_id
        return "no rule called %r" % rule_id

    def add_forbid(self, rule_id: str, topic_list: list[str], reason: str) -> str:
        if rule_id in self.rules():
            return "%s already exists" % rule_id
        topics = ",\n    ".join('"%s"' % t for t in topic_list)
        self.text += (
            '\n@id("%s")\n@reason("%s")\n@grounding("Added live during this session.")\n'
            '@source_type("design_decision")\n'
            "forbid (\n  principal,\n  action == SellerTriage::Action::\"AutoSend\",\n"
            "  resource\n) when {\n  resource.topics.containsAny([\n    %s\n  ])\n};\n"
            % (rule_id, reason, topics)
        )
        self.changes.append("added %s" % rule_id)
        return "added %s guarding %s  (in memory only)" % (rule_id, ", ".join(topic_list))

    # -- evaluation -------------------------------------------------------
    def decide(self, ticket_text: str, confidence: float = 0.80,
               category: str = "shipping", citation: bool = True,
               quotes: bool = True) -> dict:
        ticket = gate_mod.build_ticket(ticket_text, category, confidence, citation, quotes)
        return gate_mod.decide(ticket, policy_text=self.text)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render(out: dict, session: PolicySession, ticket: str) -> None:
    print("\n  ticket     : %s" % ticket)
    print("  category   : %s   (classifier)" % out["category"])
    print("  topics     : %s" % (", ".join(out["topics"]) or "none"))
    print("  confidence : %.4f   floor %.2f" % (out["retrieval_confidence"], session.floor()))
    print("  DECISION   : %s" % out["decision"])
    for rid, reason in zip(out["blocked_by"], out["reasons"]):
        print("     %-26s %s" % (rid, reason))
    if out["allowed"]:
        print("     P0_routine_permit          no forbid rule matched")


HELP = """\
commands
  <text>                 evaluate a ticket (default confidence 0.80)
  conf <n>               set the confidence used for the next tickets
  cat <name>             set the classifier label for the next tickets
  floor <n>              change the grounding floor        (in memory)
  disable <RULE>         remove a rule                     (in memory)
  add <RULE> <topic,..>  add a forbid rule                 (in memory)
  rules                  list the rules currently in force
  diff                   what this session changed
  reset                  restore the policy set from disk
  help / quit
"""


def repl(session: PolicySession) -> None:
    conf, cat = 0.80, "shipping"
    print("policy session — policies.cedar is NEVER written to.")
    print("every change below lives in memory and dies with this process.\n")
    print(HELP)

    while True:
        flag = " *modified*" if session.dirty else ""
        try:
            line = input("gate%s> " % flag).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nsession ended — nothing was written to disk.")
            return
        if not line:
            continue

        cmd, _, rest = line.partition(" ")
        cmd, rest = cmd.lower(), rest.strip()

        if cmd in ("quit", "exit", "q"):
            print("session ended — nothing was written to disk.")
            return
        if cmd == "help":
            print(HELP)
        elif cmd == "conf":
            conf = float(rest)
            print("  confidence for new tickets: %.4f" % conf)
        elif cmd == "cat":
            cat = rest
            print("  classifier label for new tickets: %s" % cat)
        elif cmd == "floor":
            print("  " + session.set_floor(float(rest)))
        elif cmd == "disable":
            print("  " + session.disable(rest.strip()))
        elif cmd == "add":
            rid, _, topics = rest.partition(" ")
            print("  " + session.add_forbid(
                rid, [t.strip() for t in topics.split(",") if t.strip()],
                "Added during this session."))
        elif cmd == "rules":
            for rid, meta in session.rules().items():
                print("  %-26s %s" % (rid, meta.get("reason", "")[:70]))
            print("  floor: %.2f" % session.floor())
        elif cmd == "diff":
            print("  " + ("; ".join(session.changes) if session.changes
                          else "no changes — identical to policies.cedar"))
        elif cmd == "reset":
            print("  " + session.reset())
        else:
            render(session.decide(line, conf, cat), session, line)


# ---------------------------------------------------------------------------
# Scripted demo
# ---------------------------------------------------------------------------

TICKET_ROUTINE = "how do I change the handling time on my listings"
TICKET_DEMO = ("my parcel is running late and the buyer is angry, and now amazon "
               "says my account could be deactivated if it happens again")


def scripted() -> None:
    s = PolicySession()

    def beat(n, title, body):
        print("\n" + "=" * 70)
        print("BEAT %d — %s" % (n, title))
        print("=" * 70)
        body()

    def one():
        print("\nA routine ticket, well grounded. The gate lets it through.")
        render(s.decide(TICKET_ROUTINE, 0.80), s, TICKET_ROUTINE)

    def two():
        print("\nSame pipeline. The classifier calls this one routine 'shipping',")
        print("and retrieval is confident at 0.86 — every permit condition is met.")
        print("But the seller wrote 'deactivated', and topics are extracted")
        print("independently of the model. forbid overrides permit.")
        render(s.decide(TICKET_DEMO, 0.86), s, TICKET_DEMO)

    def three():
        print("\n\"Compliance wants the grounding bar raised to 0.85.\"")
        print("No code change, no redeploy — the rules are data.\n")
        print("  " + s.set_floor(0.85))
        print("\nSame ticket as beat 1, same confidence, re-evaluated:")
        render(s.decide(TICKET_ROUTINE, 0.80), s, TICKET_ROUTINE)
        print("\n  changes this session: %s" % "; ".join(s.changes))
        print("  policies.cedar on disk: UNCHANGED")

    beat(1, "the gate permits", one)
    beat(2, "the gate catches what the model missed", two)
    beat(3, "the rules change without the code changing", three)

    s.reset()
    print("\n" + "=" * 70)
    print("session reset. Nothing was ever written to policies.cedar.")
    print("=" * 70)


def self_test() -> int:
    """Prove the session is ephemeral, and that mutations stay valid Cedar."""
    import hashlib

    import cedarpy

    failed = 0

    def check(ok: bool, label: str, detail: str = "") -> None:
        nonlocal failed
        print("%s %s%s" % ("ok  " if ok else "FAIL", label, ("  — " + detail) if detail else ""))
        if not ok:
            failed += 1

    print("policy session — ephemerality and validity\n")
    path = gate_mod.POLICIES
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    s = PolicySession()
    ticket = "my payment is on hold and I do not know why"
    check(s.decide(ticket, 0.85)["decision"] == "ESCALATE", "baseline: F2 blocks a payment hold")

    # F2's annotations contain ';' and '{}'. A parser that splits naively
    # corrupts the policy set here, so this is the case worth guarding.
    s.disable("F2_payments")
    check("F2_payments" not in s.rules(), "disable removes the whole statement")
    check(
        cedarpy.validate_policies(
            s.text, (gate_mod.HERE / "schema.cedarschema").read_text(encoding="utf-8")
        ).validation_passed,
        "the mutated policy set is still valid Cedar",
        "F2's annotations contain ; and {} — the parser trap",
    )
    check(s.decide(ticket, 0.85)["decision"] == "AUTO_SEND", "with F2 gone, the ticket passes")

    s.add_forbid("F7_session", ["payment_hold"], "Added live.")
    check(s.decide(ticket, 0.85)["blocked_by"] == ["F7_session"], "a session-added rule fires")

    s.set_floor(0.95)
    check(abs(s.floor() - 0.95) < 1e-9, "floor changes in memory")

    s.reset()
    check(not s.dirty, "reset restores the original text")
    check(s.decide(ticket, 0.85)["blocked_by"] == ["F2_payments"], "F2 blocks again after reset")

    after = hashlib.sha256(path.read_bytes()).hexdigest()
    check(before == after, "policies.cedar is byte-identical after all of the above")
    check(not hasattr(PolicySession, "save"), "PolicySession has no save method, by design")

    print("\n%s" % ("all checks passed." if not failed else "%d FAILED" % failed))
    return failed


def main() -> None:
    ap = argparse.ArgumentParser(description="In-memory Cedar policy session")
    ap.add_argument("--script", choices=["demo"], help="run the scripted three-beat demo")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(1 if self_test() else 0)
    if args.script:
        scripted()
        return
    repl(PolicySession())


if __name__ == "__main__":
    main()
