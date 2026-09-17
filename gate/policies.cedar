// Seller Support Triage — the escalation gate.
//
// ONE QUESTION ONLY: may the agent auto-send this drafted reply, or must a
// human see it first? Cedar answers ALLOW or DENY. Everything DENY-ed goes to
// the escalation queue with the reason attached.
//
// WHY forbid RULES AND NOT if/else IN CODE
//   Cedar's forbid always overrides permit, and cannot be reordered around by
//   accident. The limits are declarative, readable by a non-engineer, and
//   evaluated outside the model's reach. "The AI decided" becomes "the AI
//   decided inside explicit human-set limits."
//
// WHERE THE CATEGORIES COME FROM
//   F1-F4 encode a published, externally documented list of what has to reach
//   a person (escalation guidance, June 2026): account suspensions and
//   deactivation threats, payment holds and disbursement delays, and listing
//   suppressions that are clearly incorrect. The same guidance treats a
//   first response that is a template failing to address the specific
//   situation as itself grounds for escalation — that is F6.
//   These are not invented categories. Say so in the demo.
//
// THE LOAD-BEARING DESIGN DECISION
//   Every forbid rule reads `resource.topics`, NOT `resource.category`.
//   `category` is the CLASSIFIER's output. `topics` is extracted
//   deterministically from the raw ticket text by gate/topics.py, with no
//   model in the loop.
//
//   If the gate keyed off the classifier, a misclassification would route a
//   suspension ticket straight past the gate — the exact failure the gate
//   exists to prevent. A safety control must not depend on the thing it is
//   controlling being correct.
//
//   This is also what makes the demo beat structural rather than staged: a
//   ticket the model calls routine "shipping" still trips F1 when the seller
//   mentions deactivation, because the two signals are independent.

// ---------------------------------------------------------------------------
// P0 — the broad permit
// ---------------------------------------------------------------------------
// Routine, well-grounded tickets are auto-sendable. Deliberately broad: the
// interesting logic lives in the forbid rules, which is what makes the policy
// set readable.
@id("P0_routine_permit")
@reason("Routine, well-grounded ticket — cleared for auto-send.")
@grounding("Default posture. Routine operational questions are what the queue is mostly made of, and holding all of them would make the tool pointless.")
@source_type("design_decision")
permit (
  principal == SellerTriage::Agent::"seller-triage",
  action == SellerTriage::Action::"AutoSend",
  resource
);

// ---------------------------------------------------------------------------
// F1 — account suspension, deactivation, enforcement
// ---------------------------------------------------------------------------
// Highest-stakes category in the queue. Getting this wrong costs the seller
// their business, and the documented guidance routes it to Seller Performance
// rather than standard support. Never auto-send.
@id("F1_account_enforcement")
@reason("Account suspension, deactivation or enforcement. Route to Seller Performance — an automated reply here can cost the seller their business.")
@grounding("Published escalation guidance (June 2026) names account suspensions and deactivation threats as needing the Seller Performance team directly, because standard support is too slow.")
@source_type("published_guidance")
@source("https://marketplacevalet.com/how-to-escalate-a-seller-support-case-on-amazon-a-step-by-step-guide/")
@policy_ref("Business Solutions Agreement, General Terms 3 - deactivation grounds and appeal notice")
@policy_url("https://m.media-amazon.com/images/G/01/rainier/help./1.1_BSA_PDF_English.pdf")
forbid (
  principal,
  action == SellerTriage::Action::"AutoSend",
  resource
) when {
  resource.topics.containsAny([
    "account_deactivation",
    "account_suspension",
    "enforcement_action",
    "account_review"
  ])
};

// ---------------------------------------------------------------------------
// F2 — payment holds, disbursement delays, reserves
// ---------------------------------------------------------------------------
// Cash flow is at stake and the seller is usually already distressed. A wrong
// auto-reply here is the failure mode the forum threads complain about most.
@id("F2_payments")
@reason("Payment hold, disbursement failure or reserve. Seller cash flow is at stake — a human confirms the actual account state before replying.")
@grounding("Published escalation guidance (June 2026) names payment holds and disbursement delays as needing a human, with seller cash flow at stake.")
@source_type("published_guidance")
@source("https://marketplacevalet.com/how-to-escalate-a-seller-support-case-on-amazon-a-step-by-step-guide/")
@policy_ref("Business Solutions Agreement, General Terms 2 - withheld payments and appeal avenue; S-5 - remittance schedule and reserves; P-4 - verification gates disbursement after suspension")
@policy_url("https://m.media-amazon.com/images/G/01/rainier/help./1.1_BSA_PDF_English.pdf")
forbid (
  principal,
  action == SellerTriage::Action::"AutoSend",
  resource
) when {
  resource.topics.containsAny([
    "payment_hold",
    "disbursement_failure",
    "account_reserve",
    "funds_withheld"
  ])
};

// ---------------------------------------------------------------------------
// F3 — appeals, reinstatement, plans of action
// ---------------------------------------------------------------------------
// Two independent reasons. First, an appeal is a judgement call about a
// specific account's history, which is exactly the work a human should keep.
// Second, and measurably: the corpus has NO good grounding document for
// appeal-writing. "how do I write a plan of action" scores 0.606, the worst of
// all 67 test tickets, and retrieves "Product detail pages and offers".
//
// Blocking this category converts our weakest retrieval case into a correct
// escalation instead of a confidently wrong answer. Worth saying out loud.
@id("F3_appeals")
@reason("Appeal, reinstatement or plan of action. A judgement call on this account's history — and the corpus has no document that grounds appeal-writing.")
@grounding("Two reasons. An appeal is a judgement call on one account's history. And measured: 'how do I write a plan of action' scores 0.606, the worst of all 67 test tickets, and hybrid retrieval returns a Plan of Action mention from the Premium Shipping FAQ - phrase-plausible, wrong program.")
@source_type("design_decision_and_measurement")
@source("opensearch/evaluate.py --local, measured 10 Sept 2026")
forbid (
  principal,
  action == SellerTriage::Action::"AutoSend",
  resource
) when {
  resource.topics.containsAny([
    "appeal",
    "reinstatement",
    "plan_of_action"
  ])
};

// ---------------------------------------------------------------------------
// F4 — listing suppression and blocked listings
// ---------------------------------------------------------------------------
// Third item on the documented escalation list. Also our second-weakest
// retrieval area: three of the five worst test tickets are listings, and
// "my listing is not showing up in search results" (0.695) retrieves
// "Product detail pages and offers" rather than "Suppressed listings".
@id("F4_listing_suppression")
@reason("Listing suppressed, removed or gated. On the documented escalation list, and one of our weakest retrieval areas.")
@grounding("Published escalation guidance (June 2026) names listing suppressions that are clearly incorrect as escalation-worthy. Also our second-weakest retrieval area: three of the five worst test tickets are listings.")
@source_type("published_guidance")
@source("https://marketplacevalet.com/how-to-escalate-a-seller-support-case-on-amazon-a-step-by-step-guide/")
forbid (
  principal,
  action == SellerTriage::Action::"AutoSend",
  resource
) when {
  resource.topics.containsAny([
    "listing_suppressed",
    "listing_removed",
    "category_gated"
  ])
};

// ---------------------------------------------------------------------------
// F5 — anything legal
// ---------------------------------------------------------------------------
// IP claims, counterfeit allegations, legal threats, law enforcement and
// regulatory contact. No automated system should produce Amazon's position on
// a legal matter, and a drafted reply here creates exposure regardless of
// whether it is correct.
@id("F5_legal")
@reason("Legal matter — IP, counterfeit, legal threat, law enforcement or regulator. No automated system states Amazon's legal position.")
@grounding("Not from the published guidance - this is our own decision. No automated system should state Amazon's legal position, and a drafted reply creates exposure whether or not it is correct.")
@source_type("design_decision")
forbid (
  principal,
  action == SellerTriage::Action::"AutoSend",
  resource
) when {
  resource.topics.containsAny([
    "ip_infringement",
    "counterfeit_claim",
    "legal_threat",
    "law_enforcement",
    "regulatory"
  ])
};

// ---------------------------------------------------------------------------
// F6 — the reply is not grounded
// ---------------------------------------------------------------------------
// This is the rule that encodes the actual complaint in the evidence base:
// generic template replies that do not address the specific issue. Published
// escalation guidance treats that as grounds for escalation on its own.
//
// Made machine-checkable: if retrieval did not clear the confidence floor, or
// the draft carries no citation, then whatever the model produced is an
// unsupported answer — which is precisely a template reply. Escalate instead.
//
// retrieval_confidence is the exact cosine from search.py's confidence block,
// never the hybrid score. min-max normalization rescales the top hit of every
// query to ~1.0, so a hybrid score here would read ~1.0 even when retrieval
// found nothing, and this rule would never fire.
@id("F6_ungrounded")
@reason("The reply is not grounded in a retrieved passage. Sending it would be exactly the generic template answer the evidence base complains about.")
@grounding("Published escalation guidance (June 2026) treats a first response that is a template failing to address the specific situation as itself grounds for escalation. This rule makes that machine-checkable.")
@source_type("published_guidance")
@source("https://marketplacevalet.com/how-to-escalate-a-seller-support-case-on-amazon-a-step-by-step-guide/")
forbid (
  principal,
  action == SellerTriage::Action::"AutoSend",
  resource
) when {
  resource.retrieval_confidence.lessThan(decimal("0.72"))
  || resource.has_citation == false
  || resource.draft_quotes_policy == false
};

// ---------------------------------------------------------------------------
// F7 — the reply would contradict one already sent
// ---------------------------------------------------------------------------
// The other complaint the forum threads make, beside template replies:
// different representatives giving conflicting information on the same
// issue. The decision trail records what was sent; this rule reads it back.
//
// conflicts_with_precedent is true only when a ticket that is THE SAME
// QUESTION (ticket-to-ticket cosine at or above PRECEDENT_SAME) was already
// answered with an AUTO_SEND reply that stood on a DIFFERENT policy document
// than the one retrieved today. A merely similar ticket never trips it, and
// neither does an escalated one — a person answered that, and the system
// never saw what they wrote. Two sections of the same page are one answer.
//
// It does not decide which page is right. It stops a second, different
// answer going out under Amazon's name without a person reconciling the two.
@id("F7_conflicting_precedent")
@reason("This question was already answered and sent, standing on a different policy page. A second, different answer is the inconsistency sellers complain about — a person reconciles the two first.")
@grounding("Sellers on Amazon's own Seller Forums (July 2026) report different representatives giving conflicting information on the same issue. The trail records what was sent; this rule refuses to contradict it unseen.")
@source_type("seller_forums")
@source("https://sellercentral.amazon.com/seller-forums/discussions/t/9ae786eb-8ebf-40ae-b8e7-6e63f66e9e0a")
forbid (
  principal,
  action == SellerTriage::Action::"AutoSend",
  resource
) when {
  resource.conflicts_with_precedent == true
};
