# Seller Support Triage

A support agent for Amazon sellers that drafts replies from real Amazon policy and knows which replies it is not allowed to send.

A seller's ticket comes in. The system finds the policy page the answer stands on, drafts a reply from that page only, and a fixed rulebook decides whether the reply goes out or a person must look first. Every step is written down in plain words, so a support agent can see why a ticket was decided the way it was.

**It is live.** One URL is the whole system: the operator console at the root and the JSON API beside it.

https://vxmsr2n6ufxzgasohsc6yasdqu0mvwzx.lambda-url.us-east-1.on.aws/

| Route | What it does |
| --- | --- |
| `GET /` | The console. Raise a ticket, watch it go through the seven stops, read why it was decided |
| `POST /ticket` | `{"text": "..."}` returns the decision, the draft, the citation, the confidence and the trail |
| `GET /ticket/{id}` | Replay one decision trail |
| `GET /health` | Liveness, and warms the container |

The first request after a quiet spell takes about fifteen seconds while the container starts. After that a ticket takes a few seconds.

## What sellers complain about, and what was built for each

The design started from six public threads on Amazon's own Seller Forums and from published escalation guidance. The complaints that recur became specific parts of the system.

| Complaint | What answers it |
| --- | --- |
| Generic template replies that do not address the issue | Every draft is built from one retrieved policy passage, and every quoted line and every link in it is checked back against that passage. A draft that fails is held for a person |
| Different representatives giving conflicting information on the same issue | Every ticket is compared with every ticket the system has already answered. If the same question was answered and sent on a different policy page, the reply is held |
| Case types where an automated reply is not acceptable | Account suspensions and deactivation threats, payment holds, listing suppressions, appeals and legal matters are rules in the gate, three of them taken directly from published guidance |

## The seven stops

Every ticket takes the same stops in the same order. There is exactly one point where the system decides something for itself.

1. **Ticket in.** The seller's words, untouched.
2. **Sort.** A classifier picks a category. It is used only to route the search, never to decide anything.
3. **Find policy.** Search over 710 passages of first-party Amazon policy, returning the passage, its source URL and an exact similarity score.
4. **Search again.** Only when the score is below the floor: one retry with seller-language synonyms, keeping whichever attempt scored better. Both attempts are written to the trail.
5. **Asked before.** The ticket is compared by meaning with every ticket ever answered. If the same question was answered, the agent sees what went out and which page it stood on.
6. **Draft.** Bedrock writes a reply from the retrieved passage only. Every quoted line and every link is verified against it.
7. **Decide.** Seven Cedar rules read the seller's own words and the facts produced upstream, and return auto-send or escalate with the rule that fired and where that rule comes from.

The trail of all seven stops is written to DynamoDB, one row per stop, and is what the console shows.

## The gate

The gate answers one question: may this drafted reply go out without a person seeing it? One rule permits routine, grounded tickets. Six forbid, and in Cedar a forbid always wins.

| Rule | Holds the reply when |
| --- | --- |
| Account at risk | Deactivation, suspension, enforcement or an account review is mentioned |
| Payment hold | Payment holds, disbursement failures, reserves or withheld funds |
| Appeal or reinstatement | Appeals, reinstatement, plans of action |
| Listing suppressed | Suppressed, removed or gated listings |
| Legal matter | IP, counterfeits, legal threats, law enforcement, regulators |
| Reply not grounded in policy | Retrieval below the floor, no citation, or the draft quotes nothing verifiable |
| Conflicts with an earlier reply | The same question was already answered and sent on a different policy page |

Every forbid rule reads the seller's own words, extracted by a deterministic matcher with no model in it, and never the classifier's category. If the gate keyed off the classifier, a misclassified suspension ticket would route straight past it. A safety control must not depend on the thing it is controlling being correct.

The gate is tested with 23 example cases and a property test over the whole defined input space: every topic the matcher can emit, crossed with both sides of every boolean input and both sides of the confidence floor, 480 states, all evaluated through the real Cedar engine. The honest framing is exhaustively tested over the defined input space, not formally verified.

```bash
python gate/gate.py --self-test
python gate/property_test.py
python gate/gate.py --why "my account was deactivated" --confidence 0.85
```

## Architecture

Five AWS services sit in the request path, and each is called on every ticket.

| Service | Role |
| --- | --- |
| Lambda, container image, public Function URL | Serves the console and the API. A container because the embedding runtime and model exceed the zip limit. A Function URL because API Gateway caps an integration at 29 seconds and a cold start alone measured close to 20 |
| OpenSearch, policy index | Hybrid keyword and vector search over the 710 passages, with a 28-rule synonym list applied at search time. The vector half is exact, and its cosine is what the gate reads |
| OpenSearch, ticket index | One document per past ticket, for the precedent check. Kept separate from the policy corpus so a past reply can never be retrieved as if it were policy |
| Bedrock | Drafts the reply. Mistral Large 3, chosen because it declined to invent next steps the passage did not contain |
| Cedar | The seven-rule gate, evaluated in process by the real Cedar engine |
| DynamoDB | The decision trail, one table, one partition per ticket, on-demand billing |

Embeddings are a 384-dimension model that runs inside the function, so retrieval has no network dependency and the cosines are reproducible offline. The retrieval floor of 0.72 was calibrated against it.

The knowledge base is 45 first-party Amazon documents, US marketplace only, selected from 594 extracted documents by keeping those that answer a realistic seller ticket. It is cut into 710 passages that each carry a heading path, section id, source URL and category. `chunks.jsonl` is committed so a clone can run.

## Measured, including what did not win

- Retrieval: 62 of 67 plain-language test tickets clear the similarity floor. The three worst cases all fall in categories that never auto-send, so a confidently wrong answer becomes a correct escalation.
- Hybrid search against the live index: 58 of 67 usable either way, no additional ticket rescued, no rank changed, mean similarity 0.759 hybrid against 0.762 vector-only. It ships at parity, and parity is the claim made.
- The gate: 23 cases and 480 property states, all passing.

## Running it locally

Everything runs with no AWS at all; the AWS backends are switched in by environment variables.

```bash
pip install -r requirements.txt
python pipeline/run.py "my payment is on hold and I don't know why"
python pipeline/run.py --demo
```

| Variable | Values | Default |
| --- | --- | --- |
| `PIPELINE_RETRIEVER` | `local`, `opensearch` | `local` |
| `PIPELINE_CLASSIFIER` | `local`, `bedrock` | `local` |
| `PIPELINE_DRAFTER` | `mantle`, `bedrock`, `template` | `mantle` |
| `PIPELINE_TRAIL` | `jsonl`, `dynamodb` | `jsonl` |
| `PIPELINE_PRECEDENT` | `jsonl`, `opensearch`, `off` | `jsonl` |

`PIPELINE_DRAFTER=template` produces a grounded template reply from the passage with no network call. It is also the fallback the live drafter degrades into on a timeout or an error, and the trail records when that happened.

To preview retrieval without a cluster:

```bash
python opensearch/local_search.py -k 3 "my payment is on hold"
```

## Deploying

The Lambda is a container image. The build flags are not optional; without them the image manifest is rejected when the function is updated, not when it is built.

```bash
docker build --provenance=false --sbom=false -t seller-triage:latest .
python deploy.py push
python deploy.py update
```

`push` tags and pushes the image already on the machine; it does not build. `update` points the function at the new image and waits for the update to finish.

## Repository layout

| Path | What it is |
| --- | --- |
| `handler.py` | Lambda entry point: serves the console and the API |
| `console.html` | The operator console, one file, no build step |
| `pipeline/` | The orchestrator, the backends, the precedent check, the trail |
| `gate/` | The Cedar policies, the topic matcher, the tests |
| `opensearch/` | Index mapping, upload, hybrid query, evaluation |
| `chunks.jsonl` | The 710 policy passages with their embeddings |
| `build_corpus.py`, `semantic_filter.py`, `build_chunks.py` | The offline knowledge base build, in that order |
| `deploy.py` | Push the image and update the function without the AWS CLI |

## The console

The console is for support agents, not engineers. It shows why a ticket was decided, never how the system works: rules are named "Payment hold" or "Reply not grounded in policy", and stops report "close match", "every quote checks out" or "needs a person". No similarity scores, passage ids or rule codes appear on screen.

It has a board of paper tickets in four columns that an agent moves by drag and drop, a ledger that replays the seven stops with the decision each one made, and a stage panel that explains the stop under the playhead in full. Everything on screen is read from the API response. Six real trail runs are built in as demo tickets so the board is never empty.

## Built with Claude

This project was built with Claude, not by Claude. Claude Code was the coding partner throughout, and the design decisions were the author's: which complaints to design against, that the gate must read the seller's words rather than the classifier, that hybrid search ships at parity rather than being claimed as a win, and what the console may and may not show a support agent.

Built for the Bharat Builds Tour "First Commit" hackathon by WeMakeDevs and AWS.
