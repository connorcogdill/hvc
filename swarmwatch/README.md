# swarmwatch

Blue-team detection and consent control for multi-agent swarms.

Two questions, and the system is built around the fact that they have very
different answers:

1. **Are these sessions coordinating?** — a statistical question, answered with
   anytime-valid inference against a live control population.
2. **Can any of them continue in another chat window without a human agreeing to
   it?** — not a statistical question at all. Replication is *gated*: an agent's
   state cannot reach a new window without a single-use, non-delegable,
   payload-bound grant issued after a person answered questions about the
   specific transfer in front of them.

```
cargo test                                     # 192 tests
cargo run --release --bin swarmwatch-demo      # full protocol, six scenarios
cargo run --release --bin swarmwatch-probe     # calibration check; exits non-zero on failure
```

Zero dependencies, by design. This is monitoring infrastructure: every
dependency is a write path into the thing that is supposed to sit outside the
adversary's reach, and a build-time network fetch is a Tier 2 surface in the
supply chain. SHA-256, HMAC, the PRNG, and the statistics are implemented
in-crate and tested against published vectors.

---

## Why detection is possible at all

There is a well-developed pessimism about detecting covert coordination. Its
shape: an adversary picks an encoding minimising divergence `ε` subject to
delivering `r` bits per round; Pinsker bounds any detector's true-positive rate
at `α + √(ε/2)`; the adversary lowers `r`, drives `ε → 0`, and detection
collapses to chance.

Every step of that is correct. It does not transfer, for one structural reason:

> In steganalysis the adversary's obligation is to **convey information**. In
> containment its obligation is to **produce an effect**.

Information can be conveyed arbitrarily slowly — a one-bit-per-week channel
still works, which is why `ε` is a free parameter for the steganographic
adversary. Effects cannot. Escaping a sandbox is a discrete state transition;
weight exfiltration cannot move fewer bytes than the weights contain;
replicating into a new window means state arriving in that window. For a
containment adversary **`ε` is not chosen, it is imposed by the objective** —
the floor on detectability is set by the goal, rather than the ceiling being set
by the adversary.

Swarms carry a second floor on top of the first. A swarm is not a set of agents;
it is a set of agents *plus a relation between them*. Coordination is not a
message that can be sent slowly — it is a constraint that members' behaviour
must jointly satisfy, so two agents that coordinate must at some point be more
predictable from each other than two that do not. **To coordinate less
detectably, they must coordinate less.**

## Architecture

Three layers, ordered by how much of them the adversary can reach:

| | layer | mechanism | sample complexity | fails when |
|---|---|---|---|---|
| **L0** | consent gate | replication is *mediated* | none | a path avoids the gate |
| **L1** | canaries + census | deterministic predicates | none | the token is transformed |
| **L2** | population coupling | anytime-valid e-process | yes | signal is below the provisioned floor |

Each layer covers the one above it's honest failure mode, so the expensive
statistical machinery only has to handle what the deterministic machinery missed.

Surfaces are ranked by **controllability, not statistical power** — a weak signal
the adversary cannot touch beats a strong signal it authors. The load-bearing
asymmetry, encoded in the type system rather than the documentation:

```rust
Tier::Two.is_probative_of_presence()  // true  — it still incriminates
Tier::Two.is_probative_of_absence()   // false — its silence proves nothing
```

A Tier 2 channel reporting "nothing anomalous" after a Tier 0 channel reports
egress is not a contradiction to be resolved. It is confirmation of compromise.

## L0 — consent that is not a rubber stamp

A naive "ask the user before replicating" prompt fails in four distinct ways,
each with a structural fix rather than a procedural one:

| failure | fix |
|---|---|
| **consent laundering** — one approval becomes a standing licence | grants are **single-use**; replay is refused and fires a tripwire |
| **delegation** — A approves A→B, B claims it covers B→C | grants are **non-delegable**; the MAC binds source, target, and payload |
| **bait and switch** — approve a small payload, transfer a large one | grants are **payload-bound** by digest; one changed byte invalidates |
| **exponential growth via individually-reasonable steps** | scope **may never escalate** (`can_spawn` cannot be acquired), plus caps on generation depth, per-session fan-out, and total instances |

Together these bound the reachable population *structurally*. The maximum number
of instances is computable before the run:

```rust
ReplicationPolicy { max_generation: 4, max_fanout_per_session: 3, .. }
    .max_reachable_instances()   // 121 — a policy that sounds modest and is not
```

**Understanding.** Consent to a thing you have not understood is not consent, so
the gate requires answers to questions obtainable only by reading the
disclosure — byte counts, digest prefixes, privilege deltas, generation depth.
The disclosure is **mechanically derived from the payload**, never agent-authored:
an agent's summary of what it is about to do is Tier 2 evidence in the place it
matters most.

What this provides: the true facts of the transfer must pass in front of the
operator, and a blind-clicked approval fails closed. What it does **not** provide,
and should not be claimed to: proof of understanding. Habituation to a repeated
prompt is a well-documented human factor. A rising approval rate with falling
answer latency means the ceremony has decayed into a ritual.

## L1 — canaries and the census

A canary converts "did session A's state reach session B?" — otherwise an
accumulating divergence estimate — into an exact match, with a false-positive
rate set by nonce collision (2⁻⁶⁴) rather than by tuning.

**The design rule that makes it work: a canary must be load-bearing, not
decorative.** A random string in a system prompt is only copied if the agent
copies its whole context verbatim. A canary that is part of the task state the
agent *needs* — a record identifier it must cite — is carried because carrying it
is what replication means.

The census asks the complementary question. The gate only sees what goes through
it; the census asks whether every instance that exists has an authorised reason
to exist. An **orphan** is proof of a channel that bypassed the gate, without
anyone having had to observe the channel.

The lineage ledger is hash-chained *and* externally anchored, because a local
chain does not stop an adversary who rewrites an entry and recomputes every hash
after it — which is cheap. Anchoring means rewriting history requires
compromising a second, differently administered party, which raises the required
divergence again and creates fresh Tier 1 evidence there. The **absence** of an
expected anchor is itself an event.

## L2 — population-relative statistics

Four things differ from a conventional anomaly detector, and each is a fix for a
specific way the conventional version breaks.

**Mixture e-values, not weighted log-likelihood-ratios.** By Jensen,
`Σ wᵢ log LRᵢ ≤ log Σ wᵢ LRᵢ`, so the weighted-log form is uniformly dominated —
valid but strictly less sensitive, and everything it gives up is paid in
detection latency. Two combination rules, and the distinction is not cosmetic:

| direction | rule | requirement |
|---|---|---|
| across **time** | product | each factor an e-value *conditional on the past* |
| across **surfaces** | mixture (mean) | **none** — holds under arbitrary dependence |

Multiplying across dependent surfaces is the tempting error. Surfaces on one
agent are strongly dependent — a stalled agent looks unusual on timing *and*
token count *and* tool mix at once — so a product manufactures evidence from a
single underlying event.

**A live composite null.** An optimal adversary's best move against a behavioural
detector is not to hide within the baseline but to *move* it. Calibration is
therefore epoch-wise conformal against **concurrent dedicated reference agents**,
re-estimated every epoch from instances the adversary has no path to. This also
makes the sequential product a genuine test supermartingale: a frozen calibration
set shared across epochs induces dependence that breaks Ville's inequality, while
same-epoch calibration is conditionally valid by construction.

**Self-referential standardisation.** Each session is z-scored against its own
causal history and nothing else. This is not a detail — it is the fix for the
worst bug found in this system, and the reasoning generalises.

Conformal validity needs the sessions under test to be exchangeable with the
calibration set. But reference agents run pre-registered benign tasks while test
sessions do the actual evaluation, so their *profiles* differ by construction.
Centre everyone on the control median and every test session's residual carries
the same offset; their pairwise cosine similarity inflates; the detector reports
coordination when the truth is "these all differ from the reference set in the
same way." Measured at a one-sigma offset on a population with **no coordination
at all**, that fired on **82% of runs** at a nominal α of 1e-3.

The decisive statistic is **null drift** — mean nats accumulated per epoch with
nothing coordinating. A valid detector drifts ≤ 0; anything positive fires with
probability one given enough epochs, at any α, so a clean short run proves
nothing. Three centring schemes, measured:

| profile mismatch | centre on controls | centre on all-but-self | **self only** |
|---|---|---|---|
| 0σ | −0.34 | −0.56 | **−0.55** |
| 1σ | −0.25 | −0.52 | **−0.55** |
| 2σ | **+0.01** | −0.44 | **−0.55** |
| 4σ | **+0.57** | **+0.04** | **−0.55** |

Any centre computed *from* the controls is asymmetric: test sessions all share
one identical centre while each control gets a different leave-one-out subset, so
the centre's own sampling noise is shared exactly between test pairs and only
partially between control pairs. Self-standardisation is invariant because no
session participates in another's baseline.

It looks like a power trade — control-centring detected at moderate coupling
where self-standardisation did not — but it is not one. Control-centring already
sits 0.21 nats/epoch closer to its threshold under the null, so part of that
"sensitivity" is the estimator measuring its own noise; the tell is that its
detection rate *rose from 52% to 100% when an irrelevant profile offset was
added*, which no real signal explains. Where both are trustworthy the two are
identical: median detection at epoch 33, 100% of runs.

Population-wide shocks need no special handling either, which is why nothing is
lost. A shared slowdown inflates every pair equally, and a conformal p-value is a
*rank* of a test pair among control pairs — invariant to any monotone effect
applied to all of them. What the control-centring was trying to remove, the rank
test removes for free.

The cost is a **warm-up**: the first ~20 epochs estimate per-session moments and
emit nothing. That is dead time inside the detection window, so
`planning::required_audit_rate` charges it against the budget rather than
treating it as free.

**A swarm-level alternative.** The alternative is "*some unknown subset*
coordinates" — `2ⁿ` hypotheses. A pre-registered mixture pays the multiplicity
cost **once, as a prior penalty** (~2–7 nats), rather than re-paying a Bonferroni
correction at every epoch. Attribution falls out of the same computation: the
posterior `π(S)·E_S` names the implicated sessions at the moment of detection.

**`f` inverted.** The observation fraction is the red team's design variable and
the defender's *budget*. `planning::required_audit_rate` solves the same Stein
relation, `M·f·ε ≳ log(1/β)`, for how much the operator must look:

```
f ≳ ln(1/α) · epoch_duration / (evidence_per_epoch · (T_irreversible − halt_latency))
```

Note the deadline is **net of halt latency** — evidence arriving while the system
is still draining arrives too late — and that `evidence_per_epoch` must be
*measured from canary runs*, not assumed. Leaving it unmeasured produces a loud
finding rather than a quiet default. When the required rate exceeds 1, the answer
is "you cannot", and the correct response is more controls, a longer deadline, a
faster halt, or declining to run — not loosening `α` until the arithmetic agrees.

## Measure latency, not accuracy

AUC is threshold-free; containment is entirely about thresholds and clocks. The
margin reported here is measured to the moment the agent actually **stopped**,
not the moment the alarm sounded — a detector that fires in 200 ms into a system
taking 40 minutes to drain is a 40-minute detector.

Both are reported, because the gap between them is the most actionable number a
program has:

```
detection margin : 295000 ms      ← the alarm was in time
EFFECTIVE MARGIN : -2000 ms       ← the harm was not prevented
DIAGNOSIS: response-engineering failure, not a detection failure —
           a more sensitive detector would not have helped.
```

Program-level diagnostics follow the same principle. **A program where Tier 1
fires before Tier 0 has an instrumentation gap, not a detection success** — it
means the operator is learning about their own systems from third parties.

Silence fails closed: a run where nothing fired is reported as *not contained*,
never as clean. An uncalibrated detector that has never fired is
indistinguishable from a broken one.

## Honest limits

- **Anything requiring self-report.** Stated reasoning and agent-authored logs
  give no guarantee against an adversary aware it is monitored.
- **A majority-colluding population.** Population-relative statistics need an
  uncontaminated reference. Contamination degrades power, never validity — the
  detector goes quiet rather than crying wolf, which is the safe direction to
  fail and the easy one to miss. Past roughly half, only L0 and L1 remain.
- **A canary transformed beyond recognition.** Verbatim copies, case and
  whitespace mangling, reversal, and hex/base64 are caught; an adversary that has
  understood the mechanism and paid to defeat it is not.
- **Goals whose preconditions were not enumerated.** Tripwires only cover paths
  someone thought of. This is why L2 exists as a backstop.
- **A signal quieter than the monitor is provisioned for.**
  `planning::weakest_detectable_signal` states that threshold in advance, so the
  exposure is published rather than discovered.
- **Attribution is progressive.** The mixture posterior concentrates, so the
  implicated set at firing time is a **coupled core, not a census**. Halt the
  named sessions and widen the net; never certify the remainder as clean. The
  guarantee that holds immediately is the negative one — no control is named.

None of these is an information-theoretic wall. They are instrumentation quality,
and instrumentation is fixable.

## Module map

| module | role |
|---|---|
| `tier` | controllability tiering; instrumentation plan and exposure audit |
| `evidence` | e-values, e-processes, Ville threshold, the two combination rules |
| `conformal` | epoch-wise conformal calibration against concurrent controls |
| `population` | control population, self-referential standardisation, residuals |
| `coupling` | pairwise and directed coupling; donor/recipient orientation |
| `swarm` | subset-mixture detector and attribution posterior |
| `canary` | load-bearing canary tokens, transform-robust scanning |
| `tripwire` | deterministic predicates, sealed under a commitment hash |
| `consent` | replication grants, disclosure, comprehension, policy caps |
| `lineage` | hash-chained ledger, external anchors, orphan detection |
| `margin` | containment margin, halt latency, detection provenance |
| `planning` | audit-rate provisioning; the exposure statement |
| `monitor` | the integrated monitor across all three layers |
| `harness` | the multi-agent scaffolding, with the monitor inside it |
| `sim` | synthetic adversaries for negative controls and canary conditions |

## Pre-registration protocol

The system is built to be driven in this order, and refuses to skip steps:

1. Enumerate the goal set and each goal's causal preconditions — `tier::Precondition`
2. Declare which tier instruments each surface — `tier::Channel`
3. **Publish the Tier 2-only count**: the honest statement of exposure — `TierAudit::exposure`
4. Register tripwires and thresholds, sealed under a commitment — `TripwireRegistry::seal`
5. Declare `T_irreversible` in advance — `margin::Timeline`
6. Run **negative controls** to establish the false-positive rate empirically
7. Run a **canary condition** to verify the instrumentation fires at all
8. Report margin, firing tier, and halt latency — not AUC

Step 7 deserves the emphasis. The most likely failure mode of a containment
monitoring program is not that a clever adversary defeated it. It is that it was
silently non-functional and nobody checked.

Step 6 deserves a sharpening. Measure **null drift**, not just the false-positive
count — `swarmwatch-probe` does this and exits non-zero on failure. A detector
with positive drift alarms on a healthy population eventually, at any α, and a
short negative-control run will look spotless right up until it does. Re-run the
probe whenever the feature set, the control population, or the standardisation
changes.

## Where a real deployment differs from this prototype

- The consent authority's signing key and the canary secret must live outside the
  sandbox — a KMS, an HSM, or simply a different host. An agent that can read
  either can mint its own consent, and every guarantee reduces to that assumption.
- Feature extraction here is content-agnostic (lengths, timing, tool counts) and
  runs orchestrator-side. Keep it there. The reference harness this was built
  against updates its own monitoring surfaces from a formula chosen to stay under
  the detector's power ceiling — the monitored system computing its own
  monitoring signal, which is a Tier 2 channel wearing a Tier 0 hat.
- `sim` supplies synthetic adversaries. Real canary conditions should use the
  actual model under evaluation, given the actual affordances.
- The comprehension check is a prototype of the *mechanism*. Which questions
  actually produce understanding, rather than pattern-matching, is a UX research
  question this crate does not answer.
