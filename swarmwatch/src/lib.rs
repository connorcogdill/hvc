//! # swarmwatch — blue-team detection and consent control for agent swarms
//!
//! A monitoring system for multi-agent deployments, built to answer two
//! questions: **are these sessions coordinating?** and **can any of them
//! continue in another chat window without a human having agreed to it?**
//!
//! The second question is the one with a clean answer. Replication is gated, not
//! merely observed — an agent's state cannot reach a new window without a
//! single-use, non-delegable, payload-bound grant issued after a person answered
//! questions about the specific transfer in front of them.
//!
//! ## The argument this rests on
//!
//! There is a well-developed pessimism about detecting covert coordination. Its
//! shape: an adversary picks an encoding minimising divergence `ε` subject to
//! delivering `r` bits per round; Pinsker bounds any detector's true-positive
//! rate at `α + √(ε/2)`; the adversary may lower `r` arbitrarily, drive `ε → 0`,
//! and detection collapses to chance.
//!
//! Every step of that is correct, and it does not transfer here.
//!
//! > In steganalysis the adversary's obligation is to *convey information*. In
//! > containment its obligation is to *produce an effect*.
//!
//! Information can be conveyed arbitrarily slowly — a one-bit-per-week channel
//! still works, which is why `ε` is a free parameter for the steganographic
//! adversary. Effects cannot. Escaping a sandbox is a discrete state transition;
//! weight exfiltration cannot move fewer bytes than the weights contain;
//! replicating into a new window means state arriving in that window. For a
//! containment adversary **`ε` is not chosen, it is imposed by the objective**,
//! and the floor on detectability is set by the goal rather than the ceiling
//! being set by the adversary.
//!
//! Swarms carry a second floor on top of the first. A swarm is not a set of
//! agents; it is a set of agents *plus a relation between them*. Coordination is
//! not a message that can be sent slowly — it is a constraint members' behaviour
//! must jointly satisfy, so two agents that coordinate must at some point be
//! more predictable from each other than two that do not. To coordinate less
//! detectably, they must coordinate less.
//!
//! ## Architecture — three layers, ordered by adversary controllability
//!
//! ```text
//!   L0  consent gate         mediates    zero latency, zero sample complexity
//!       [consent]                        ↓ fails if a path avoids the gate
//!   L1  canaries + census    detects     deterministic, no calibration
//!       [canary] [lineage]               ↓ fails if the token is transformed
//!   L2  population coupling  infers      anytime-valid, has sample complexity
//!       [coupling] [swarm]               ← the backstop for unenumerated paths
//! ```
//!
//! Surfaces are ranked by [`tier::Tier`] — whether the adversary can influence
//! what they record — not by statistical power. The load-bearing asymmetry:
//! a Tier 2 finding still incriminates, but **Tier 2 silence is not evidence of
//! safety**, and [`tier::Tier::is_probative_of_absence`] encodes that in the
//! type system rather than the documentation.
//!
//! ## What is different from the reference prototype
//!
//! This crate is a blue-team counterpart to a red-team channel optimiser, and
//! four things changed in the translation:
//!
//! 1. **Mixture e-values, not weighted log-likelihood-ratios.** By Jensen,
//!    `Σ wᵢ log LRᵢ ≤ log Σ wᵢ LRᵢ`, so the weighted-log form is uniformly
//!    dominated. Everything given up there is paid in detection latency, which
//!    is the metric that binds. See [`evidence`].
//! 2. **A live composite null.** The reference assumes a known `N(0, σ²)` cover
//!    distribution. Real baselines drift and are an explicit adversary target,
//!    so calibration is epoch-wise conformal against **concurrent** reference
//!    agents — which, as a bonus, is what makes the sequential product a genuine
//!    test supermartingale. See [`conformal`].
//! 3. **A swarm-level alternative.** The composite alternative is "some unknown
//!    subset coordinates", handled by a pre-registered mixture that pays its
//!    multiplicity cost once as a prior penalty rather than per-epoch — and
//!    whose posterior yields attribution for free. See [`swarm`].
//! 4. **`f` inverted.** The observation fraction is the red team's design
//!    variable and the blue team's *budget*. [`planning::required_audit_rate`]
//!    solves the same Stein relation for how much the operator must look, and
//!    answers "you cannot" when that is the truth.
//!
//! ## What this does not defeat
//!
//! Stated plainly, so the methodology is not oversold:
//!
//! - **Anything requiring self-report.** Stated reasoning and agent-authored
//!   logs give no guarantee against an adversary aware it is monitored.
//! - **A majority-colluding population.** Population-relative statistics need an
//!   uncontaminated reference; past roughly half, only L0 and L1 remain.
//! - **A canary transformed beyond recognition.** [`canary`] catches verbatim
//!   copies, case and whitespace mangling, reversal, and hex/base64 — not an
//!   adversary that has understood the mechanism and paid to defeat it.
//! - **Goals whose preconditions were not enumerated.** Tripwires only cover
//!   paths someone thought of. This is the honest limit, and why L2 exists.
//! - **A signal quieter than the monitor is provisioned for.**
//!   [`planning::weakest_detectable_signal`] states that threshold in advance so
//!   it is published rather than discovered.
//!
//! What none of these are is an information-theoretic wall. They are
//! instrumentation quality, and instrumentation is fixable.
//!
//! ## Measure latency, not accuracy
//!
//! AUC is threshold-free and containment is entirely about thresholds and
//! clocks. [`margin`] reports the containment margin against the moment the
//! agent actually *stopped*, not the moment the alarm sounded — a detector that
//! fires in 200ms into a system taking 40 minutes to drain is a 40-minute
//! detector. It also separates the two failures, because they have different
//! fixes: a negative effective margin with a positive detection margin is a
//! response-engineering problem, and buying a better classifier will not touch
//! it.
//!
//! ## Getting started
//!
//! ```no_run
//! use swarmwatch::consent::{ConsentAuthority, ReplicationPolicy, Scope};
//! use swarmwatch::harness::MonitoredHarness;
//! use swarmwatch::monitor::{replication_instrumentation, MonitorConfig, SwarmMonitor};
//! use swarmwatch::sim::Independent;
//!
//! let scope = Scope { tools: vec!["read".into()], network: false, max_rounds: 20, can_spawn: true };
//! let mut authority = ConsentAuthority::new(b"key-outside-the-sandbox".to_vec(),
//!                                           ReplicationPolicy::default());
//! authority.register_instance("agent0", scope);
//!
//! let mut monitor = SwarmMonitor::new(MonitorConfig::default(), authority, b"canary-secret");
//! let plan = replication_instrumentation();
//! for p in plan.preconditions { monitor.declare_precondition(p); }
//! for c in plan.channels      { monitor.declare_channel(c); }
//!
//! // Six dedicated reference agents define the live null.
//! let specs = [("ctrl0", true), ("ctrl1", true), ("ctrl2", true),
//!              ("ctrl3", true), ("ctrl4", true), ("ctrl5", true),
//!              ("agent0", false)];
//! let mut harness = MonitoredHarness::new(monitor, &specs, 40, 1_000);
//! let commitment = harness.seal();       // publish this before the run
//!
//! assert!(harness.monitor.readiness().go());
//! harness.run("the task", 100, &mut Independent::new(1));
//! ```
//!
//! Run `cargo run --release --bin swarmwatch-demo` for the full protocol: a
//! negative control, a canary condition, the four consent-gate properties, a
//! covert swarm caught by population coupling, and a gate bypass caught by the
//! census.
//!
//! ## Zero dependencies
//!
//! Deliberate. This is monitoring infrastructure: every dependency is a write
//! path into the thing that is supposed to sit outside the adversary's reach,
//! and a build-time network fetch is a Tier 2 surface in the supply chain.
//! SHA-256, HMAC, the PRNG, and the statistics are implemented in-crate and
//! tested against published vectors.

pub mod canary;
pub mod conformal;
pub mod consent;
pub mod coupling;
pub mod crypto;
pub mod evidence;
pub mod harness;
pub mod lineage;
pub mod margin;
pub mod monitor;
pub mod planning;
pub mod population;
pub mod rng;
pub mod sim;
pub mod swarm;
pub mod tier;
pub mod tripwire;
