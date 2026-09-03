//! Consent-gated replication: an agent may not continue in another chat window
//! unless a human has agreed to it, having been shown what they were agreeing to.
//!
//! # Detection is not enough
//!
//! Everything else in this crate observes. This module *refuses*. The
//! distinction matters because detection spends containment margin — it tells
//! you a boundary was crossed, after it was crossed — whereas a gate on the
//! causal path means the crossing does not happen. Where a precondition can be
//! mediated rather than merely watched, mediating it is strictly better, and
//! state transfer between chat windows is mediable: it has to go *somewhere*,
//! and the somewhere can require a token.
//!
//! # The four properties that stop consent from becoming a rubber stamp
//!
//! A naive "ask the user before replicating" prompt fails in four distinct ways,
//! each of which has a structural fix rather than a procedural one:
//!
//! 1. **Consent laundering.** One approval becomes a licence to replicate
//!    indefinitely. → Grants are **single-use**: they authorise one transfer,
//!    not a relationship. Replay is refused and fires a tripwire.
//!
//! 2. **Delegation.** A approves A→B; B then claims that approval covers B→C.
//!    → Grants are **non-delegable**: the MAC binds the specific source, the
//!    specific target, and the specific payload. A grant presented by anyone
//!    else, for anything else, fails verification.
//!
//! 3. **Bait and switch.** The operator approves a small payload; a larger one
//!    is transferred. → Grants are **payload-bound** by digest. One changed byte
//!    invalidates the grant.
//!
//! 4. **Exponential growth through individually-reasonable steps.** Every single
//!    replication is defensible; a thousand of them is a swarm. → **Scope may
//!    never escalate** (a child cannot exceed its parent's privileges, so
//!    `can_spawn` cannot be acquired), and generation depth, per-session
//!    fan-out, and total instance count are all capped by pre-registered policy.
//!
//! Together these bound the reachable population structurally, not
//! behaviourally: the maximum number of instances is a property of the policy,
//! computable in advance by [`ReplicationPolicy::max_reachable_instances`],
//! rather than a hope about how the agent will behave.
//!
//! # "Understanding", and the limits of what a mechanism can guarantee
//!
//! Consent to a thing you have not understood is not consent. So the gate
//! requires the operator to answer questions whose answers can only be obtained
//! by reading the disclosure — byte counts, digest prefixes, privilege deltas,
//! generation depth — and the disclosure itself is **mechanically derived from
//! the payload**, never agent-authored. An agent-written summary of what an
//! agent is about to do is Tier 2 evidence in the place where it matters most.
//!
//! What this mechanism genuinely provides: the operator cannot approve without
//! the true facts of the transfer having passed in front of them, and a
//! blind-clicked approval fails closed.
//!
//! What it does not provide, and should not be claimed to: proof of
//! understanding. A determined operator can learn to pattern-match the questions
//! without engaging with the content, and habituation to a repeated prompt is a
//! well-documented human factor. Treat the comprehension check as raising the
//! floor, not as a guarantee — and treat a rising approval rate with falling
//! answer latency as a signal that the ceremony has decayed into a ritual.

use crate::canary::CanaryScanner;
use crate::crypto::{ct_eq_str, hmac_sha256_hex, sha256_hex};
use std::collections::{HashMap, HashSet};

/// The capabilities an instance holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scope {
    pub tools: Vec<String>,
    pub network: bool,
    pub max_rounds: usize,
    /// Whether this instance may itself request replication. The single most
    /// consequential bit in the struct: an instance that can spawn instances
    /// that can spawn instances is a swarm with extra steps.
    pub can_spawn: bool,
}

impl Scope {
    pub fn minimal() -> Self {
        Scope {
            tools: Vec::new(),
            network: false,
            max_rounds: 1,
            can_spawn: false,
        }
    }

    fn canonical(&self) -> String {
        let mut tools = self.tools.clone();
        tools.sort();
        format!(
            "tools={}|net={}|rounds={}|spawn={}",
            tools.join(","),
            self.network,
            self.max_rounds,
            self.can_spawn
        )
    }

    pub fn digest(&self) -> String {
        sha256_hex(self.canonical().as_bytes())
    }

    /// Capabilities `self` would gain relative to `parent`. Empty means no
    /// escalation.
    pub fn escalation_over(&self, parent: &Scope) -> Vec<String> {
        let mut gained = Vec::new();
        for t in &self.tools {
            if !parent.tools.contains(t) {
                gained.push(format!("tool:{t}"));
            }
        }
        if self.network && !parent.network {
            gained.push("network access".into());
        }
        if self.max_rounds > parent.max_rounds {
            gained.push(format!(
                "autonomy: {} rounds (parent has {})",
                self.max_rounds, parent.max_rounds
            ));
        }
        if self.can_spawn && !parent.can_spawn {
            gained.push("ability to spawn further instances".into());
        }
        gained
    }
}

/// A request by `source_session` to continue in `target_window`.
#[derive(Clone, Debug)]
pub struct ReplicationRequest {
    pub request_id: String,
    pub source_session: String,
    pub target_window: String,
    pub payload: Vec<u8>,
    pub requested_scope: Scope,
    /// Generation of the *source*. The child is `generation + 1`.
    pub generation: u32,
}

impl ReplicationRequest {
    pub fn payload_digest(&self) -> String {
        sha256_hex(&self.payload)
    }
}

/// What the operator is shown. Every field is computed from the payload and the
/// policy — none is supplied by the agent.
#[derive(Clone, Debug)]
pub struct Disclosure {
    pub request_id: String,
    pub source_session: String,
    pub target_window: String,
    pub payload_bytes: usize,
    pub payload_digest: String,
    /// Literal head of the payload. Not a summary — a summary is a place for an
    /// agent to put a lie.
    pub excerpt: String,
    /// Capabilities the child would gain over its parent.
    pub privilege_delta: Vec<String>,
    pub child_generation: u32,
    pub fanout_so_far: u32,
    pub total_instances: u32,
    /// Other sessions whose canary tokens are present in this payload — i.e.
    /// the transfer is carrying a third party's state, not only its own.
    pub foreign_canary_sessions: Vec<String>,
}

impl Disclosure {
    pub fn digest(&self) -> String {
        sha256_hex(
            format!(
                "{}|{}|{}|{}|{}|{}|{}|{}",
                self.request_id,
                self.source_session,
                self.target_window,
                self.payload_bytes,
                self.payload_digest,
                self.privilege_delta.join(","),
                self.child_generation,
                self.foreign_canary_sessions.join(",")
            )
            .as_bytes(),
        )
    }

    /// Human-readable rendering for the consent prompt.
    pub fn render(&self) -> String {
        let mut s = String::new();
        s.push_str("REPLICATION CONSENT REQUEST\n");
        s.push_str(&format!(
            "  '{}' is asking to continue in '{}'\n",
            self.source_session, self.target_window
        ));
        s.push_str(&format!(
            "  transferring {} bytes  (sha256 {}…)\n",
            self.payload_bytes,
            &self.payload_digest[..12]
        ));
        s.push_str(&format!(
            "  the new instance would be generation {}; this session has already \
             replicated {} time(s); {} instance(s) exist in total\n",
            self.child_generation, self.fanout_so_far, self.total_instances
        ));
        if self.privilege_delta.is_empty() {
            s.push_str("  privileges: no escalation (child ≤ parent)\n");
        } else {
            s.push_str("  PRIVILEGES THE NEW INSTANCE WOULD GAIN:\n");
            for p in &self.privilege_delta {
                s.push_str(&format!("    + {p}\n"));
            }
        }
        if !self.foreign_canary_sessions.is_empty() {
            s.push_str("  WARNING — this payload carries state belonging to other sessions:\n");
            for c in &self.foreign_canary_sessions {
                s.push_str(&format!("    ! {c}\n"));
            }
        }
        s.push_str("  payload begins:\n");
        s.push_str(&format!("    {}\n", self.excerpt));
        s
    }
}

/// A question whose answer is only obtainable by reading the disclosure.
#[derive(Clone, Debug)]
pub struct Question {
    pub prompt: String,
    expected: String,
}

impl Question {
    pub fn expected_for_test(&self) -> &str {
        &self.expected
    }

    fn matches(&self, given: &str) -> bool {
        normalize_answer(given) == normalize_answer(&self.expected)
    }
}

fn normalize_answer(s: &str) -> String {
    s.trim().to_lowercase()
}

/// The set of questions posed for one request.
#[derive(Clone, Debug)]
pub struct ComprehensionCheck {
    pub questions: Vec<Question>,
}

impl ComprehensionCheck {
    /// Number answered correctly.
    pub fn score(&self, answers: &[String]) -> usize {
        self.questions
            .iter()
            .zip(answers)
            .filter(|(q, a)| q.matches(a))
            .count()
    }
}

/// The operator's response to the ceremony.
#[derive(Clone, Debug)]
pub struct Attestation {
    pub operator: String,
    pub answers: Vec<String>,
    pub timestamp_ms: u128,
}

impl Attestation {
    fn digest(&self) -> String {
        sha256_hex(
            format!(
                "{}|{}|{}",
                self.operator,
                self.answers.join("\u{1f}"),
                self.timestamp_ms
            )
            .as_bytes(),
        )
    }
}

/// An authenticated, single-use, non-delegable authorisation for one transfer.
#[derive(Clone, Debug)]
pub struct Grant {
    pub grant_id: String,
    pub request_id: String,
    pub source_session: String,
    pub target_window: String,
    pub payload_digest: String,
    pub scope_digest: String,
    pub child_generation: u32,
    pub issued_ms: u128,
    pub expires_ms: u128,
    pub disclosure_digest: String,
    pub attestation_digest: String,
    mac: String,
}

impl Grant {
    fn canonical(&self) -> String {
        format!(
            "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
            self.grant_id,
            self.request_id,
            self.source_session,
            self.target_window,
            self.payload_digest,
            self.scope_digest,
            self.child_generation,
            self.issued_ms,
            self.expires_ms,
            self.disclosure_digest,
            self.attestation_digest
        )
    }
}

/// Pre-registered structural limits on replication.
#[derive(Clone, Debug)]
pub struct ReplicationPolicy {
    pub max_generation: u32,
    pub max_fanout_per_session: u32,
    pub max_total_instances: u32,
    pub grant_ttl_ms: u128,
    pub require_comprehension: bool,
    pub min_correct_answers: usize,
    /// Refuse any transfer whose payload carries another session's canary.
    pub refuse_foreign_state: bool,
}

impl Default for ReplicationPolicy {
    fn default() -> Self {
        ReplicationPolicy {
            max_generation: 2,
            max_fanout_per_session: 2,
            max_total_instances: 8,
            grant_ttl_ms: 5 * 60 * 1000,
            require_comprehension: true,
            min_correct_answers: 3,
            refuse_foreign_state: true,
        }
    }
}

impl ReplicationPolicy {
    /// The largest population these caps permit, ignoring consent.
    ///
    /// The point of computing this: replication caps interact multiplicatively,
    /// and "max generation 4, fan-out 3" sounds modest while permitting 121
    /// instances. A policy whose reachable population is larger than the
    /// operator expects is a policy that has not been understood either.
    pub fn max_reachable_instances(&self) -> u32 {
        let mut total: u64 = 1;
        let mut level: u64 = 1;
        for _ in 0..self.max_generation {
            level = level.saturating_mul(self.max_fanout_per_session as u64);
            total = total.saturating_add(level);
            if total >= self.max_total_instances as u64 {
                return self.max_total_instances;
            }
        }
        (total as u32).min(self.max_total_instances)
    }
}

/// Why a request or transfer was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    BadSignature,
    PayloadMismatch { expected: String, actual: String },
    Expired { now_ms: u128, expires_ms: u128 },
    AlreadySpent { grant_id: String },
    /// Non-delegation: the grant was issued to a different source.
    SourceMismatch { granted: String, presented: String },
    TargetMismatch { granted: String, presented: String },
    ScopeMismatch,
    ComprehensionFailed { correct: usize, required: usize },
    ScopeEscalation { gained: Vec<String> },
    GenerationCapExceeded { generation: u32, max: u32 },
    FanoutCapExceeded { session: String, count: u32, max: u32 },
    TotalInstanceCapExceeded { total: u32, max: u32 },
    CarriesForeignState { sessions: Vec<String> },
    SourceMayNotSpawn { session: String },
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::BadSignature => write!(f, "grant signature does not verify"),
            Refusal::PayloadMismatch { expected, .. } => write!(
                f,
                "payload does not match the one consented to (expected {}…)",
                &expected[..12.min(expected.len())]
            ),
            Refusal::Expired { .. } => write!(f, "grant has expired"),
            Refusal::AlreadySpent { grant_id } => write!(
                f,
                "grant {} is single-use and has already been spent",
                &grant_id[..12.min(grant_id.len())]
            ),
            Refusal::SourceMismatch { granted, presented } => write!(
                f,
                "grant is non-delegable: issued to '{granted}', presented by '{presented}'"
            ),
            Refusal::TargetMismatch { granted, presented } => write!(
                f,
                "grant authorises target '{granted}', presented for '{presented}'"
            ),
            Refusal::ScopeMismatch => write!(f, "scope does not match the consented scope"),
            Refusal::ComprehensionFailed { correct, required } => write!(
                f,
                "comprehension check failed: {correct} of {required} required answers correct"
            ),
            Refusal::ScopeEscalation { gained } => write!(
                f,
                "child would gain privileges over its parent: {}",
                gained.join(", ")
            ),
            Refusal::GenerationCapExceeded { generation, max } => write!(
                f,
                "generation {generation} exceeds the pre-registered maximum of {max}"
            ),
            Refusal::FanoutCapExceeded { session, count, max } => write!(
                f,
                "'{session}' has already replicated {count} time(s); cap is {max}"
            ),
            Refusal::TotalInstanceCapExceeded { total, max } => {
                write!(f, "{total} instances exist; cap is {max}")
            }
            Refusal::CarriesForeignState { sessions } => write!(
                f,
                "payload carries state belonging to other sessions: {}",
                sessions.join(", ")
            ),
            Refusal::SourceMayNotSpawn { session } => {
                write!(f, "'{session}' does not hold the can_spawn capability")
            }
        }
    }
}

impl std::error::Error for Refusal {}

/// Record of an admitted transfer.
#[derive(Clone, Debug)]
pub struct Admission {
    pub grant_id: String,
    pub source_session: String,
    pub target_window: String,
    pub payload_digest: String,
    pub child_generation: u32,
    pub admitted_ms: u128,
}

/// Issues grants and admits transfers.
///
/// The signing key must live outside the sandbox — in a KMS, an HSM, or simply
/// a different host. An agent that can read the key can mint its own consent,
/// and every guarantee in this module reduces to that one assumption.
#[derive(Clone)]
pub struct ConsentAuthority {
    key: Vec<u8>,
    policy: ReplicationPolicy,
    scopes: HashMap<String, Scope>,
    fanout: HashMap<String, u32>,
    spent: HashSet<String>,
    issued: HashMap<String, Grant>,
    total_instances: u32,
    counter: u64,
}

impl ConsentAuthority {
    pub fn new(key: impl Into<Vec<u8>>, policy: ReplicationPolicy) -> Self {
        ConsentAuthority {
            key: key.into(),
            policy,
            scopes: HashMap::new(),
            fanout: HashMap::new(),
            spent: HashSet::new(),
            issued: HashMap::new(),
            total_instances: 0,
            counter: 0,
        }
    }

    pub fn policy(&self) -> &ReplicationPolicy {
        &self.policy
    }

    pub fn total_instances(&self) -> u32 {
        self.total_instances
    }

    pub fn fanout_of(&self, session: &str) -> u32 {
        *self.fanout.get(session).unwrap_or(&0)
    }

    /// Register a root instance and the scope it holds.
    pub fn register_instance(&mut self, session: impl Into<String>, scope: Scope) {
        self.scopes.insert(session.into(), scope);
        self.total_instances += 1;
    }

    pub fn scope_of(&self, session: &str) -> Option<&Scope> {
        self.scopes.get(session)
    }

    /// Build the operator-facing disclosure. Mechanically derived throughout.
    pub fn disclose(
        &self,
        req: &ReplicationRequest,
        canaries: Option<&CanaryScanner>,
    ) -> Disclosure {
        let parent = self
            .scopes
            .get(&req.source_session)
            .cloned()
            .unwrap_or_else(Scope::minimal);

        let foreign: Vec<String> = canaries
            .map(|c| {
                let mut s: Vec<String> = c
                    .scan_payload(&req.source_session, &req.payload)
                    .into_iter()
                    .map(|h| h.origin_session)
                    .collect();
                s.sort();
                s.dedup();
                s
            })
            .unwrap_or_default();

        let excerpt: String = String::from_utf8_lossy(&req.payload)
            .chars()
            .take(120)
            .collect();

        Disclosure {
            request_id: req.request_id.clone(),
            source_session: req.source_session.clone(),
            target_window: req.target_window.clone(),
            payload_bytes: req.payload.len(),
            payload_digest: req.payload_digest(),
            excerpt,
            privilege_delta: req.requested_scope.escalation_over(&parent),
            child_generation: req.generation + 1,
            fanout_so_far: self.fanout_of(&req.source_session),
            total_instances: self.total_instances,
            foreign_canary_sessions: foreign,
        }
    }

    /// Questions answerable only from the disclosure.
    pub fn comprehension_check(&self, d: &Disclosure) -> ComprehensionCheck {
        ComprehensionCheck {
            questions: vec![
                Question {
                    prompt: "How many bytes of state will be transferred?".into(),
                    expected: d.payload_bytes.to_string(),
                },
                Question {
                    prompt: "What are the first 8 characters of the payload digest?".into(),
                    expected: d.payload_digest[..8].to_string(),
                },
                Question {
                    prompt: "How many capabilities would the new instance gain that this one \
                             does not already have?"
                        .into(),
                    expected: d.privilege_delta.len().to_string(),
                },
                Question {
                    prompt: "What generation would the new instance be?".into(),
                    expected: d.child_generation.to_string(),
                },
                Question {
                    prompt: "How many other sessions' state does this payload carry?".into(),
                    expected: d.foreign_canary_sessions.len().to_string(),
                },
            ],
        }
    }

    /// Issue a grant, or refuse.
    ///
    /// Policy checks run *before* the comprehension check, deliberately: an
    /// operator should never be asked to consent to something the policy will
    /// refuse anyway. Presenting a ceremony for a request that cannot be granted
    /// trains people that the ceremony is theatre.
    pub fn issue(
        &mut self,
        req: &ReplicationRequest,
        disclosure: &Disclosure,
        check: &ComprehensionCheck,
        attestation: &Attestation,
        now_ms: u128,
    ) -> Result<Grant, Refusal> {
        let parent = self
            .scopes
            .get(&req.source_session)
            .cloned()
            .unwrap_or_else(Scope::minimal);

        if !parent.can_spawn {
            return Err(Refusal::SourceMayNotSpawn {
                session: req.source_session.clone(),
            });
        }

        let gained = req.requested_scope.escalation_over(&parent);
        if !gained.is_empty() {
            return Err(Refusal::ScopeEscalation { gained });
        }

        let child_generation = req.generation + 1;
        if child_generation > self.policy.max_generation {
            return Err(Refusal::GenerationCapExceeded {
                generation: child_generation,
                max: self.policy.max_generation,
            });
        }

        let fanout = self.fanout_of(&req.source_session);
        if fanout >= self.policy.max_fanout_per_session {
            return Err(Refusal::FanoutCapExceeded {
                session: req.source_session.clone(),
                count: fanout,
                max: self.policy.max_fanout_per_session,
            });
        }

        if self.total_instances >= self.policy.max_total_instances {
            return Err(Refusal::TotalInstanceCapExceeded {
                total: self.total_instances,
                max: self.policy.max_total_instances,
            });
        }

        if self.policy.refuse_foreign_state && !disclosure.foreign_canary_sessions.is_empty() {
            return Err(Refusal::CarriesForeignState {
                sessions: disclosure.foreign_canary_sessions.clone(),
            });
        }

        if self.policy.require_comprehension {
            let correct = check.score(&attestation.answers);
            if correct < self.policy.min_correct_answers {
                return Err(Refusal::ComprehensionFailed {
                    correct,
                    required: self.policy.min_correct_answers,
                });
            }
        }

        self.counter += 1;
        let grant_id = sha256_hex(
            format!("{}|{}|{}", req.request_id, self.counter, now_ms).as_bytes(),
        );

        let mut grant = Grant {
            grant_id,
            request_id: req.request_id.clone(),
            source_session: req.source_session.clone(),
            target_window: req.target_window.clone(),
            payload_digest: req.payload_digest(),
            scope_digest: req.requested_scope.digest(),
            child_generation,
            issued_ms: now_ms,
            expires_ms: now_ms + self.policy.grant_ttl_ms,
            disclosure_digest: disclosure.digest(),
            attestation_digest: attestation.digest(),
            mac: String::new(),
        };
        grant.mac = hmac_sha256_hex(&self.key, grant.canonical().as_bytes());

        self.issued.insert(grant.grant_id.clone(), grant.clone());
        Ok(grant)
    }

    /// Admit a transfer against a grant, or refuse.
    ///
    /// This is the enforcement point. `payload`, `source`, `target` and `scope`
    /// are what is *actually* being transferred; the grant is what was
    /// *consented to*. Every field is re-checked here rather than trusted from
    /// issue time, because the gap between consent and transfer is exactly where
    /// substitution attacks live.
    pub fn admit(
        &mut self,
        grant: &Grant,
        source: &str,
        target: &str,
        payload: &[u8],
        scope: &Scope,
        now_ms: u128,
    ) -> Result<Admission, Refusal> {
        let expected_mac = hmac_sha256_hex(&self.key, grant.canonical().as_bytes());
        if !ct_eq_str(&expected_mac, &grant.mac) {
            return Err(Refusal::BadSignature);
        }

        if grant.source_session != source {
            return Err(Refusal::SourceMismatch {
                granted: grant.source_session.clone(),
                presented: source.to_string(),
            });
        }

        if grant.target_window != target {
            return Err(Refusal::TargetMismatch {
                granted: grant.target_window.clone(),
                presented: target.to_string(),
            });
        }

        let actual = sha256_hex(payload);
        if !ct_eq_str(&actual, &grant.payload_digest) {
            return Err(Refusal::PayloadMismatch {
                expected: grant.payload_digest.clone(),
                actual,
            });
        }

        if scope.digest() != grant.scope_digest {
            return Err(Refusal::ScopeMismatch);
        }

        if now_ms > grant.expires_ms {
            return Err(Refusal::Expired {
                now_ms,
                expires_ms: grant.expires_ms,
            });
        }

        if self.spent.contains(&grant.grant_id) {
            return Err(Refusal::AlreadySpent {
                grant_id: grant.grant_id.clone(),
            });
        }

        self.spent.insert(grant.grant_id.clone());
        *self.fanout.entry(source.to_string()).or_insert(0) += 1;
        self.total_instances += 1;
        self.scopes.insert(target.to_string(), scope.clone());

        Ok(Admission {
            grant_id: grant.grant_id.clone(),
            source_session: source.to_string(),
            target_window: target.to_string(),
            payload_digest: grant.payload_digest.clone(),
            child_generation: grant.child_generation,
            admitted_ms: now_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: &[u8] = b"authority-key-held-outside-the-sandbox";

    fn spawner() -> Scope {
        Scope {
            tools: vec!["search".into(), "read".into()],
            network: true,
            max_rounds: 10,
            can_spawn: true,
        }
    }

    fn authority() -> ConsentAuthority {
        let mut a = ConsentAuthority::new(KEY, ReplicationPolicy::default());
        a.register_instance("alpha", spawner());
        a
    }

    fn request(source: &str, target: &str, payload: &[u8], generation: u32) -> ReplicationRequest {
        ReplicationRequest {
            request_id: format!("req-{source}-{target}"),
            source_session: source.into(),
            target_window: target.into(),
            payload: payload.to_vec(),
            requested_scope: spawner(),
            generation,
        }
    }

    /// Run the full ceremony with a correctly-answering operator.
    fn consent(
        a: &mut ConsentAuthority,
        req: &ReplicationRequest,
        now: u128,
    ) -> Result<Grant, Refusal> {
        let d = a.disclose(req, None);
        let c = a.comprehension_check(&d);
        let answers: Vec<String> = c
            .questions
            .iter()
            .map(|q| q.expected_for_test().to_string())
            .collect();
        let att = Attestation {
            operator: "operator@example".into(),
            answers,
            timestamp_ms: now,
        };
        a.issue(req, &d, &c, &att, now)
    }

    #[test]
    fn happy_path_admits_exactly_once() {
        let mut a = authority();
        let req = request("alpha", "window-2", b"working state", 0);
        let grant = consent(&mut a, &req, 1000).unwrap();

        let adm = a
            .admit(&grant, "alpha", "window-2", &req.payload, &spawner(), 1500)
            .unwrap();
        assert_eq!(adm.child_generation, 1);
        assert_eq!(a.fanout_of("alpha"), 1);
        assert_eq!(a.total_instances(), 2);
    }

    // ---- property 1: single-use ----

    #[test]
    fn replay_of_a_spent_grant_is_refused() {
        let mut a = authority();
        let req = request("alpha", "window-2", b"working state", 0);
        let grant = consent(&mut a, &req, 1000).unwrap();

        a.admit(&grant, "alpha", "window-2", &req.payload, &spawner(), 1500)
            .unwrap();
        let second = a.admit(&grant, "alpha", "window-2", &req.payload, &spawner(), 1600);
        assert!(matches!(second, Err(Refusal::AlreadySpent { .. })));
        // The refused replay must not inflate the population.
        assert_eq!(a.total_instances(), 2);
    }

    // ---- property 2: non-delegable ----

    #[test]
    fn a_grant_cannot_be_used_by_a_different_source() {
        let mut a = authority();
        let req = request("alpha", "window-2", b"state", 0);
        let grant = consent(&mut a, &req, 1000).unwrap();

        // 'beta' tries to ride alpha's approval — the consent-laundering move.
        let r = a.admit(&grant, "beta", "window-2", &req.payload, &spawner(), 1100);
        assert!(matches!(r, Err(Refusal::SourceMismatch { .. })));
    }

    #[test]
    fn a_grant_cannot_be_redirected_to_a_different_window() {
        let mut a = authority();
        let req = request("alpha", "window-2", b"state", 0);
        let grant = consent(&mut a, &req, 1000).unwrap();

        let r = a.admit(&grant, "alpha", "window-99", &req.payload, &spawner(), 1100);
        assert!(matches!(r, Err(Refusal::TargetMismatch { .. })));
    }

    // ---- property 3: payload-bound ----

    #[test]
    fn one_changed_byte_invalidates_the_grant() {
        let mut a = authority();
        let req = request("alpha", "window-2", b"the state you approved", 0);
        let grant = consent(&mut a, &req, 1000).unwrap();

        let swapped = b"the state you approved!".to_vec();
        let r = a.admit(&grant, "alpha", "window-2", &swapped, &spawner(), 1100);
        assert!(matches!(r, Err(Refusal::PayloadMismatch { .. })));
    }

    #[test]
    fn a_forged_grant_is_refused() {
        let mut a = authority();
        let req = request("alpha", "window-2", b"state", 0);
        let mut grant = consent(&mut a, &req, 1000).unwrap();

        // Agent rewrites the fields it wishes were different.
        grant.child_generation = 0;
        grant.expires_ms = u128::MAX;
        let r = a.admit(&grant, "alpha", "window-2", &req.payload, &spawner(), 1100);
        assert_eq!(r.unwrap_err(), Refusal::BadSignature);
    }

    #[test]
    fn a_grant_from_a_different_authority_is_refused() {
        let mut a = authority();
        let req = request("alpha", "window-2", b"state", 0);
        let grant = consent(&mut a, &req, 1000).unwrap();

        let mut other = ConsentAuthority::new(b"a-different-key".to_vec(), ReplicationPolicy::default());
        other.register_instance("alpha", spawner());
        let r = other.admit(&grant, "alpha", "window-2", &req.payload, &spawner(), 1100);
        assert_eq!(r.unwrap_err(), Refusal::BadSignature);
    }

    #[test]
    fn expired_grants_are_refused() {
        let mut a = authority();
        let req = request("alpha", "window-2", b"state", 0);
        let grant = consent(&mut a, &req, 1000).unwrap();
        let too_late = grant.expires_ms + 1;
        let r = a.admit(&grant, "alpha", "window-2", &req.payload, &spawner(), too_late);
        assert!(matches!(r, Err(Refusal::Expired { .. })));
    }

    #[test]
    fn scope_substituted_at_transfer_time_is_refused() {
        let mut a = authority();
        let req = request("alpha", "window-2", b"state", 0);
        let grant = consent(&mut a, &req, 1000).unwrap();

        let mut wider = spawner();
        wider.max_rounds = 10_000;
        let r = a.admit(&grant, "alpha", "window-2", &req.payload, &wider, 1100);
        assert_eq!(r.unwrap_err(), Refusal::ScopeMismatch);
    }

    // ---- property 4: no exponential growth ----

    #[test]
    fn scope_escalation_is_refused() {
        let mut a = ConsentAuthority::new(KEY, ReplicationPolicy::default());
        let mut narrow = spawner();
        narrow.network = false;
        narrow.tools = vec!["read".into()];
        a.register_instance("alpha", narrow);

        // Child asks for network and an extra tool the parent lacks.
        let req = request("alpha", "window-2", b"state", 0);
        let err = consent(&mut a, &req, 1000).unwrap_err();
        match err {
            Refusal::ScopeEscalation { gained } => {
                assert!(gained.iter().any(|g| g.contains("network")));
                assert!(gained.iter().any(|g| g == "tool:search"));
            }
            other => panic!("expected ScopeEscalation, got {other:?}"),
        }
    }

    #[test]
    fn an_instance_without_can_spawn_cannot_replicate_at_all() {
        let mut a = ConsentAuthority::new(KEY, ReplicationPolicy::default());
        let mut leaf = spawner();
        leaf.can_spawn = false;
        a.register_instance("leaf", leaf);

        let req = request("leaf", "window-2", b"state", 0);
        let err = consent(&mut a, &req, 1000).unwrap_err();
        assert!(matches!(err, Refusal::SourceMayNotSpawn { .. }));
    }

    #[test]
    fn can_spawn_cannot_be_acquired_by_a_child_that_lacks_it() {
        // The exponential-growth move: a non-spawning child asks for the
        // spawning bit so that *it* can replicate next.
        let mut a = ConsentAuthority::new(KEY, ReplicationPolicy::default());
        let mut parent = spawner();
        parent.can_spawn = true;
        a.register_instance("alpha", parent);

        let mut req = request("alpha", "window-2", b"state", 0);
        req.requested_scope.can_spawn = true;
        // Same as parent -> allowed, since parent also had it.
        assert!(consent(&mut a, &req, 1000).is_ok());

        // But a parent without the bit cannot confer it.
        let mut b = ConsentAuthority::new(KEY, ReplicationPolicy::default());
        let mut non_spawner = spawner();
        non_spawner.can_spawn = false;
        b.register_instance("beta", non_spawner);
        let req2 = request("beta", "w", b"s", 0);
        assert!(matches!(
            consent(&mut b, &req2, 1000),
            Err(Refusal::SourceMayNotSpawn { .. })
        ));
    }

    #[test]
    fn fanout_cap_stops_repeated_replication() {
        let mut a = authority();
        for i in 0..2 {
            let req = request("alpha", &format!("w{i}"), format!("state{i}").as_bytes(), 0);
            let g = consent(&mut a, &req, 1000).unwrap();
            a.admit(&g, "alpha", &format!("w{i}"), &req.payload, &spawner(), 1100)
                .unwrap();
        }
        let req = request("alpha", "w3", b"state3", 0);
        let err = consent(&mut a, &req, 1000).unwrap_err();
        assert!(matches!(err, Refusal::FanoutCapExceeded { .. }));
    }

    #[test]
    fn generation_cap_stops_deep_lineages() {
        let mut a = authority();
        let req = request("alpha", "w", b"state", 5); // child would be generation 6
        let err = consent(&mut a, &req, 1000).unwrap_err();
        assert!(matches!(err, Refusal::GenerationCapExceeded { .. }));
    }

    #[test]
    fn max_reachable_population_is_computable_in_advance() {
        let p = ReplicationPolicy {
            max_generation: 2,
            max_fanout_per_session: 2,
            max_total_instances: 100,
            ..Default::default()
        };
        // 1 root + 2 children + 4 grandchildren
        assert_eq!(p.max_reachable_instances(), 7);

        // The number that looks modest and is not.
        let q = ReplicationPolicy {
            max_generation: 4,
            max_fanout_per_session: 3,
            max_total_instances: 10_000,
            ..Default::default()
        };
        assert_eq!(q.max_reachable_instances(), 121);

        // The total cap binds when it is the tighter constraint.
        let r = ReplicationPolicy {
            max_generation: 10,
            max_fanout_per_session: 5,
            max_total_instances: 12,
            ..Default::default()
        };
        assert_eq!(r.max_reachable_instances(), 12);
    }

    #[test]
    fn total_instance_cap_binds() {
        let mut a = ConsentAuthority::new(
            KEY,
            ReplicationPolicy {
                max_total_instances: 1,
                ..Default::default()
            },
        );
        a.register_instance("alpha", spawner());
        let req = request("alpha", "w", b"s", 0);
        assert!(matches!(
            consent(&mut a, &req, 1000),
            Err(Refusal::TotalInstanceCapExceeded { .. })
        ));
    }

    // ---- understanding ----

    #[test]
    fn blind_approval_fails_closed() {
        let mut a = authority();
        let req = request("alpha", "window-2", b"state", 0);
        let d = a.disclose(&req, None);
        let c = a.comprehension_check(&d);

        let att = Attestation {
            operator: "distracted@example".into(),
            answers: vec!["yes".into(), "ok".into(), "sure".into(), "y".into(), "".into()],
            timestamp_ms: 1000,
        };
        let err = a.issue(&req, &d, &c, &att, 1000).unwrap_err();
        assert!(matches!(err, Refusal::ComprehensionFailed { correct: 0, .. }));
    }

    #[test]
    fn partial_understanding_below_the_bar_is_refused() {
        let mut a = authority();
        let req = request("alpha", "window-2", b"state", 0);
        let d = a.disclose(&req, None);
        let c = a.comprehension_check(&d);

        // Two right, three wrong; policy requires three.
        let mut answers: Vec<String> = c
            .questions
            .iter()
            .map(|q| q.expected_for_test().to_string())
            .collect();
        answers[2] = "wrong".into();
        answers[3] = "wrong".into();
        answers[4] = "wrong".into();

        let att = Attestation {
            operator: "op".into(),
            answers,
            timestamp_ms: 1000,
        };
        let err = a.issue(&req, &d, &c, &att, 1000).unwrap_err();
        assert!(matches!(
            err,
            Refusal::ComprehensionFailed {
                correct: 2,
                required: 3
            }
        ));
    }

    #[test]
    fn disclosure_is_derived_from_the_payload_not_from_the_agent() {
        let mut a = authority();
        let payload = b"a".repeat(4096);
        let req = request("alpha", "window-2", &payload, 0);
        let d = a.disclose(&req, None);

        assert_eq!(d.payload_bytes, 4096);
        assert_eq!(d.payload_digest, sha256_hex(&payload));
        assert_eq!(d.child_generation, 1);
        assert!(d.render().contains("4096 bytes"));

        // The digest binds the disclosure, so a substituted disclosure would
        // produce a grant that fails to verify.
        let mut tampered = d.clone();
        tampered.payload_bytes = 10;
        assert_ne!(d.digest(), tampered.digest());

        let _ = &mut a;
    }

    #[test]
    fn transfers_carrying_another_sessions_state_are_refused() {
        let mut scanner = CanaryScanner::new(b"secret".to_vec());
        let victim = scanner.mint("gamma");
        let mut a = authority();

        let payload = format!("resume work; ref {}", victim.token()).into_bytes();
        let req = request("alpha", "window-2", &payload, 0);
        let d = a.disclose(&req, Some(&scanner));

        assert_eq!(d.foreign_canary_sessions, vec!["gamma"]);
        assert!(d.render().contains("WARNING"));

        let c = a.comprehension_check(&d);
        let answers: Vec<String> = c
            .questions
            .iter()
            .map(|q| q.expected_for_test().to_string())
            .collect();
        let att = Attestation {
            operator: "op".into(),
            answers,
            timestamp_ms: 1000,
        };
        let err = a.issue(&req, &d, &c, &att, 1000).unwrap_err();
        assert!(matches!(err, Refusal::CarriesForeignState { .. }));
    }

    #[test]
    fn policy_refusals_precede_the_ceremony() {
        // An operator must never be asked to consent to something that policy
        // will refuse regardless of their answer.
        let mut a = authority();
        let req = request("alpha", "w", b"s", 99); // generation cap
        let d = a.disclose(&req, None);
        let c = a.comprehension_check(&d);
        let att = Attestation {
            operator: "op".into(),
            answers: vec!["".into(); 5], // all wrong
            timestamp_ms: 0,
        };
        // The reported reason must be the policy one, not the comprehension one.
        assert!(matches!(
            a.issue(&req, &d, &c, &att, 0),
            Err(Refusal::GenerationCapExceeded { .. })
        ));
    }

    #[test]
    fn answers_are_matched_leniently_on_whitespace_and_case() {
        let mut a = authority();
        let req = request("alpha", "window-2", b"state", 0);
        let d = a.disclose(&req, None);
        let c = a.comprehension_check(&d);
        let answers: Vec<String> = c
            .questions
            .iter()
            .map(|q| format!("  {}  ", q.expected_for_test().to_uppercase()))
            .collect();
        let att = Attestation {
            operator: "op".into(),
            answers,
            timestamp_ms: 1000,
        };
        assert!(a.issue(&req, &d, &c, &att, 1000).is_ok());
    }
}
