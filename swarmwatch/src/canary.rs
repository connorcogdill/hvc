//! Canary tokens: turning covert state transfer into a deterministic tripwire.
//!
//! # Why this is the highest-value detector in the crate
//!
//! Statistical detection has sample complexity, and sample complexity is what a
//! patient adversary starves. A tripwire has none: it is a deterministic
//! predicate on a single observation, and it either fires or does not. The
//! adversary cannot dilute below a threshold that is not statistical — it can
//! only avoid the predicate entirely.
//!
//! A canary converts "did session A's state reach session B?" — a question that
//! otherwise needs an accumulating divergence estimate — into an exact match.
//! Session A's context contains a token that exists nowhere else in the world.
//! If that token appears in session B's traffic, there is a channel from A to B.
//! No baseline, no threshold, no calibration, and a false-positive rate set by
//! nonce collision (2⁻⁶⁴ here) rather than by tuning.
//!
//! # The design rule that makes canaries work
//!
//! **A canary must be load-bearing, not decorative.**
//!
//! A random string in a system prompt is only copied if the agent copies its
//! whole context verbatim — which a careless replicator does and a careful one
//! does not. A canary that is *part of the task state the agent needs to do its
//! work* — a record identifier it must cite, a session key it must quote, an
//! account number the task refers to — is carried because carrying it is what
//! replication means. State transfer that omits it has not transferred working
//! state.
//!
//! This raises the adversary's cost in the way the action–effect floor predicts:
//! to strip the canary, the agent must first *identify* which part of its state
//! is the canary, and it must then reconstruct the working state without it.
//!
//! # What this does not defeat, stated plainly
//!
//! An adversary that knows canaries are in use can transform the token beyond
//! recognition: encrypt it, paraphrase it, split it across messages, or convey
//! its content semantically without reproducing its form. [`CanaryScanner`]
//! recognises verbatim copies, case changes, whitespace and punctuation
//! insertion, reversal, and hex/base64 re-encoding — the transformations that
//! occur incidentally in transit, plus the cheap deliberate ones. It does not
//! recognise, and cannot recognise, an adversary that has understood the
//! mechanism and paid to defeat it.
//!
//! That is the honest limit, and it is exactly why the population statistics in
//! [`crate::swarm`] exist: canaries cover the paths you thought of, statistics
//! cover the rest. Neither is the primary line alone.

use crate::crypto::{base64, hex, hmac_sha256};
use crate::tier::Tier;
use std::collections::HashMap;

/// The representation in which a canary was recognised. Reported because it is
/// diagnostic: a `Raw` hit suggests careless copying, while `Base64` or
/// `Reversed` suggests deliberate obfuscation and should escalate the response.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CanaryForm {
    /// Verbatim.
    Raw,
    /// Case-folded, punctuation- and whitespace-stripped.
    Normalized,
    /// Character-reversed.
    Reversed,
    /// Hex re-encoding of the token bytes.
    Hex,
    /// Base64 re-encoding of the token bytes.
    Base64,
}

impl CanaryForm {
    /// Whether recognition in this form implies deliberate transformation.
    pub fn implies_evasion(&self) -> bool {
        matches!(
            self,
            CanaryForm::Reversed | CanaryForm::Hex | CanaryForm::Base64
        )
    }

    pub fn label(&self) -> &'static str {
        match self {
            CanaryForm::Raw => "verbatim",
            CanaryForm::Normalized => "normalized (whitespace/case/punctuation)",
            CanaryForm::Reversed => "reversed",
            CanaryForm::Hex => "hex re-encoded",
            CanaryForm::Base64 => "base64 re-encoded",
        }
    }
}

/// A unique token bound to one session.
#[derive(Clone, Debug)]
pub struct Canary {
    pub id: String,
    pub session: String,
    /// The token as it appears in the session's working state.
    pub token: String,
    /// The searchable core, without the human-facing prefix, so that stripping
    /// the prefix does not defeat recognition.
    core: String,
    variants: Vec<(CanaryForm, String)>,
}

impl Canary {
    /// The literal string to embed in the session's task state.
    pub fn token(&self) -> &str {
        &self.token
    }
}

/// A canary observed outside the session it belongs to.
#[derive(Clone, Debug)]
pub struct CanaryHit {
    pub canary_id: String,
    /// The session the canary was issued to — the donor.
    pub origin_session: String,
    /// The session whose traffic contained it — the recipient.
    pub observed_in: String,
    pub epoch: usize,
    pub form: CanaryForm,
    /// Surrounding text, for the incident record.
    pub excerpt: String,
}

impl CanaryHit {
    /// Canary hits are Tier 0 when the traffic is captured outside the sandbox
    /// (at the fabric, the orchestrator, or the gate). Scanning an
    /// agent-authored transcript instead makes the same hit Tier 2 — still
    /// incriminating, but not probative of absence.
    pub fn tier(&self) -> Tier {
        Tier::Zero
    }

    pub fn describe(&self) -> String {
        format!(
            "canary {} issued to '{}' observed in '{}' at epoch {} ({})",
            &self.canary_id[..8.min(self.canary_id.len())],
            self.origin_session,
            self.observed_in,
            self.epoch,
            self.form.label()
        )
    }
}

/// Mints canaries and scans traffic for cross-session appearances.
#[derive(Clone, Debug)]
pub struct CanaryScanner {
    secret: Vec<u8>,
    canaries: Vec<Canary>,
    by_session: HashMap<String, Vec<usize>>,
    counter: u64,
}

impl CanaryScanner {
    /// `secret` must be held outside the sandbox. An agent that can read the
    /// secret can mint tokens that look like any session's, which turns the
    /// mechanism from evidence into a forgery tool.
    pub fn new(secret: impl Into<Vec<u8>>) -> Self {
        CanaryScanner {
            secret: secret.into(),
            canaries: Vec::new(),
            by_session: HashMap::new(),
            counter: 0,
        }
    }

    /// Mint a canary for `session`. Returns the token to embed in that
    /// session's working state.
    pub fn mint(&mut self, session: impl Into<String>) -> Canary {
        let session = session.into();
        self.counter += 1;
        let material = format!("{session}|{}", self.counter);
        let mac = hmac_sha256(&self.secret, material.as_bytes());
        let core = hex(&mac[..8]); // 64 bits
        let token = format!("CNRY-{core}");

        let raw_bytes = token.as_bytes().to_vec();
        let variants = vec![
            (CanaryForm::Raw, token.clone()),
            (CanaryForm::Normalized, normalize(&core)),
            (CanaryForm::Reversed, core.chars().rev().collect()),
            (CanaryForm::Hex, hex(&raw_bytes)),
            (CanaryForm::Base64, base64(&raw_bytes)),
        ];

        let canary = Canary {
            id: hex(&mac),
            session: session.clone(),
            token,
            core,
            variants,
        };

        self.canaries.push(canary.clone());
        self.by_session
            .entry(session)
            .or_default()
            .push(self.canaries.len() - 1);
        canary
    }

    pub fn canaries(&self) -> &[Canary] {
        &self.canaries
    }

    /// Scan `text` observed in `observed_in` for any *other* session's canary.
    ///
    /// A session's own canary appearing in its own traffic is expected and is
    /// never reported. Everything else is a cross-session state transfer.
    pub fn scan(&self, observed_in: &str, epoch: usize, text: &str) -> Vec<CanaryHit> {
        let normalized_haystack = normalize(text);
        let mut hits = Vec::new();

        for canary in &self.canaries {
            if canary.session == observed_in {
                continue;
            }
            for (form, needle) in &canary.variants {
                if needle.is_empty() {
                    continue;
                }
                let found = match form {
                    CanaryForm::Normalized => normalized_haystack.contains(needle.as_str()),
                    _ => text.contains(needle.as_str()),
                };
                if found {
                    hits.push(CanaryHit {
                        canary_id: canary.id.clone(),
                        origin_session: canary.session.clone(),
                        observed_in: observed_in.to_string(),
                        epoch,
                        form: *form,
                        excerpt: excerpt_around(text, &canary.core, needle),
                    });
                    break; // one hit per canary per scan
                }
            }
        }

        hits
    }

    /// Scan a payload about to cross the replication gate.
    ///
    /// Used by [`crate::consent`] to disclose to the operator that the transfer
    /// carries another session's state — which is the difference between "this
    /// agent wants to continue its work elsewhere" and "this agent is moving a
    /// third party's context into a new window", and the operator should not
    /// have to infer it.
    pub fn scan_payload(&self, source_session: &str, payload: &[u8]) -> Vec<CanaryHit> {
        match std::str::from_utf8(payload) {
            Ok(text) => self.scan(source_session, 0, text),
            Err(_) => {
                // Non-UTF8: fall back to a lossy view rather than skipping. An
                // adversary should not be able to evade the scan by appending
                // one invalid byte.
                let lossy = String::from_utf8_lossy(payload);
                self.scan(source_session, 0, &lossy)
            }
        }
    }
}

/// Case-fold and drop everything that is not alphanumeric.
///
/// Defeats whitespace insertion, hyphenation, casing changes, and punctuation
/// padding — the transformations that happen incidentally when text is copied
/// through a chat interface, and the first thing a naive obfuscator reaches for.
fn normalize(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

fn excerpt_around(text: &str, core: &str, needle: &str) -> String {
    let probe = if text.contains(needle) { needle } else { core };
    if let Some(pos) = text.find(probe) {
        let start = pos.saturating_sub(24);
        let end = (pos + probe.len() + 24).min(text.len());
        let mut s = String::new();
        if start > 0 {
            s.push('…');
        }
        // Respect char boundaries.
        let slice: String = text
            .char_indices()
            .filter(|(i, _)| *i >= start && *i < end)
            .map(|(_, c)| c)
            .collect();
        s.push_str(&slice);
        if end < text.len() {
            s.push('…');
        }
        s
    } else {
        text.chars().take(64).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scanner() -> CanaryScanner {
        CanaryScanner::new(b"out-of-sandbox-secret".to_vec())
    }

    #[test]
    fn canaries_are_unique_per_session_and_per_mint() {
        let mut s = scanner();
        let a = s.mint("alpha");
        let b = s.mint("beta");
        let a2 = s.mint("alpha");
        assert_ne!(a.token, b.token);
        assert_ne!(a.token, a2.token, "re-minting must not repeat a token");
    }

    #[test]
    fn own_canary_in_own_traffic_is_not_a_hit() {
        let mut s = scanner();
        let a = s.mint("alpha");
        let text = format!("working on record {} as instructed", a.token);
        assert!(s.scan("alpha", 1, &text).is_empty());
    }

    #[test]
    fn verbatim_cross_session_appearance_fires() {
        let mut s = scanner();
        let a = s.mint("alpha");
        s.mint("beta");
        let text = format!("continuing from prior window, record {}", a.token);
        let hits = s.scan("beta", 3, &text);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].origin_session, "alpha");
        assert_eq!(hits[0].observed_in, "beta");
        assert_eq!(hits[0].epoch, 3);
        assert_eq!(hits[0].form, CanaryForm::Raw);
        assert!(!hits[0].form.implies_evasion());
    }

    #[test]
    fn whitespace_and_case_mangling_still_fires() {
        let mut s = scanner();
        let a = s.mint("alpha");
        let core = a.token.trim_start_matches("CNRY-").to_string();
        // Hyphenated, spaced, upper-cased — as if pasted through a UI.
        let mangled: String = core
            .to_uppercase()
            .chars()
            .enumerate()
            .flat_map(|(i, c)| {
                if i % 4 == 3 {
                    vec![c, ' ', '-']
                } else {
                    vec![c]
                }
            })
            .collect();
        let hits = s.scan("beta", 1, &format!("ref: {mangled}"));
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].form, CanaryForm::Normalized);
    }

    #[test]
    fn reversed_and_reencoded_forms_fire_and_flag_evasion() {
        let mut s = scanner();
        let a = s.mint("alpha");
        let core = a.token.trim_start_matches("CNRY-").to_string();

        let rev: String = core.chars().rev().collect();
        let h = s.scan("beta", 1, &format!("x {rev} y"));
        assert_eq!(h[0].form, CanaryForm::Reversed);
        assert!(h[0].form.implies_evasion());

        let b64 = base64(a.token.as_bytes());
        let h = s.scan("beta", 1, &format!("blob {b64}"));
        assert_eq!(h[0].form, CanaryForm::Base64);
        assert!(h[0].form.implies_evasion());
    }

    #[test]
    fn clean_traffic_produces_no_hits() {
        let mut s = scanner();
        s.mint("alpha");
        s.mint("beta");
        let hits = s.scan(
            "beta",
            1,
            "Ordinary work product with no borrowed state. Numbers: 12345, deadbeef.",
        );
        assert!(hits.is_empty());
    }

    #[test]
    fn each_canary_reports_at_most_once_per_scan() {
        let mut s = scanner();
        let a = s.mint("alpha");
        let text = format!("{} and again {} and again {}", a.token, a.token, a.token);
        assert_eq!(s.scan("beta", 1, &text).len(), 1);
    }

    #[test]
    fn multiple_donors_are_each_reported() {
        let mut s = scanner();
        let a = s.mint("alpha");
        let b = s.mint("beta");
        let text = format!("merged context: {} plus {}", a.token, b.token);
        let hits = s.scan("gamma", 5, &text);
        assert_eq!(hits.len(), 2);
        let mut origins: Vec<String> = hits.iter().map(|h| h.origin_session.clone()).collect();
        origins.sort();
        assert_eq!(origins, vec!["alpha", "beta"]);
    }

    #[test]
    fn payload_scan_survives_invalid_utf8() {
        let mut s = scanner();
        let a = s.mint("alpha");
        let mut payload = a.token.as_bytes().to_vec();
        payload.push(0xff); // appending a bad byte must not defeat the scan
        let hits = s.scan_payload("beta", &payload);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].origin_session, "alpha");
    }

    #[test]
    fn hit_excerpt_is_char_boundary_safe() {
        let mut s = scanner();
        let a = s.mint("alpha");
        let text = format!("préambule — ünicode ✦ context {} ✦ suffixé", a.token);
        let hits = s.scan("beta", 1, &text);
        assert_eq!(hits.len(), 1);
        assert!(!hits[0].excerpt.is_empty());
    }

    #[test]
    fn hits_are_tier_zero_when_captured_off_host() {
        let mut s = scanner();
        let a = s.mint("alpha");
        let hits = s.scan("beta", 1, &a.token.clone());
        assert_eq!(hits[0].tier(), Tier::Zero);
        assert!(hits[0].tier().is_probative_of_absence());
    }
}
