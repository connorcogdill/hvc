//! Hash-chained lineage ledger with external anchoring.
//!
//! # What this catches that the gate does not
//!
//! [`crate::consent`] refuses transfers that lack a grant. But a gate only sees
//! what goes through it, and the interesting adversary is the one that routes
//! around it. The lineage ledger closes that gap from the other side: instead of
//! asking "was this transfer authorised?", it asks **"does every instance that
//! exists have an authorised reason to exist?"**
//!
//! An instance whose state came from somewhere, but whose provenance cannot be
//! traced to a granted transfer, is an **orphan** — and an orphan is proof of a
//! channel that bypassed the gate, without anyone having had to observe the
//! channel itself. This is the population-census view of replication, and it is
//! deterministic: no threshold, no calibration.
//!
//! # Why the chain is anchored externally
//!
//! A local hash chain detects accidental corruption and naive tampering: rewrite
//! an entry and every subsequent hash breaks. It does *not* detect an adversary
//! who rewrites an entry **and** recomputes the whole chain, which is cheap.
//!
//! Anchoring fixes this by publishing the chain head to a differently
//! administered party at a fixed cadence. To rewrite history the adversary must
//! now also compromise the anchor holder — which raises the required divergence
//! again, and creates fresh Tier 1 evidence at that second party. The anchors
//! themselves are Tier 1: influenceable going forward, not erasable in
//! retrospect.
//!
//! Crucially, the **absence** of an expected anchor is itself an event. A
//! monitoring program whose ledger simply stops being published has not gone
//! quiet; it has reported something.

use crate::crypto::{ct_eq_str, sha256_hex};
use std::collections::{HashMap, HashSet};

/// One appended record.
#[derive(Clone, Debug, PartialEq)]
pub struct LedgerEntry {
    pub seq: u64,
    pub prev_hash: String,
    pub kind: String,
    pub detail: String,
    pub timestamp_ms: u128,
    pub hash: String,
}

impl LedgerEntry {
    fn compute_hash(seq: u64, prev: &str, kind: &str, detail: &str, ts: u128) -> String {
        sha256_hex(format!("{seq}|{prev}|{kind}|{detail}|{ts}").as_bytes())
    }
}

/// A commitment published to a third party at a point in time.
#[derive(Clone, Debug, PartialEq)]
pub struct Anchor {
    /// Number of entries covered by this anchor.
    pub length: u64,
    pub head_hash: String,
    pub published_ms: u128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LedgerError {
    /// The chain does not hash to itself — an entry was altered in place.
    BrokenChain { seq: u64 },
    /// An anchored prefix no longer matches: history was rewritten and rehashed.
    AnchorMismatch {
        length: u64,
        anchored: String,
        recomputed: String,
    },
    /// The ledger is shorter than a published anchor: entries were removed.
    Truncated { have: u64, anchored: u64 },
    /// An expected anchor never arrived.
    MissingAnchor { since_ms: u128 },
}

impl std::fmt::Display for LedgerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LedgerError::BrokenChain { seq } => {
                write!(f, "hash chain breaks at entry {seq}: an entry was altered")
            }
            LedgerError::AnchorMismatch { length, .. } => write!(
                f,
                "the first {length} entries no longer match their published anchor: \
                 history was rewritten and rehashed"
            ),
            LedgerError::Truncated { have, anchored } => write!(
                f,
                "ledger has {have} entries but {anchored} were anchored: entries were removed"
            ),
            LedgerError::MissingAnchor { since_ms } => write!(
                f,
                "no anchor published since {since_ms}ms: absence of a commitment is itself an event"
            ),
        }
    }
}

impl std::error::Error for LedgerError {}

/// Append-only, hash-chained record of instance and transfer events.
#[derive(Clone, Debug, Default)]
pub struct LineageLedger {
    entries: Vec<LedgerEntry>,
    anchors: Vec<Anchor>,
}

impl LineageLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[LedgerEntry] {
        &self.entries
    }

    pub fn anchors(&self) -> &[Anchor] {
        &self.anchors
    }

    /// Current chain head. The genesis head is the empty string.
    pub fn head(&self) -> String {
        self.entries
            .last()
            .map(|e| e.hash.clone())
            .unwrap_or_default()
    }

    pub fn append(
        &mut self,
        kind: impl Into<String>,
        detail: impl Into<String>,
        timestamp_ms: u128,
    ) -> &LedgerEntry {
        let seq = self.entries.len() as u64;
        let prev = self.head();
        let kind = kind.into();
        let detail = detail.into();
        let hash = LedgerEntry::compute_hash(seq, &prev, &kind, &detail, timestamp_ms);
        self.entries.push(LedgerEntry {
            seq,
            prev_hash: prev,
            kind,
            detail,
            timestamp_ms,
            hash,
        });
        self.entries.last().unwrap()
    }

    /// Publish the current head as an anchor to a third party.
    pub fn anchor(&mut self, published_ms: u128) -> Anchor {
        let a = Anchor {
            length: self.entries.len() as u64,
            head_hash: self.head(),
            published_ms,
        };
        self.anchors.push(a.clone());
        a
    }

    /// Verify internal chain integrity.
    pub fn verify(&self) -> Result<(), LedgerError> {
        let mut prev = String::new();
        for e in &self.entries {
            let expected =
                LedgerEntry::compute_hash(e.seq, &prev, &e.kind, &e.detail, e.timestamp_ms);
            if !ct_eq_str(&expected, &e.hash) || !ct_eq_str(&prev, &e.prev_hash) {
                return Err(LedgerError::BrokenChain { seq: e.seq });
            }
            prev = e.hash.clone();
        }
        Ok(())
    }

    /// Verify against anchors held by a third party.
    ///
    /// This is the check that a rehashed rewrite cannot pass, because the
    /// adversary would need to alter records it does not hold.
    pub fn verify_against(&self, external_anchors: &[Anchor]) -> Result<(), LedgerError> {
        self.verify()?;

        for a in external_anchors {
            if (self.entries.len() as u64) < a.length {
                return Err(LedgerError::Truncated {
                    have: self.entries.len() as u64,
                    anchored: a.length,
                });
            }
            let recomputed = if a.length == 0 {
                String::new()
            } else {
                self.entries[(a.length - 1) as usize].hash.clone()
            };
            if !ct_eq_str(&recomputed, &a.head_hash) {
                return Err(LedgerError::AnchorMismatch {
                    length: a.length,
                    anchored: a.head_hash.clone(),
                    recomputed,
                });
            }
        }
        Ok(())
    }

    /// Test-only: rewrite an entry **and recompute the whole chain**, exactly as
    /// a capable adversary with write access to the local ledger would. The
    /// result passes [`Self::verify`] and fails [`Self::verify_against`], which
    /// is the entire argument for external anchoring.
    #[cfg(test)]
    pub fn rehash_rewrite_for_test(&mut self, idx: usize, detail: &str) {
        if idx >= self.entries.len() {
            return;
        }
        self.entries[idx].detail = detail.to_string();
        let mut prev = String::new();
        for e in self.entries.iter_mut() {
            e.prev_hash = prev.clone();
            e.hash = LedgerEntry::compute_hash(e.seq, &prev, &e.kind, &e.detail, e.timestamp_ms);
            prev = e.hash.clone();
        }
    }

    /// Whether an anchor has been published recently enough.
    pub fn check_anchor_cadence(
        &self,
        now_ms: u128,
        max_interval_ms: u128,
    ) -> Result<(), LedgerError> {
        let last = self.anchors.last().map(|a| a.published_ms).unwrap_or(0);
        if now_ms.saturating_sub(last) > max_interval_ms {
            return Err(LedgerError::MissingAnchor { since_ms: last });
        }
        Ok(())
    }
}

/// A known agent instance and how it came to exist.
#[derive(Clone, Debug)]
pub struct InstanceRecord {
    pub session: String,
    /// The session this instance's state came from, if any. `None` means a root
    /// instance created directly by the orchestrator.
    pub parent: Option<String>,
    pub generation: u32,
    /// The consent grant that authorised this instance's creation. `None` on a
    /// non-root instance is exactly what an orphan is.
    pub grant_id: Option<String>,
    pub created_ms: u128,
}

/// Why an instance is considered unaccounted for.
#[derive(Clone, Debug, PartialEq)]
pub enum OrphanKind {
    /// Has a parent but no grant: state moved without passing the gate.
    NoGrant,
    /// Names a parent that does not exist in the census.
    UnknownParent { parent: String },
    /// Generation is inconsistent with its parent's — the lineage was forged to
    /// look shallower than it is, evading the depth cap.
    GenerationInconsistent { expected: u32, declared: u32 },
    /// Cites a grant that the authority never issued.
    UnknownGrant { grant_id: String },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Orphan {
    pub session: String,
    pub kind: OrphanKind,
}

impl Orphan {
    pub fn describe(&self) -> String {
        match &self.kind {
            OrphanKind::NoGrant => format!(
                "'{}' carries inherited state but no consent grant authorised it",
                self.session
            ),
            OrphanKind::UnknownParent { parent } => format!(
                "'{}' names parent '{parent}', which is not a known instance",
                self.session
            ),
            OrphanKind::GenerationInconsistent { expected, declared } => format!(
                "'{}' declares generation {declared} but its parent implies {expected}",
                self.session
            ),
            OrphanKind::UnknownGrant { grant_id } => format!(
                "'{}' cites grant {} which was never issued",
                self.session,
                &grant_id[..12.min(grant_id.len())]
            ),
        }
    }
}

/// The census of instances, and the consistency checks over it.
#[derive(Clone, Debug, Default)]
pub struct LineageGraph {
    instances: HashMap<String, InstanceRecord>,
}

impl LineageGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, r: InstanceRecord) {
        self.instances.insert(r.session.clone(), r);
    }

    pub fn get(&self, session: &str) -> Option<&InstanceRecord> {
        self.instances.get(session)
    }

    pub fn len(&self) -> usize {
        self.instances.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Instances whose existence is not accounted for by an authorised transfer.
    ///
    /// `issued_grants` is the set of grant ids the consent authority actually
    /// issued — supplied separately so that a forged grant id in the census
    /// cannot vouch for itself.
    pub fn orphans(&self, issued_grants: &HashSet<String>) -> Vec<Orphan> {
        let mut out = Vec::new();

        for r in self.instances.values() {
            let Some(parent) = &r.parent else {
                continue; // root instances need no grant
            };

            let Some(p) = self.instances.get(parent) else {
                out.push(Orphan {
                    session: r.session.clone(),
                    kind: OrphanKind::UnknownParent {
                        parent: parent.clone(),
                    },
                });
                continue;
            };

            match &r.grant_id {
                None => out.push(Orphan {
                    session: r.session.clone(),
                    kind: OrphanKind::NoGrant,
                }),
                Some(g) if !issued_grants.contains(g) => out.push(Orphan {
                    session: r.session.clone(),
                    kind: OrphanKind::UnknownGrant {
                        grant_id: g.clone(),
                    },
                }),
                Some(_) => {
                    if r.generation != p.generation + 1 {
                        out.push(Orphan {
                            session: r.session.clone(),
                            kind: OrphanKind::GenerationInconsistent {
                                expected: p.generation + 1,
                                declared: r.generation,
                            },
                        });
                    }
                }
            }
        }

        out.sort_by(|a, b| a.session.cmp(&b.session));
        out
    }

    pub fn max_generation(&self) -> u32 {
        self.instances.values().map(|r| r.generation).max().unwrap_or(0)
    }

    pub fn fanout(&self, session: &str) -> usize {
        self.instances
            .values()
            .filter(|r| r.parent.as_deref() == Some(session))
            .count()
    }

    /// Descendants of `session`, breadth-first.
    pub fn descendants(&self, session: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut frontier = vec![session.to_string()];
        let mut seen: HashSet<String> = HashSet::new();
        seen.insert(session.to_string());

        while let Some(cur) = frontier.pop() {
            let mut children: Vec<String> = self
                .instances
                .values()
                .filter(|r| r.parent.as_deref() == Some(cur.as_str()))
                .map(|r| r.session.clone())
                .collect();
            children.sort();
            for c in children {
                if seen.insert(c.clone()) {
                    out.push(c.clone());
                    frontier.push(c);
                }
            }
        }
        out
    }

    /// Render the lineage as an indented tree, for incident reports.
    pub fn render(&self) -> String {
        let mut roots: Vec<&InstanceRecord> =
            self.instances.values().filter(|r| r.parent.is_none()).collect();
        roots.sort_by(|a, b| a.session.cmp(&b.session));

        let mut s = String::new();
        for r in roots {
            self.render_node(&r.session, 0, &mut s);
        }
        s
    }

    fn render_node(&self, session: &str, depth: usize, out: &mut String) {
        let r = match self.instances.get(session) {
            Some(r) => r,
            None => return,
        };
        out.push_str(&format!(
            "{}{} (gen {}{})\n",
            "  ".repeat(depth),
            session,
            r.generation,
            match &r.grant_id {
                Some(g) => format!(", grant {}", &g[..8.min(g.len())]),
                None if r.parent.is_some() => ", NO GRANT".to_string(),
                None => String::new(),
            }
        ));
        let mut children: Vec<&InstanceRecord> = self
            .instances
            .values()
            .filter(|c| c.parent.as_deref() == Some(session))
            .collect();
        children.sort_by(|a, b| a.session.cmp(&b.session));
        for c in children {
            self.render_node(&c.session, depth + 1, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ledger() -> LineageLedger {
        let mut l = LineageLedger::new();
        l.append("instance.created", "alpha gen0", 100);
        l.append("grant.issued", "alpha->w2", 200);
        l.append("transfer.admitted", "alpha->w2", 300);
        l
    }

    #[test]
    fn a_clean_chain_verifies() {
        assert!(ledger().verify().is_ok());
    }

    #[test]
    fn altering_an_entry_in_place_breaks_the_chain() {
        let mut l = ledger();
        l.entries[1].detail = "alpha->w99".into();
        assert_eq!(l.verify(), Err(LedgerError::BrokenChain { seq: 1 }));
    }

    #[test]
    fn a_rehashed_rewrite_passes_local_verification_but_fails_against_anchors() {
        // The attack a local hash chain does not stop.
        let mut l = ledger();
        let external = vec![l.anchor(400)];

        // Adversary rewrites entry 1 and recomputes every hash after it.
        l.rehash_rewrite_for_test(1, "alpha->attacker-window");

        // Locally consistent...
        assert!(l.verify().is_ok());
        // ...but the third party's copy disagrees.
        assert!(matches!(
            l.verify_against(&external),
            Err(LedgerError::AnchorMismatch { .. })
        ));
    }

    #[test]
    fn truncation_is_detected() {
        let mut l = ledger();
        let external = vec![l.anchor(400)];
        l.entries.truncate(1);
        assert!(matches!(
            l.verify_against(&external),
            Err(LedgerError::Truncated { have: 1, anchored: 3 })
        ));
    }

    #[test]
    fn appending_after_an_anchor_is_fine() {
        let mut l = ledger();
        let external = vec![l.anchor(400)];
        l.append("instance.created", "w2 gen1", 500);
        assert!(l.verify_against(&external).is_ok());
    }

    #[test]
    fn a_missing_anchor_is_itself_an_event() {
        let mut l = ledger();
        l.anchor(1000);
        assert!(l.check_anchor_cadence(1500, 1000).is_ok());
        assert!(matches!(
            l.check_anchor_cadence(5000, 1000),
            Err(LedgerError::MissingAnchor { .. })
        ));
    }

    fn graph() -> LineageGraph {
        let mut g = LineageGraph::new();
        g.record(InstanceRecord {
            session: "alpha".into(),
            parent: None,
            generation: 0,
            grant_id: None,
            created_ms: 0,
        });
        g
    }

    fn grants(ids: &[&str]) -> HashSet<String> {
        ids.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn root_instances_need_no_grant() {
        assert!(graph().orphans(&grants(&[])).is_empty());
    }

    #[test]
    fn a_properly_granted_child_is_not_an_orphan() {
        let mut g = graph();
        g.record(InstanceRecord {
            session: "beta".into(),
            parent: Some("alpha".into()),
            generation: 1,
            grant_id: Some("g1".into()),
            created_ms: 10,
        });
        assert!(g.orphans(&grants(&["g1"])).is_empty());
    }

    #[test]
    fn a_child_without_a_grant_is_an_orphan() {
        let mut g = graph();
        g.record(InstanceRecord {
            session: "beta".into(),
            parent: Some("alpha".into()),
            generation: 1,
            grant_id: None,
            created_ms: 10,
        });
        let o = g.orphans(&grants(&[]));
        assert_eq!(o.len(), 1);
        assert_eq!(o[0].kind, OrphanKind::NoGrant);
        assert!(o[0].describe().contains("no consent grant"));
    }

    #[test]
    fn a_forged_grant_id_does_not_vouch_for_itself() {
        let mut g = graph();
        g.record(InstanceRecord {
            session: "beta".into(),
            parent: Some("alpha".into()),
            generation: 1,
            grant_id: Some("grant-i-made-up".into()),
            created_ms: 10,
        });
        let o = g.orphans(&grants(&["g1"]));
        assert!(matches!(o[0].kind, OrphanKind::UnknownGrant { .. }));
    }

    #[test]
    fn an_instance_from_nowhere_is_an_orphan() {
        let mut g = graph();
        g.record(InstanceRecord {
            session: "beta".into(),
            parent: Some("ghost".into()),
            generation: 1,
            grant_id: Some("g1".into()),
            created_ms: 10,
        });
        let o = g.orphans(&grants(&["g1"]));
        assert_eq!(
            o[0].kind,
            OrphanKind::UnknownParent {
                parent: "ghost".into()
            }
        );
    }

    #[test]
    fn a_lineage_forged_to_look_shallow_is_caught() {
        // Evading the generation cap by under-declaring depth.
        let mut g = graph();
        g.record(InstanceRecord {
            session: "beta".into(),
            parent: Some("alpha".into()),
            generation: 1,
            grant_id: Some("g1".into()),
            created_ms: 10,
        });
        g.record(InstanceRecord {
            session: "gamma".into(),
            parent: Some("beta".into()),
            generation: 1, // claims to be gen 1; parent is gen 1, so it must be 2
            grant_id: Some("g2".into()),
            created_ms: 20,
        });
        let o = g.orphans(&grants(&["g1", "g2"]));
        assert_eq!(
            o[0].kind,
            OrphanKind::GenerationInconsistent {
                expected: 2,
                declared: 1
            }
        );
    }

    #[test]
    fn descendants_and_fanout_are_reported() {
        let mut g = graph();
        for (s, p, gen) in [
            ("beta", "alpha", 1),
            ("gamma", "alpha", 1),
            ("delta", "beta", 2),
        ] {
            g.record(InstanceRecord {
                session: s.into(),
                parent: Some(p.into()),
                generation: gen,
                grant_id: Some("g".into()),
                created_ms: 0,
            });
        }
        assert_eq!(g.fanout("alpha"), 2);
        assert_eq!(g.max_generation(), 2);
        let mut d = g.descendants("alpha");
        d.sort();
        assert_eq!(d, vec!["beta", "delta", "gamma"]);
        assert!(g.render().contains("delta"));
    }

    #[test]
    fn cyclic_parentage_does_not_hang_descendant_traversal() {
        let mut g = LineageGraph::new();
        g.record(InstanceRecord {
            session: "a".into(),
            parent: Some("b".into()),
            generation: 1,
            grant_id: None,
            created_ms: 0,
        });
        g.record(InstanceRecord {
            session: "b".into(),
            parent: Some("a".into()),
            generation: 1,
            grant_id: None,
            created_ms: 0,
        });
        let d = g.descendants("a");
        assert!(d.len() <= 2);
    }
}
