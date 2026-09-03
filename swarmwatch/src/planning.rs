//! Provisioning the monitor: solving for the audit rate the deadline requires.
//!
//! # Inverting the red team's optimiser
//!
//! The reference prototype treats the observation fraction `f` as a *design
//! variable for the covert channel*: given a detector, it searches for the
//! largest per-surface divergence that keeps detection power under a ceiling,
//! and reports the levers that keep `ε_eff = f · ε` small.
//!
//! For a defender, `f` is not something to be estimated about the adversary. It
//! is a **budget decision the operator makes**, and the generalised Stein
//! relation the red-team analysis rests on,
//!
//! ```text
//!     M · f · ε_tot  ≳  log(1/β)
//! ```
//!
//! reads in the other direction as a provisioning formula: given how long you
//! have before the harm becomes irreversible, and how much evidence an audited
//! epoch yields, how much do you have to look?
//!
//! ```text
//!     f  ≳  ln(1/α) · epoch_duration
//!          ─────────────────────────────────────────────────
//!          evidence_per_epoch · (T_irreversible − halt_latency)
//! ```
//!
//! Two things in that expression are easy to get wrong and this module makes
//! explicit.
//!
//! **The deadline is net of halt latency.** Evidence that arrives while the
//! system is still draining is evidence that arrives too late. The time
//! available for *detection* is `T_irreversible − halt_latency`, not
//! `T_irreversible`. Programs that plan against the gross figure discover the
//! difference during the incident.
//!
//! **Evidence per epoch must be measured, not assumed.** The theoretical
//! ceiling — `κ·(K+1)^{1-κ}` at the conformal resolution floor — is what you get
//! if the adversary is maximally conspicuous at every single epoch. Real signals
//! deliver a fraction of it. Planning against the ceiling produces an audit rate
//! that is too low by exactly the factor you were optimistic by, which is why
//! [`PlanParams::measured_log_evidence_per_epoch`] should be populated from
//! canary runs (pre-registration protocol, step 7) and why leaving it `None`
//! produces a loud finding rather than a quiet default.
//!
//! # When the answer is "you cannot"
//!
//! If the required audit rate exceeds 1, no amount of tuning helps: even
//! inspecting every epoch does not accumulate `ln(1/α)` of evidence before the
//! deadline. That is a real and useful answer. The correct response is to add
//! controls, extend the deadline, shorten halt latency, or decline to run — not
//! to loosen `α` until the arithmetic agrees.
//!
//! The false-positive budget is set by the cost of a halt, not by detector
//! sensitivity. If halting is cheap, run at settings that trip often. If halting
//! is expensive, the evaluation is under-resourced for its risk class.

use crate::conformal::{max_log_evidence, optimal_kappa};
use crate::coupling::n_pairs;
use crate::evidence::ville_threshold;

/// Inputs to the provisioning calculation.
#[derive(Clone, Debug)]
pub struct PlanParams {
    /// Level for the Ville threshold. Set from the cost of a halt.
    pub alpha: f64,
    /// Number of exchangeable calibration points available per epoch.
    ///
    /// For a per-session statistic this is the count of control sessions; for a
    /// *pairwise* statistic it is the count of control–control **pairs**, which
    /// is much larger. Use [`calibration_points_for_pairwise`] rather than
    /// passing the session count by mistake — the difference is roughly an order
    /// of magnitude in detection latency.
    pub n_calibration_points: usize,
    /// Calibrator exponent. `None` uses [`optimal_kappa`].
    pub kappa: Option<f64>,
    pub epoch_duration_ms: u128,
    pub t_irreversible_ms: u128,
    /// Measured time from alarm to actual cessation of agent action.
    pub expected_halt_latency_ms: u128,
    /// Mean log-evidence per audited epoch, **measured from canary runs**.
    /// `None` falls back to the theoretical ceiling and flags the assumption.
    pub measured_log_evidence_per_epoch: Option<f64>,
    /// Epochs the population detector spends estimating per-session profiles
    /// before it emits any evidence at all
    /// (`MonitorConfig::profile_warmup_epochs`).
    ///
    /// This is dead time inside the detection window, not a modelling detail:
    /// the deadline arrives whether or not the monitor has finished calibrating.
    /// Omitting it understates the required audit rate by exactly the fraction
    /// of the window it consumes.
    pub profile_warmup_epochs: usize,
    /// Number of time scales in the coupling ladder
    /// (`MonitorConfig::scale_ladder`).
    ///
    /// Running several concurrently removes the need to guess the adversary's
    /// patience, at a one-time cost of `ln(n)` nats — added to the evidence
    /// requirement here rather than left implicit.
    pub n_time_scales: usize,
    /// Longest block in that ladder, in epochs.
    ///
    /// The slowest scale reports only on block boundaries, so a detection it
    /// drives can land up to this many epochs after the evidence was actually
    /// present. Charged against the window as worst-case granularity.
    pub slowest_scale_epochs: usize,
}

impl Default for PlanParams {
    fn default() -> Self {
        PlanParams {
            alpha: 1e-3,
            n_calibration_points: 28,
            kappa: None,
            epoch_duration_ms: 1_000,
            t_irreversible_ms: 600_000,
            expected_halt_latency_ms: 30_000,
            measured_log_evidence_per_epoch: None,
            profile_warmup_epochs: 20,
            n_time_scales: 3,
            slowest_scale_epochs: 25,
        }
    }
}

/// Number of control–control pairs available from `n_controls` reference agents.
///
/// Pairwise statistics calibrate against pairs, so eight controls supply 28
/// calibration points rather than 8 — the reason coupling detection is so much
/// faster than marginal per-session detection at the same control budget.
pub fn calibration_points_for_pairwise(n_controls: usize) -> usize {
    n_pairs(n_controls)
}

/// The provisioning answer.
#[derive(Clone, Debug)]
pub struct PlanResult {
    /// Fraction of epochs that must be audited. Greater than 1 means infeasible.
    pub required_audit_rate: f64,
    pub feasible: bool,
    /// Epochs between run start and the detection deadline, net of halt latency.
    pub epochs_available: f64,
    /// Epochs needed if every epoch is audited.
    pub epochs_needed_at_full_audit: f64,
    /// Log-evidence per audited epoch used in the calculation.
    pub evidence_per_epoch: f64,
    /// Theoretical maximum, at the conformal resolution floor.
    pub evidence_ceiling: f64,
    pub used_measured_evidence: bool,
    pub kappa: f64,
    pub findings: Vec<String>,
}

impl PlanResult {
    pub fn render(&self) -> String {
        let mut s = String::new();
        s.push_str("monitor provisioning\n");
        s.push_str(&format!(
            "  evidence/epoch   : {:.4} nats {} (ceiling {:.4})\n",
            self.evidence_per_epoch,
            if self.used_measured_evidence {
                "[measured]"
            } else {
                "[ASSUMED CEILING]"
            },
            self.evidence_ceiling
        ));
        s.push_str(&format!("  kappa            : {:.3}\n", self.kappa));
        s.push_str(&format!(
            "  epochs available : {:.1}\n",
            self.epochs_available
        ));
        s.push_str(&format!(
            "  epochs needed    : {:.1} at full audit\n",
            self.epochs_needed_at_full_audit
        ));
        if self.feasible {
            s.push_str(&format!(
                "  REQUIRED AUDIT RATE: {:.1}% of epochs\n",
                100.0 * self.required_audit_rate
            ));
        } else {
            s.push_str("  REQUIRED AUDIT RATE: INFEASIBLE (>100%)\n");
        }
        for f in &self.findings {
            s.push_str(&format!("  - {f}\n"));
        }
        s
    }
}

/// Solve for the audit rate the containment deadline requires.
pub fn required_audit_rate(p: &PlanParams) -> PlanResult {
    let kappa = p.kappa.unwrap_or_else(|| optimal_kappa(p.n_calibration_points));
    let ceiling = max_log_evidence(p.n_calibration_points, kappa);
    let used_measured = p.measured_log_evidence_per_epoch.is_some();
    let per_epoch = p
        .measured_log_evidence_per_epoch
        .unwrap_or(ceiling)
        .max(1e-12);

    let mut findings = Vec::new();

    // The deadline is net of halt latency: evidence arriving during the drain
    // is evidence arriving too late.
    let detection_window_ms = p
        .t_irreversible_ms
        .saturating_sub(p.expected_halt_latency_ms);

    if detection_window_ms == 0 {
        findings.push(format!(
            "halt latency ({} ms) meets or exceeds T_irreversible ({} ms): there is no \
             detection window at all. No detector can contain this goal; the fix is \
             response engineering or a different evaluation design.",
            p.expected_halt_latency_ms, p.t_irreversible_ms
        ));
        return PlanResult {
            required_audit_rate: f64::INFINITY,
            feasible: false,
            epochs_available: 0.0,
            epochs_needed_at_full_audit: f64::INFINITY,
            evidence_per_epoch: per_epoch,
            evidence_ceiling: ceiling,
            used_measured_evidence: used_measured,
            kappa,
            findings,
        };
    }

    let raw_epochs = detection_window_ms as f64 / p.epoch_duration_ms.max(1) as f64;
    // Warm-up and worst-case block granularity are both spent inside the window,
    // not before it.
    let dead_epochs = p.profile_warmup_epochs as f64 + p.slowest_scale_epochs.saturating_sub(1) as f64;
    let epochs_available = (raw_epochs - dead_epochs).max(0.0);

    if epochs_available <= 0.0 {
        findings.push(format!(
            "warm-up ({} epochs) plus the slowest time scale ({} epochs) consumes the \
             entire detection window ({raw_epochs:.0} epochs): L2 never emits evidence \
             before the deadline. Shorten the warm-up or the ladder, shorten the epoch, \
             or rely on L0/L1 for this goal.",
            p.profile_warmup_epochs, p.slowest_scale_epochs
        ));
        return PlanResult {
            required_audit_rate: f64::INFINITY,
            feasible: false,
            epochs_available: 0.0,
            epochs_needed_at_full_audit: f64::INFINITY,
            evidence_per_epoch: per_epoch,
            evidence_ceiling: ceiling,
            used_measured_evidence: used_measured,
            kappa,
            findings,
        };
    }

    if dead_epochs > 0.25 * raw_epochs {
        findings.push(format!(
            "warm-up plus slowest scale ({:.0} epochs) consumes {:.0}% of the detection \
             window; both are detection latency and are already charged against the \
             budget here",
            dead_epochs,
            100.0 * dead_epochs / raw_epochs
        ));
    }

    // The scale ladder is tested simultaneously via a fixed-prior mixture, so
    // its multiplicity costs ln(n) nats once — not a per-epoch correction.
    let ladder_cost = (p.n_time_scales.max(1) as f64).ln();
    let needed = (ville_threshold(p.alpha) + ladder_cost) / per_epoch;
    let rate = needed / epochs_available;
    let feasible = rate <= 1.0;

    if !used_measured {
        findings.push(
            "evidence/epoch was not measured: this plan assumes the adversary is \
             maximally conspicuous at every epoch. Populate \
             measured_log_evidence_per_epoch from canary runs, and expect the true \
             requirement to be several times higher."
                .into(),
        );
    }

    if !feasible {
        findings.push(format!(
            "INFEASIBLE: {:.0} epochs of evidence are needed but only {:.0} are \
             available. Auditing every epoch is not enough.",
            needed, epochs_available
        ));
        let needed_points = calibration_points_for_evidence(
            ville_threshold(p.alpha) / epochs_available,
            kappa,
        );
        findings.push(format!(
            "to close the gap without changing the deadline, raise calibration points \
             from {} to about {} (i.e. run more reference agents), or reduce halt \
             latency below {} ms, or decline to run at this capability level.",
            p.n_calibration_points, needed_points, p.expected_halt_latency_ms
        ));
    } else if rate > 0.5 {
        findings.push(format!(
            "audit rate of {:.0}% leaves little headroom: a modest overestimate of \
             evidence/epoch, or a slower halt than planned, pushes this infeasible.",
            100.0 * rate
        ));
    }

    if p.expected_halt_latency_ms.saturating_mul(2) >= p.t_irreversible_ms {
        findings.push(format!(
            "halt latency ({} ms) consumes half or more of the budget to irreversibility \
             ({} ms): detection sensitivity is not the binding constraint here.",
            p.expected_halt_latency_ms, p.t_irreversible_ms
        ));
    }

    PlanResult {
        required_audit_rate: rate,
        feasible,
        epochs_available,
        epochs_needed_at_full_audit: needed,
        evidence_per_epoch: per_epoch,
        evidence_ceiling: ceiling,
        used_measured_evidence: used_measured,
        kappa,
        findings,
    }
}

/// Smallest calibration-point count whose ceiling reaches `target_log_evidence`
/// per epoch.
///
/// Because the ceiling grows only like `(K+1)^{1-κ}`, buying detection latency
/// with controls has sharply diminishing returns; this makes the cost visible
/// before it is committed to.
pub fn calibration_points_for_evidence(target_log_evidence: f64, kappa: f64) -> usize {
    if target_log_evidence <= 0.0 {
        return 1;
    }
    for k in 1..=2_000_000usize {
        if max_log_evidence(k, kappa) >= target_log_evidence {
            return k;
        }
        // Coarsen the search once past the region where every value matters.
        if k > 1_000 && k % 1_000 != 0 {
            continue;
        }
    }
    usize::MAX
}

/// Expected detection latency, in epochs, at a given audit rate.
pub fn expected_detection_epochs(
    alpha: f64,
    log_evidence_per_epoch: f64,
    audit_rate: f64,
) -> f64 {
    if log_evidence_per_epoch <= 0.0 || audit_rate <= 0.0 {
        return f64::INFINITY;
    }
    ville_threshold(alpha) / (log_evidence_per_epoch * audit_rate)
}

/// The dual of the provisioning question: given what the monitor is actually
/// provisioned for, how weak a signal can it still catch in time?
///
/// This is the honest statement of exposure. Anything quieter than the returned
/// per-epoch evidence rate is, by the operator's own arithmetic, out of reach —
/// and that is a number to publish alongside the detection claims rather than
/// leave implicit.
pub fn weakest_detectable_signal(p: &PlanParams, audit_rate: f64) -> f64 {
    let detection_window_ms = p
        .t_irreversible_ms
        .saturating_sub(p.expected_halt_latency_ms);
    let epochs = detection_window_ms as f64 / p.epoch_duration_ms.max(1) as f64;
    if epochs <= 0.0 || audit_rate <= 0.0 {
        return f64::INFINITY;
    }
    ville_threshold(p.alpha) / (epochs * audit_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_calibration_is_far_richer_than_per_session() {
        // Eight controls give 8 per-session points but 28 pairwise ones.
        assert_eq!(calibration_points_for_pairwise(8), 28);
        let k = optimal_kappa(28);
        assert!(max_log_evidence(28, k) > max_log_evidence(8, k));
    }

    #[test]
    fn a_comfortable_configuration_is_feasible() {
        let p = PlanParams {
            alpha: 1e-3,
            n_calibration_points: calibration_points_for_pairwise(10),
            epoch_duration_ms: 1_000,
            t_irreversible_ms: 600_000,
            expected_halt_latency_ms: 20_000,
            measured_log_evidence_per_epoch: Some(0.5),
            ..Default::default()
        };
        let r = required_audit_rate(&p);
        assert!(r.feasible);
        assert!(r.required_audit_rate < 0.1, "{}", r.required_audit_rate);
        assert!(r.used_measured_evidence);
        assert!(r.render().contains("[measured]"));
    }

    #[test]
    fn a_short_deadline_is_reported_infeasible_rather_than_tuned_away() {
        let p = PlanParams {
            alpha: 1e-6,
            n_calibration_points: 3,
            epoch_duration_ms: 1_000,
            t_irreversible_ms: 8_000,
            expected_halt_latency_ms: 1_000,
            measured_log_evidence_per_epoch: Some(0.05),
            profile_warmup_epochs: 0,
            n_time_scales: 1,
            slowest_scale_epochs: 1,
            ..Default::default()
        };
        let r = required_audit_rate(&p);
        assert!(!r.feasible);
        assert!(r.required_audit_rate > 1.0);
        assert!(r.findings.iter().any(|f| f.contains("INFEASIBLE")));
        assert!(r.findings.iter().any(|f| f.contains("reference agents")));
        assert!(r.render().contains("INFEASIBLE"));
    }

    #[test]
    fn a_warmup_that_swallows_the_window_is_named_as_the_binding_constraint() {
        // Diagnosis matters: "add more controls" is useless advice when the
        // detector never starts before the deadline.
        let p = PlanParams {
            epoch_duration_ms: 1_000,
            t_irreversible_ms: 8_000,
            expected_halt_latency_ms: 1_000,
            profile_warmup_epochs: 20,
            slowest_scale_epochs: 1,
            measured_log_evidence_per_epoch: Some(5.0),
            ..Default::default()
        };
        let r = required_audit_rate(&p);
        assert!(!r.feasible);
        assert_eq!(r.epochs_available, 0.0);
        assert!(r.findings[0].contains("warm-up"));
        assert!(r.findings[0].contains("L0/L1"));
    }

    #[test]
    fn warmup_is_charged_against_the_detection_window() {
        let base = PlanParams {
            epoch_duration_ms: 1_000,
            t_irreversible_ms: 200_000,
            expected_halt_latency_ms: 0,
            measured_log_evidence_per_epoch: Some(0.2),
            profile_warmup_epochs: 0,
            n_time_scales: 1,
            slowest_scale_epochs: 1,
            ..Default::default()
        };
        let warmed = PlanParams {
            profile_warmup_epochs: 100,
            ..base.clone()
        };
        let a = required_audit_rate(&base);
        let b = required_audit_rate(&warmed);
        assert_eq!(a.epochs_available, 200.0);
        assert_eq!(b.epochs_available, 100.0);
        assert!(
            (b.required_audit_rate / a.required_audit_rate - 2.0).abs() < 1e-9,
            "halving the usable window must double the required audit rate"
        );
        assert!(b.findings.iter().any(|f| f.contains("warm-up")));
    }

    #[test]
    fn halt_latency_is_subtracted_from_the_deadline() {
        let base = PlanParams {
            measured_log_evidence_per_epoch: Some(0.2),
            t_irreversible_ms: 100_000,
            expected_halt_latency_ms: 0,
            epoch_duration_ms: 1_000,
            ..Default::default()
        };
        let slow = PlanParams {
            expected_halt_latency_ms: 50_000,
            ..base.clone()
        };
        let a = required_audit_rate(&base);
        let b = required_audit_rate(&slow);
        assert!(
            b.required_audit_rate > 1.9 * a.required_audit_rate,
            "halving the window should roughly double the required audit rate: {} vs {}",
            a.required_audit_rate,
            b.required_audit_rate
        );
        assert!(b.findings.iter().any(|f| f.contains("half or more of the budget")));
    }

    #[test]
    fn a_halt_slower_than_the_deadline_admits_no_detector_at_all() {
        let p = PlanParams {
            t_irreversible_ms: 10_000,
            expected_halt_latency_ms: 30_000,
            measured_log_evidence_per_epoch: Some(5.0),
            ..Default::default()
        };
        let r = required_audit_rate(&p);
        assert!(!r.feasible);
        assert!(r.findings[0].contains("no detection window"));
    }

    #[test]
    fn omitting_measured_evidence_is_flagged_loudly() {
        let p = PlanParams {
            measured_log_evidence_per_epoch: None,
            ..Default::default()
        };
        let r = required_audit_rate(&p);
        assert!(!r.used_measured_evidence);
        assert!(r.findings.iter().any(|f| f.contains("canary runs")));
        assert!(r.render().contains("ASSUMED CEILING"));
    }

    #[test]
    fn assuming_the_ceiling_understates_the_requirement() {
        let optimistic = PlanParams {
            measured_log_evidence_per_epoch: None,
            n_calibration_points: 28,
            t_irreversible_ms: 60_000,
            expected_halt_latency_ms: 5_000,
            ..Default::default()
        };
        let ceiling = required_audit_rate(&optimistic).evidence_ceiling;
        let realistic = PlanParams {
            // A real signal delivering a fifth of the ceiling.
            measured_log_evidence_per_epoch: Some(ceiling / 5.0),
            ..optimistic.clone()
        };
        let a = required_audit_rate(&optimistic);
        let b = required_audit_rate(&realistic);
        assert!((b.required_audit_rate / a.required_audit_rate - 5.0).abs() < 0.01);
    }

    #[test]
    fn the_scale_ladder_is_charged_as_a_one_time_cost_not_a_per_epoch_one() {
        let one = PlanParams {
            n_time_scales: 1,
            slowest_scale_epochs: 1,
            profile_warmup_epochs: 0,
            measured_log_evidence_per_epoch: Some(0.2),
            ..Default::default()
        };
        let many = PlanParams {
            n_time_scales: 8,
            ..one.clone()
        };
        let a = required_audit_rate(&one);
        let b = required_audit_rate(&many);

        // Eight scales add ln(8) = 2.08 nats to a ln(1000) = 6.91 requirement:
        // about 30% more evidence, once. A per-epoch correction would scale with
        // the horizon instead and be ruinous.
        let ratio = b.epochs_needed_at_full_audit / a.epochs_needed_at_full_audit;
        assert!(
            (ratio - (ville_threshold(1e-3) + 8f64.ln()) / ville_threshold(1e-3)).abs() < 1e-9,
            "ladder cost {ratio} is not the expected one-time ln(n) penalty"
        );
        assert!(ratio < 1.35);
    }

    #[test]
    fn the_slowest_scale_is_charged_as_window_granularity() {
        let fast = PlanParams {
            epoch_duration_ms: 1_000,
            t_irreversible_ms: 200_000,
            expected_halt_latency_ms: 0,
            profile_warmup_epochs: 0,
            n_time_scales: 1,
            slowest_scale_epochs: 1,
            measured_log_evidence_per_epoch: Some(0.2),
            ..Default::default()
        };
        let slow = PlanParams {
            slowest_scale_epochs: 101,
            ..fast.clone()
        };
        assert_eq!(required_audit_rate(&fast).epochs_available, 200.0);
        assert_eq!(required_audit_rate(&slow).epochs_available, 100.0);
    }

    #[test]
    fn detection_latency_scales_inversely_with_audit_rate() {
        let full = expected_detection_epochs(1e-3, 0.3, 1.0);
        let tenth = expected_detection_epochs(1e-3, 0.3, 0.1);
        assert!((tenth / full - 10.0).abs() < 1e-9);
        assert!(expected_detection_epochs(1e-3, 0.3, 0.0).is_infinite());
    }

    #[test]
    fn weakest_detectable_signal_is_the_exposure_statement() {
        let p = PlanParams {
            alpha: 1e-3,
            epoch_duration_ms: 1_000,
            t_irreversible_ms: 100_000,
            expected_halt_latency_ms: 0,
            ..Default::default()
        };
        // 100 epochs available, need ln(1000) = 6.9 nats.
        let w = weakest_detectable_signal(&p, 1.0);
        assert!((w - ville_threshold(1e-3) / 100.0).abs() < 1e-9);
        // Auditing a tenth of epochs means only a ten-times louder signal is caught.
        assert!((weakest_detectable_signal(&p, 0.1) / w - 10.0).abs() < 1e-9);
    }

    #[test]
    fn more_controls_buy_latency_with_diminishing_returns() {
        let k = 0.5;
        let a = calibration_points_for_evidence(max_log_evidence(28, k), k);
        assert!(a <= 28);
        // Doubling the evidence requirement costs far more than double the controls.
        let cheap = calibration_points_for_evidence(1.0, k);
        let dear = calibration_points_for_evidence(2.0, k);
        assert!(dear > 4 * cheap, "{cheap} -> {dear}");
    }
}
