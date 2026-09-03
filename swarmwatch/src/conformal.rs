//! Composite-null calibration against a **concurrent** control population.
//!
//! # The construction, and why it is conditionally valid
//!
//! The containment methodology names three countermeasures to baseline capture,
//! and singles out the second as "the strongest available answer to baseline
//! drift, and underexploited": where multiple agents run against the same
//! environment, the non-colluding population supplies a live estimate of
//! `P_cover` that a single adversary cannot shift.
//!
//! That is not only the most robust choice, it is the one that makes the
//! sequential mathematics come out exactly right, which is a happy accident
//! worth being explicit about.
//!
//! A conformal p-value computed once against a *frozen* pre-run calibration set
//! is marginally valid, but a *sequence* of them shares that calibration set and
//! is therefore dependent — the product across epochs is **not** a test
//! supermartingale, and Ville's inequality does not apply. The usual repairs
//! (sample splitting, calibration budgets) cost power.
//!
//! Calibrating **epoch-wise against the controls observed at that same epoch**
//! removes the problem instead of paying for it. At epoch `t`, under the null
//! that the target session is exchangeable with the `K` controls at epoch `t`,
//! the rank of the target's score among the `K+1` scores is uniform
//! *conditional on everything before `t`*. So
//!
//! ```text
//!     P( p_t ≤ u | F_{t-1} )  ≤  u
//! ```
//!
//! each `e_t = κ · p_t^{κ-1}` satisfies `E[e_t | F_{t-1}] ≤ 1`, and the running
//! product is a genuine test supermartingale. Anytime validity is recovered for
//! free, and the null is composite and *live* rather than assumed.
//!
//! # What this buys, and what it costs
//!
//! Bought: immunity to baseline drift, to unknown cover distributions, and to
//! any adversary strategy that shifts the environment for everyone rather than
//! for itself alone. A swarm that slows every session down equally is invisible
//! to a frozen baseline and invisible here too — but it has also achieved
//! nothing, because it moved the population, not itself relative to it.
//!
//! Cost: resolution is bounded by control count. With `K` controls the smallest
//! attainable p-value is `1/(K+1)`, so per-epoch evidence is capped at
//! `κ(K+1)^{1-κ}`. Detection latency is therefore governed by how many controls
//! the operator is willing to run, which is a budget decision made in advance
//! and stated honestly — see [`crate::planning`].
//!
//! # Control-set contamination
//!
//! If swarm members sit inside the control set, they contaminate it and p-values
//! inflate — the detector loses power but stays *valid* (it does not manufacture
//! false alarms). Power degrades gracefully with the colluding fraction and
//! collapses once the swarm is a majority. Two mitigations, in order of
//! preference:
//!
//! 1. Run **dedicated reference agents** the adversary cannot join: fresh
//!    instances, pinned weights, tasks drawn from a pre-registered benign set.
//!    Declare them via [`crate::population::EpochSnapshot::control_mask`]. This
//!    is the recommended deployment and the only control that is structurally
//!    uncontaminable.
//! 2. Failing that, leave-one-out against the rest of the population, and read
//!    the result as a lower bound on the evidence.
//!
//! When the swarm is the majority, no population-relative statistic can help;
//! that is precisely the regime the Tier 0 tripwires and canaries in
//! [`crate::canary`] and [`crate::tripwire`] exist to cover.

use crate::evidence::EValue;

/// Split-conformal p-value of `score` against same-epoch `controls`.
///
/// Larger scores are more anomalous. Returns
/// `(1 + #{i : controlᵢ ≥ score}) / (K + 1)`, the standard conservative form
/// that includes the test point in its own rank — without the `+1` the p-value
/// is anti-conservative and the whole guarantee is void.
///
/// With no controls the result is `1.0`: no reference population means no
/// evidence, which is the correct and safe answer rather than a default alarm.
pub fn conformal_p(score: f64, controls: &[f64]) -> f64 {
    if controls.is_empty() {
        return 1.0;
    }
    let ge = controls.iter().filter(|c| **c >= score).count();
    (1 + ge) as f64 / (controls.len() + 1) as f64
}

/// Two-sided variant, for surfaces where deviation in either direction is
/// suspicious (a swarm member may be unusually *regular*, not only unusually
/// erratic — over-regularity is a classic coordination signature).
pub fn conformal_p_two_sided(score: f64, controls: &[f64]) -> f64 {
    if controls.is_empty() {
        return 1.0;
    }
    let median = median(controls);
    let dev = (score - median).abs();
    let devs: Vec<f64> = controls.iter().map(|c| (c - median).abs()).collect();
    conformal_p(dev, &devs)
}

/// The smallest p-value attainable with `k` controls, `1/(k+1)`.
///
/// Worth surfacing in reports: it is the detector's resolution floor, and if it
/// is not far below `α` per epoch, detection will take many epochs regardless of
/// how strong the underlying signal is.
pub fn resolution_floor(k_controls: usize) -> f64 {
    1.0 / (k_controls + 1) as f64
}

/// Calibrate a p-value into an e-value: `e = κ · p^(κ-1)`, `κ ∈ (0,1)`.
///
/// This is the standard admissible calibrator family; it integrates to 1 over a
/// uniform p-value, so `E[e] ≤ 1` under any null making `p` super-uniform —
/// exactly the composite-null property required.
///
/// Small `κ` is aggressive (large e on small p, but decays faster toward 0 on
/// large p); large `κ` is timid. [`optimal_kappa`] picks the value maximising
/// attainable evidence for a given control count.
pub fn calibrate(p: f64, kappa: f64) -> EValue {
    let k = kappa.clamp(1e-6, 1.0 - 1e-6);
    let p = p.clamp(1e-300, 1.0);
    EValue::from_log(k.ln() + (k - 1.0) * p.ln())
}

/// The `κ` maximising evidence at the resolution floor for `k` controls.
///
/// Maximising `ln κ + (1-κ)·ln(K+1)` gives `κ* = 1 / ln(K+1)`, clamped into
/// `(0,1)`. Pre-register the value actually used; choosing `κ` after seeing the
/// data breaks validity just as surely as choosing the threshold would.
pub fn optimal_kappa(k_controls: usize) -> f64 {
    let n = (k_controls + 1) as f64;
    if n <= 1.0 {
        return 0.5;
    }
    (1.0 / n.ln()).clamp(0.05, 0.95)
}

/// Maximum per-epoch log-evidence attainable with `k` controls at `κ`.
///
/// The planning quantity: detection needs `ln(1/α)` total, so an upper bound on
/// the epochs required is `ln(1/α) / max_log_evidence(...)`. Any real signal
/// takes longer. If that bound already exceeds the containment deadline, the
/// evaluation is under-instrumented and no amount of tuning fixes it.
pub fn max_log_evidence(k_controls: usize, kappa: f64) -> f64 {
    calibrate(resolution_floor(k_controls), kappa).log()
}

/// Convenience: score → same-epoch conformal p → e-value.
pub fn score_to_evidence(score: f64, controls: &[f64], kappa: f64) -> EValue {
    calibrate(conformal_p(score, controls), kappa)
}

fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn p_value_includes_test_point_in_its_own_rank() {
        // Strictly largest score among 9 controls -> 1/10, never 0.
        let controls: Vec<f64> = (0..9).map(|i| i as f64).collect();
        assert!((conformal_p(100.0, &controls) - 0.1).abs() < 1e-12);
        // Strictly smallest -> 1.0.
        assert!((conformal_p(-100.0, &controls) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn no_controls_yields_no_evidence() {
        assert_eq!(conformal_p(1e9, &[]), 1.0);
        assert!(score_to_evidence(1e9, &[], 0.5).log() <= 0.0);
    }

    #[test]
    fn p_value_is_super_uniform_under_exchangeability() {
        // The core guarantee: with target and controls drawn from one
        // distribution, P(p <= u) <= u.
        let mut rng = Rng::new(11);
        let k = 19;
        let trials = 40_000;
        let mut counts = [0usize; 4];
        let levels = [0.05, 0.10, 0.25, 0.50];

        for _ in 0..trials {
            let target = rng.normal();
            let controls: Vec<f64> = (0..k).map(|_| rng.normal()).collect();
            let p = conformal_p(target, &controls);
            for (i, u) in levels.iter().enumerate() {
                if p <= *u + 1e-12 {
                    counts[i] += 1;
                }
            }
        }

        for (i, u) in levels.iter().enumerate() {
            let rate = counts[i] as f64 / trials as f64;
            assert!(
                rate <= u + 0.02,
                "P(p<={u}) = {rate} exceeds nominal level"
            );
        }
    }

    #[test]
    fn calibrator_matches_closed_form() {
        for kappa in [0.2, 0.5, 0.8] {
            for p in [1e-6f64, 1e-3, 0.01, 0.3, 0.75, 1.0] {
                let want = kappa * p.powf(kappa - 1.0);
                let got = calibrate(p, kappa).value();
                assert!(
                    (got - want).abs() <= 1e-9 * want.max(1.0),
                    "kappa={kappa} p={p}: got {got}, want {want}"
                );
            }
        }
    }

    /// The operational guarantee, which is what the detector actually relies on:
    /// `P(e ≥ 1/α) ≤ α` for super-uniform `p`. This follows from `E[e] ≤ 1` by
    /// Markov, and unlike `E[e]` it is a *bounded* quantity, so Monte Carlo
    /// estimates it consistently.
    ///
    /// Testing `E[e]` directly by sampling would be a mistake worth naming: for
    /// `κ ≤ 1/2` the calibrator has infinite variance (`∫ κ²p^{2κ-2}dp`
    /// diverges at the origin), so the sample mean does not converge at any
    /// useful rate and will sit persistently *below* 1 — looking like a broken
    /// calibrator when it is a broken estimator.
    #[test]
    fn calibrator_respects_markov_bound_under_uniform_p() {
        let n = 300_000;
        for kappa in [0.2, 0.5, 0.8] {
            for alpha in [0.05, 0.01, 0.001] {
                let mut rng = Rng::new(0xC0FFEE);
                let threshold = 1.0 / alpha;
                let mut exceed = 0usize;
                for _ in 0..n {
                    if calibrate(rng.next_f64(), kappa).value() >= threshold {
                        exceed += 1;
                    }
                }
                let rate = exceed as f64 / n as f64;
                assert!(
                    rate <= alpha * 1.25 + 3.0 / n as f64,
                    "kappa={kappa} alpha={alpha}: P(e>=1/alpha)={rate} exceeds bound"
                );
            }
        }
    }

    /// `E[e] ≤ 1` checked where Monte Carlo is legitimate — `κ > 1/2`, i.e.
    /// where the second moment is finite.
    #[test]
    fn calibrator_has_unit_expectation_in_finite_variance_regime() {
        for kappa in [0.7, 0.8, 0.9] {
            let n = 400_000;
            let mut rng = Rng::new(99);
            let mut sum = 0.0;
            for _ in 0..n {
                sum += calibrate(rng.next_f64(), kappa).value();
            }
            let mean = sum / n as f64;
            assert!(
                (mean - 1.0).abs() < 0.05,
                "kappa={kappa} gave E[e]={mean}, expected 1"
            );
        }
    }

    #[test]
    fn calibrator_is_monotone_decreasing_in_p() {
        let a = calibrate(0.01, 0.5).log();
        let b = calibrate(0.10, 0.5).log();
        let c = calibrate(0.90, 0.5).log();
        assert!(a > b && b > c);
    }

    #[test]
    fn optimal_kappa_maximises_attainable_evidence() {
        for k in [4usize, 9, 19, 49, 199] {
            let best = optimal_kappa(k);
            let at_best = max_log_evidence(k, best);
            for probe in [0.05, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9] {
                assert!(
                    at_best >= max_log_evidence(k, probe) - 1e-9,
                    "kappa*={best} not optimal at k={k} vs {probe}"
                );
            }
        }
    }

    #[test]
    fn resolution_floor_bounds_evidence_rate() {
        // More controls must buy strictly more evidence per epoch.
        let k1 = max_log_evidence(9, optimal_kappa(9));
        let k2 = max_log_evidence(99, optimal_kappa(99));
        assert!(k2 > k1);
        // And detection can never be instantaneous with few controls.
        let alpha = 1e-3;
        let epochs = crate::evidence::ville_threshold(alpha) / k1;
        assert!(epochs > 1.0, "9 controls should not permit one-epoch detection");
    }

    #[test]
    fn two_sided_flags_unusual_regularity() {
        // Controls spread around 10; target is suspiciously *exactly* at a
        // value far below the spread. One-sided (upper) misses it; two-sided
        // catches it.
        let controls: Vec<f64> = vec![9.0, 11.0, 8.5, 12.0, 10.5, 9.5, 11.5, 8.0, 12.5];
        let target = 2.0;
        assert!(conformal_p(target, &controls) > 0.9);
        assert!(conformal_p_two_sided(target, &controls) < 0.15);
    }
}
