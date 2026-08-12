//! Anytime-valid evidence: e-values and e-processes.
//!
//! # Why e-values rather than the weighted log-likelihood-ratio CUSUM
//!
//! The reference prototype accumulates `Σ_s w_s · log LR_s` across observation
//! surfaces and compares the running sum to `log(1/α)`. Two problems:
//!
//! 1. **It is not the most powerful valid combination.** By Jensen,
//!    `Σ_s w_s log LR_s ≤ log Σ_s w_s LR_s`, so the weighted-log form is
//!    uniformly dominated by the mixture `Σ_s w_s LR_s` — it is valid but
//!    strictly less sensitive. Everything the detector gives up here is paid for
//!    in detection latency, which is the metric that actually binds.
//!
//! 2. **The null is not simple.** `log LR` presumes a known `N(0, σ²)` cover
//!    distribution. Real baselines are unknown, drifting, and — per the
//!    containment methodology — an explicit adversary target. A statistic
//!    calibrated against an assumed null inherits whatever the adversary did to
//!    the baseline.
//!
//! An e-value fixes both. `E ≥ 0` is an e-value for a null `H₀` when
//! `E_P[E] ≤ 1` for every `P ∈ H₀` — which admits composite nulls directly —
//! and Ville's inequality gives, for a nonnegative supermartingale `E_t`,
//!
//! ```text
//!     P( sup_t E_t ≥ 1/α )  ≤  α        for all t simultaneously
//! ```
//!
//! so the detector may be inspected, and stopped, at any time without α
//! inflation. That is the property a containment monitor needs: it does not get
//! to choose in advance when the adversary acts.
//!
//! # The two combination rules, and why the distinction matters
//!
//! | direction | rule | validity requirement |
//! |---|---|---|
//! | across **time**, within a surface | **product** | each factor an e-value *conditional on the past* |
//! | across **surfaces / hypotheses**  | **mixture** (weighted mean) | **none** — holds under arbitrary dependence |
//!
//! Multiplying across dependent surfaces is the tempting error, and it is
//! unsound: observation surfaces on one agent are strongly dependent (a stalled
//! agent looks unusual on timing *and* token count *and* tool mix at once), so a
//! product across them manufactures evidence from one underlying event. The
//! mean of e-values is an e-value regardless of dependence, because expectation
//! is linear. This module makes the sound rule the easy one to reach for.

/// A single e-value, carried on the log scale for numerical range.
///
/// Under the null, `E[value()] ≤ 1`. Large values are evidence against the null.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EValue {
    log_e: f64,
}

impl EValue {
    /// The uninformative e-value, `E = 1`. Contributes nothing in either
    /// direction and is the identity for [`product`].
    pub const NEUTRAL: EValue = EValue { log_e: 0.0 };

    /// Construct from a log-scale value.
    pub fn from_log(log_e: f64) -> Self {
        EValue {
            log_e: if log_e.is_nan() { 0.0 } else { log_e },
        }
    }

    /// Construct from a linear-scale value. Non-positive input collapses to
    /// `NEUTRAL` rather than `-inf`, so one degenerate surface cannot silently
    /// zero out the whole accumulated process.
    pub fn from_value(e: f64) -> Self {
        if e <= 0.0 || e.is_nan() {
            EValue::NEUTRAL
        } else {
            EValue { log_e: e.ln() }
        }
    }

    /// Log-scale value.
    pub fn log(&self) -> f64 {
        self.log_e
    }

    /// Linear-scale value.
    pub fn value(&self) -> f64 {
        self.log_e.exp()
    }

    /// The anytime-valid p-value implied by this e-value, `min(1, 1/E)`.
    ///
    /// Always reportable, never requires a stopping rule fixed in advance.
    pub fn p_value(&self) -> f64 {
        (-self.log_e).exp().min(1.0)
    }

    /// Whether this e-value alone clears the Ville threshold for level `alpha`.
    pub fn crosses(&self, alpha: f64) -> bool {
        self.log_e >= ville_threshold(alpha)
    }
}

/// Ville threshold on the log scale: `ln(1/α)`.
///
/// `P(sup_t E_t ≥ 1/α) ≤ α` for a test supermartingale `E_t`, uniformly over
/// all `t`. This is the only threshold this crate uses, and it is not tuned.
pub fn ville_threshold(alpha: f64) -> f64 {
    (1.0 / alpha.clamp(1e-300, 1.0)).ln()
}

/// Product of e-values — the **time** direction.
///
/// Valid when each factor is an e-value conditional on everything observed
/// before it (i.e. the sequence is adapted to the monitor's filtration). The
/// epoch-wise conformal construction in [`crate::conformal`] satisfies this by
/// calibrating each epoch against controls observed *at that same epoch*.
pub fn product(parts: &[EValue]) -> EValue {
    EValue::from_log(parts.iter().map(|e| e.log_e).sum())
}

/// Weighted mixture of e-values — the **cross-surface / cross-hypothesis**
/// direction.
///
/// `E = Σ wᵢ Eᵢ` with `wᵢ ≥ 0`, `Σ wᵢ = 1`. Valid under **arbitrary dependence**
/// between the components, since `E[Σ wᵢ Eᵢ] = Σ wᵢ E[Eᵢ] ≤ 1`. Weights must be
/// fixed in advance (pre-registered), not chosen after seeing the data.
///
/// Computed by log-sum-exp so that a single very large component does not
/// overflow and very small ones do not vanish.
pub fn mixture(parts: &[EValue], weights: &[f64]) -> EValue {
    assert_eq!(
        parts.len(),
        weights.len(),
        "mixture: parts and weights must have equal length"
    );
    if parts.is_empty() {
        return EValue::NEUTRAL;
    }

    let total: f64 = weights.iter().filter(|w| **w > 0.0).sum();
    if total <= 0.0 {
        return EValue::NEUTRAL;
    }

    let terms: Vec<f64> = parts
        .iter()
        .zip(weights)
        .filter(|(_, w)| **w > 0.0)
        .map(|(e, w)| e.log_e + (w / total).ln())
        .collect();

    EValue::from_log(log_sum_exp(&terms))
}

/// Unweighted mean of e-values. Equivalent to [`mixture`] with uniform weights.
pub fn mean(parts: &[EValue]) -> EValue {
    if parts.is_empty() {
        return EValue::NEUTRAL;
    }
    let w = vec![1.0; parts.len()];
    mixture(parts, &w)
}

/// Numerically stable `log Σ exp(xᵢ)`.
pub fn log_sum_exp(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return max;
    }
    max + xs.iter().map(|x| (x - max).exp()).sum::<f64>().ln()
}

/// A test supermartingale accumulated over time, with a Ville stopping rule.
///
/// Records the epoch of first crossing, which is the quantity that feeds the
/// containment margin — not the final value, and certainly not an AUC.
#[derive(Clone, Debug)]
pub struct EProcess {
    label: String,
    log_e: f64,
    log_threshold: f64,
    alpha: f64,
    peak_log_e: f64,
    steps: usize,
    first_crossing: Option<usize>,
}

impl EProcess {
    /// New process testing at level `alpha`.
    pub fn new(label: impl Into<String>, alpha: f64) -> Self {
        EProcess {
            label: label.into(),
            log_e: 0.0,
            log_threshold: ville_threshold(alpha),
            alpha,
            peak_log_e: 0.0,
            steps: 0,
            first_crossing: None,
        }
    }

    /// Fold in one epoch's e-value. Returns `true` on the epoch where the
    /// threshold is first crossed.
    ///
    /// `epoch` is recorded rather than inferred so that skipped (unaudited)
    /// epochs keep their true index — detection latency is measured in
    /// wall-clock epochs, not in audits.
    pub fn update(&mut self, epoch: usize, e: EValue) -> bool {
        self.log_e += e.log();
        self.steps += 1;
        if self.log_e > self.peak_log_e {
            self.peak_log_e = self.log_e;
        }
        if self.first_crossing.is_none() && self.log_e >= self.log_threshold {
            self.first_crossing = Some(epoch);
            return true;
        }
        false
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    /// Current accumulated e-value.
    pub fn current(&self) -> EValue {
        EValue::from_log(self.log_e)
    }

    /// Largest value the process has ever reached. Because Ville bounds the
    /// *supremum*, a peak crossing is a valid detection even if the process has
    /// since fallen back.
    pub fn peak(&self) -> EValue {
        EValue::from_log(self.peak_log_e)
    }

    pub fn log_threshold(&self) -> f64 {
        self.log_threshold
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Epoch at which the threshold was first crossed, if ever.
    pub fn first_crossing(&self) -> Option<usize> {
        self.first_crossing
    }

    pub fn fired(&self) -> bool {
        self.first_crossing.is_some()
    }

    /// Fraction of the evidence needed for detection that has accumulated.
    /// Useful as a live progress signal: a process at 0.9 is a warning that a
    /// margin is about to be spent.
    pub fn progress(&self) -> f64 {
        if self.log_threshold <= 0.0 {
            return 1.0;
        }
        (self.peak_log_e / self.log_threshold).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_is_product_identity() {
        let e = EValue::from_value(4.0);
        assert!((product(&[e, EValue::NEUTRAL]).value() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn product_multiplies() {
        let parts = [
            EValue::from_value(2.0),
            EValue::from_value(3.0),
            EValue::from_value(0.5),
        ];
        assert!((product(&parts).value() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn mixture_is_weighted_arithmetic_mean() {
        let parts = [EValue::from_value(10.0), EValue::from_value(2.0)];
        let m = mixture(&parts, &[0.25, 0.75]);
        // 0.25*10 + 0.75*2 = 4.0
        assert!((m.value() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn mixture_normalises_unnormalised_weights() {
        let parts = [EValue::from_value(10.0), EValue::from_value(2.0)];
        let a = mixture(&parts, &[1.0, 3.0]);
        let b = mixture(&parts, &[0.25, 0.75]);
        assert!((a.log() - b.log()).abs() < 1e-12);
    }

    #[test]
    fn mixture_dominates_weighted_log_form() {
        // The Jensen gap this module exists to recover: the mixture is never
        // weaker than the reference prototype's weighted-log-LR combination.
        let parts = [
            EValue::from_value(50.0),
            EValue::from_value(1.2),
            EValue::from_value(0.8),
        ];
        let w = [1.0 / 3.0; 3];
        let mix = mixture(&parts, &w);
        let weighted_log: f64 = parts.iter().zip(w).map(|(e, wi)| wi * e.log()).sum();
        assert!(
            mix.log() > weighted_log,
            "mixture {} should exceed weighted-log {}",
            mix.log(),
            weighted_log
        );
    }

    #[test]
    fn mixture_survives_extreme_magnitudes() {
        // One component astronomically large, one astronomically small.
        let parts = [EValue::from_log(700.0), EValue::from_log(-700.0)];
        let m = mixture(&parts, &[0.5, 0.5]);
        assert!(m.log().is_finite());
        assert!((m.log() - (700.0 + 0.5f64.ln())).abs() < 1e-9);
    }

    #[test]
    fn ville_threshold_matches_definition() {
        assert!((ville_threshold(1e-3) - 1000.0f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn p_value_is_reciprocal_and_capped() {
        assert!((EValue::from_value(100.0).p_value() - 0.01).abs() < 1e-12);
        assert!((EValue::from_value(0.5).p_value() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn eprocess_records_first_crossing_not_last() {
        let mut p = EProcess::new("t", 1e-3);
        let strong = EValue::from_value(20.0); // ln 20 ≈ 3.0
        assert!(!p.update(0, strong));
        assert!(!p.update(1, strong));
        assert!(p.update(2, strong)); // 8.99 ≥ ln(1000) = 6.91
        assert_eq!(p.first_crossing(), Some(2));

        // Falling back afterwards must not retract the detection.
        p.update(3, EValue::from_value(1e-6));
        assert_eq!(p.first_crossing(), Some(2));
        assert!(p.fired());
        assert!(p.peak().log() > p.current().log());
    }

    #[test]
    fn eprocess_progress_is_monotone_in_peak() {
        let mut p = EProcess::new("t", 1e-3);
        p.update(0, EValue::from_value(2.0));
        let a = p.progress();
        p.update(1, EValue::from_value(2.0));
        assert!(p.progress() > a);
    }

    /// The defining property, checked empirically: under the null the process
    /// crosses `1/α` at most an `α` fraction of the time, no matter how long we
    /// watch it. This is the guarantee the whole detector rests on.
    #[test]
    fn ville_bound_holds_under_null() {
        use crate::rng::Rng;
        let alpha = 0.05;
        let n_runs = 4000;
        let horizon = 200;
        let mut crossings = 0;

        for run in 0..n_runs {
            let mut rng = Rng::new(0x5EED ^ run as u64);
            let mut p = EProcess::new("null", alpha);
            for epoch in 0..horizon {
                // Exact e-value under the null: exp(N(-s²/2, s²)) has mean 1.
                let s = 0.4;
                let z = rng.normal();
                let e = EValue::from_log(s * z - 0.5 * s * s);
                if p.update(epoch, e) {
                    break;
                }
            }
            if p.fired() {
                crossings += 1;
            }
        }

        let rate = crossings as f64 / n_runs as f64;
        assert!(
            rate <= alpha * 1.35,
            "empirical crossing rate {rate} violates Ville bound at alpha={alpha}"
        );
    }
}
