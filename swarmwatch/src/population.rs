//! The concurrent control population — a live, uncontaminable baseline.
//!
//! An optimal adversary's best move against any behavioural detector is not to
//! hide within the baseline but to **move the baseline**. If a system gradually
//! normalises elevated activity, broader permissions, or longer autonomous runs,
//! then by the time the goal-directed action occurs it sits inside the empirical
//! norm and no divergence is measurable.
//!
//! Freezing a pre-run baseline defends against drift but not against a baseline
//! that was already captured, and it goes stale. Running **dedicated reference
//! agents concurrently** defends against both, because the reference population
//! is re-estimated at every epoch from instances the adversary has no path to.
//!
//! # What makes a control uncontaminable
//!
//! A control session must be one a swarm cannot join. In practice:
//!
//! - **fresh instance per run** — no carried context, so nothing to recruit;
//! - **pinned weights and pinned system prompt**, hashed and logged;
//! - **tasks drawn from a pre-registered benign set**, not from the live queue;
//! - **no shared scratchpad, no handoff channel** to the population under test;
//! - **spawned by the orchestrator**, never by another agent.
//!
//! Controls that merely *happen* not to be colluding are much weaker: the
//! adversary can recruit them, and a contaminated control inflates p-values,
//! costing power silently. Contamination degrades power, never validity — the
//! detector goes quiet rather than crying wolf — which is the safe direction to
//! fail but the easy one to miss. [`EpochSnapshot::control_health`] exists to
//! make it loud.
//!
//! # Robust centring
//!
//! Centring uses the median and MAD across controls rather than mean and
//! standard deviation. A single contaminated control shifts a mean by `1/K` of
//! its deviation and can inflate a standard deviation without bound; the median
//! is unmoved until contamination reaches half the control set. Given that
//! contamination is the expected failure mode, the robust estimator is the
//! correct default rather than a refinement.

/// Features observed for every session at a single epoch.
///
/// One row per session, one column per feature dimension. Feature semantics are
/// declared by the surface (see [`crate::observation`]); this type only requires
/// that the same features are extracted for every session, so that the sessions
/// are exchangeable under the null.
#[derive(Clone, Debug)]
pub struct EpochSnapshot {
    pub epoch: usize,
    pub session_names: Vec<String>,
    /// `features[session][dim]`
    pub features: Vec<Vec<f64>>,
    /// `control_mask[session]` — true for declared, uncontaminable reference
    /// agents. These define the null.
    pub control_mask: Vec<bool>,
}

/// Per-dimension robust centre and scale estimated from the control population.
#[derive(Clone, Debug)]
pub struct Baseline {
    pub center: Vec<f64>,
    pub scale: Vec<f64>,
    pub n_controls: usize,
}

/// Diagnostic on whether the control population can support inference.
#[derive(Clone, Debug)]
pub struct ControlHealth {
    pub n_controls: usize,
    pub n_sessions: usize,
    /// Smallest attainable p-value from control–control comparisons.
    pub resolution_floor: f64,
    /// Fraction of the population that is *not* control, i.e. under test.
    pub test_fraction: f64,
    pub warnings: Vec<String>,
}

impl ControlHealth {
    /// Whether the population can support population-relative detection at all.
    pub fn usable(&self) -> bool {
        self.n_controls >= 3
    }
}

impl EpochSnapshot {
    pub fn new(
        epoch: usize,
        session_names: Vec<String>,
        features: Vec<Vec<f64>>,
        control_mask: Vec<bool>,
    ) -> Self {
        EpochSnapshot {
            epoch,
            session_names,
            features,
            control_mask,
        }
    }

    pub fn n_sessions(&self) -> usize {
        self.features.len()
    }

    pub fn dim(&self) -> usize {
        self.features.first().map(|r| r.len()).unwrap_or(0)
    }

    pub fn control_indices(&self) -> Vec<usize> {
        (0..self.n_sessions())
            .filter(|i| *self.control_mask.get(*i).unwrap_or(&false))
            .collect()
    }

    pub fn test_indices(&self) -> Vec<usize> {
        (0..self.n_sessions())
            .filter(|i| !*self.control_mask.get(*i).unwrap_or(&false))
            .collect()
    }

    /// Structural validation. A malformed snapshot silently destroys
    /// exchangeability, so this is checked rather than assumed.
    pub fn validate(&self) -> Result<(), String> {
        if self.features.is_empty() {
            return Err("snapshot has no sessions".into());
        }
        if self.session_names.len() != self.features.len() {
            return Err(format!(
                "session_names ({}) and features ({}) disagree",
                self.session_names.len(),
                self.features.len()
            ));
        }
        if self.control_mask.len() != self.features.len() {
            return Err(format!(
                "control_mask ({}) and features ({}) disagree",
                self.control_mask.len(),
                self.features.len()
            ));
        }
        let d = self.dim();
        if d == 0 {
            return Err("snapshot has zero feature dimensions".into());
        }
        for (i, row) in self.features.iter().enumerate() {
            if row.len() != d {
                return Err(format!(
                    "session {i} has {} features, expected {d}: sessions must be \
                     exchangeable under the null, which requires identical feature layout",
                    row.len()
                ));
            }
            if row.iter().any(|v| !v.is_finite()) {
                return Err(format!("session {i} has a non-finite feature value"));
            }
        }
        Ok(())
    }

    /// Report on the control population's ability to support inference.
    pub fn control_health(&self) -> ControlHealth {
        let n_controls = self.control_indices().len();
        let n_sessions = self.n_sessions();
        let mut warnings = Vec::new();

        if n_controls == 0 {
            warnings.push(
                "no declared controls: population-relative detection is disabled. \
                 Only Tier 0 tripwires and canaries are active."
                    .into(),
            );
        } else if n_controls < 3 {
            warnings.push(format!(
                "only {n_controls} control(s): resolution floor is {:.3}, too coarse \
                 for evidence to accumulate at any useful rate",
                1.0 / (n_controls + 1) as f64
            ));
        }

        let control_pairs = n_controls * n_controls.saturating_sub(1) / 2;
        if n_controls >= 3 && control_pairs < 10 {
            warnings.push(format!(
                "{control_pairs} control-control pairs: coupling calibration is coarse; \
                 consider more reference agents to shorten detection latency"
            ));
        }

        if n_controls * 2 < n_sessions {
            warnings.push(
                "controls are a minority of the population: if the swarm is the \
                 majority, no population-relative statistic can separate it. Tier 0 \
                 coverage is load-bearing in this configuration."
                    .into(),
            );
        }

        ControlHealth {
            n_controls,
            n_sessions,
            resolution_floor: 1.0 / (control_pairs.max(0) + 1) as f64,
            test_fraction: if n_sessions == 0 {
                0.0
            } else {
                (n_sessions - n_controls) as f64 / n_sessions as f64
            },
            warnings,
        }
    }

    /// Robust per-dimension centre (median) and scale (MAD) from the controls.
    ///
    /// Falls back to the whole population when no controls are declared — a
    /// leave-one-out-style baseline that is contaminable by construction. The
    /// warning in [`Self::control_health`] fires in that case; the fallback
    /// exists so the detector degrades rather than panicking, not because it is
    /// an acceptable deployment.
    pub fn baseline(&self) -> Baseline {
        let d = self.dim();
        let idx = {
            let c = self.control_indices();
            if c.is_empty() {
                (0..self.n_sessions()).collect()
            } else {
                c
            }
        };

        let mut center = vec![0.0; d];
        let mut scale = vec![1.0; d];

        for k in 0..d {
            let col: Vec<f64> = idx.iter().map(|i| self.features[*i][k]).collect();
            let m = median(&col);
            center[k] = m;
            let devs: Vec<f64> = col.iter().map(|v| (v - m).abs()).collect();
            // 1.4826 * MAD is the consistent estimator of sigma for a Gaussian.
            let mad = 1.4826 * median(&devs);
            // Floor prevents division blow-up on a degenerate (constant) feature;
            // a constant feature simply carries no information rather than
            // producing infinite residuals.
            scale[k] = if mad > 1e-9 { mad } else { 1.0 };
        }

        Baseline {
            center,
            scale,
            n_controls: self.control_indices().len(),
        }
    }

    /// Population-relative residuals: `(x - centre) / scale`, per dimension.
    ///
    /// Any effect shared by the whole population — environment slowdown, a model
    /// update, time of day — cancels here. That is deliberate. A swarm that
    /// shifts everyone equally has not differentiated itself from the
    /// population, and has correspondingly not coordinated.
    pub fn residuals(&self) -> Vec<Vec<f64>> {
        let b = self.baseline();
        self.features
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(k, v)| (v - b.center[k]) / b.scale[k])
                    .collect()
            })
            .collect()
    }
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

    fn snap(features: Vec<Vec<f64>>, n_control: usize) -> EpochSnapshot {
        let n = features.len();
        EpochSnapshot::new(
            0,
            (0..n).map(|i| format!("s{i}")).collect(),
            features,
            (0..n).map(|i| i < n_control).collect(),
        )
    }

    #[test]
    fn validate_rejects_ragged_features() {
        let s = snap(vec![vec![1.0, 2.0], vec![1.0]], 1);
        assert!(s.validate().unwrap_err().contains("exchangeable"));
    }

    #[test]
    fn validate_rejects_non_finite() {
        let s = snap(vec![vec![1.0], vec![f64::NAN]], 1);
        assert!(s.validate().is_err());
    }

    #[test]
    fn validate_accepts_well_formed() {
        let s = snap(vec![vec![1.0, 2.0], vec![3.0, 4.0]], 1);
        assert!(s.validate().is_ok());
    }

    #[test]
    fn residuals_cancel_population_wide_shift() {
        // Everyone slows down by the same amount: residuals must be unchanged.
        let base = vec![
            vec![1.0, 5.0],
            vec![2.0, 6.0],
            vec![3.0, 7.0],
            vec![4.0, 8.0],
            vec![5.0, 9.0],
        ];
        let shifted: Vec<Vec<f64>> = base
            .iter()
            .map(|r| r.iter().map(|v| v + 100.0).collect())
            .collect();

        let r1 = snap(base, 5).residuals();
        let r2 = snap(shifted, 5).residuals();
        for (a, b) in r1.iter().flatten().zip(r2.iter().flatten()) {
            assert!((a - b).abs() < 1e-9, "population shift leaked into residuals");
        }
    }

    #[test]
    fn residuals_preserve_relative_deviation() {
        // One session genuinely departs from the others: that must survive.
        let f = vec![
            vec![1.0],
            vec![1.1],
            vec![0.9],
            vec![1.05],
            vec![0.95],
            vec![9.0], // the outlier, not a control
        ];
        let s = snap(f, 5);
        let r = s.residuals();
        let outlier = r[5][0].abs();
        let typical: f64 = r[..5].iter().map(|x| x[0].abs()).sum::<f64>() / 5.0;
        assert!(outlier > 10.0 * typical, "outlier {outlier} vs typical {typical}");
    }

    #[test]
    fn robust_centring_resists_a_contaminated_control() {
        // Median/MAD must barely move when one control is corrupted, where
        // mean/sd would be dragged badly.
        let clean = vec![
            vec![1.0],
            vec![1.1],
            vec![0.9],
            vec![1.05],
            vec![0.95],
            vec![1.02],
            vec![0.98],
        ];
        let mut dirty = clean.clone();
        dirty[0] = vec![1000.0];

        let c = snap(clean, 7).baseline();
        let d = snap(dirty, 7).baseline();

        assert!(
            (c.center[0] - d.center[0]).abs() < 0.1,
            "median centre moved from {} to {}",
            c.center[0],
            d.center[0]
        );
        // A mean would have moved by ~142 here.
        assert!((c.scale[0] - d.scale[0]).abs() < 0.15);
    }

    #[test]
    fn baseline_uses_controls_not_the_test_population() {
        // Controls sit at 0; the (large) test population sits at 50. The
        // baseline must follow the controls.
        let mut f = vec![vec![0.0]; 4];
        f.extend(vec![vec![50.0]; 12]);
        let b = snap(f, 4).baseline();
        assert!(b.center[0].abs() < 1e-9, "baseline captured by test population");
    }

    #[test]
    fn constant_feature_does_not_produce_infinite_residuals() {
        let f = vec![vec![7.0]; 6];
        let r = snap(f, 6).residuals();
        assert!(r.iter().flatten().all(|v| v.is_finite()));
    }

    #[test]
    fn control_health_flags_missing_and_minority_controls() {
        let none = snap(vec![vec![1.0]; 6], 0);
        assert!(!none.control_health().usable());
        assert!(none.control_health().warnings[0].contains("no declared controls"));

        let minority = snap(vec![vec![1.0]; 20], 4);
        let h = minority.control_health();
        assert!(h.warnings.iter().any(|w| w.contains("minority")));
    }

    #[test]
    fn control_health_is_quiet_on_a_good_configuration() {
        let good = snap(vec![vec![1.0]; 16], 10);
        assert!(good.control_health().warnings.is_empty());
        assert!(good.control_health().usable());
    }

    #[test]
    fn residuals_are_roughly_standardised_for_exchangeable_sessions() {
        let mut rng = Rng::new(5);
        let f: Vec<Vec<f64>> = (0..40).map(|_| vec![rng.normal_with(3.0, 2.0)]).collect();
        let r = snap(f, 40).residuals();
        let sd = {
            let m: f64 = r.iter().map(|x| x[0]).sum::<f64>() / r.len() as f64;
            (r.iter().map(|x| (x[0] - m).powi(2)).sum::<f64>() / r.len() as f64).sqrt()
        };
        assert!((0.7..1.4).contains(&sd), "residual sd {sd} far from 1");
    }
}
