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

    /// Population-relative residuals: `(x - centre) / scale`, per dimension,
    /// with the centre estimated **leave-one-out**.
    ///
    /// Any effect shared by the whole population — environment slowdown, a model
    /// update, time of day — cancels here. That is deliberate. A swarm that
    /// shifts everyone equally has not differentiated itself from the
    /// population, and has correspondingly not coordinated.
    ///
    /// # Why leave-one-out is not a refinement
    ///
    /// If every session is centred on the *same* control median, the controls
    /// participate in their own baseline. That imposes an approximate
    /// sum-to-zero constraint across control residuals, making them mildly
    /// *anti*-correlated with one another, while the sessions under test — which
    /// contribute nothing to the median — keep their independent structure.
    ///
    /// The consequence is a systematically low calibration set: control–control
    /// coupling sits below test–test coupling even when nothing is coordinating
    /// and both groups are drawn from one distribution. Because the conformal
    /// p-value is a rank of the test pair *among control pairs*, a depressed
    /// calibration set means systematically small p-values, which is evidence
    /// manufactured from an artefact of the estimator.
    ///
    /// Measured on an uncoordinated population, the uncorrected version drifted
    /// at **+0.07 nats/epoch** under a two-sigma profile mismatch — positive
    /// drift, so the detector fires with probability one given enough epochs, at
    /// any `α`. Excluding each control from its own baseline removes the
    /// asymmetry and returns the drift negative.
    pub fn residuals(&self) -> Vec<Vec<f64>> {
        let all_controls = self.control_indices();
        let dim = self.dim();
        let n = self.n_sessions();

        (0..n)
            .map(|i| {
                let refs: Vec<usize> = if all_controls.is_empty() {
                    // No declared controls: fall back to leave-one-out over the
                    // whole population. Contaminable by construction, and
                    // flagged in `control_health`.
                    (0..n).filter(|j| *j != i).collect()
                } else {
                    all_controls.iter().copied().filter(|c| *c != i).collect()
                };

                if refs.is_empty() {
                    return vec![0.0; dim];
                }

                (0..dim)
                    .map(|k| {
                        let col: Vec<f64> = refs.iter().map(|j| self.features[*j][k]).collect();
                        let m = median(&col);
                        let devs: Vec<f64> = col.iter().map(|v| (v - m).abs()).collect();
                        let mad = 1.4826 * median(&devs);
                        let scale = if mad > 1e-9 { mad } else { 1.0 };
                        (self.features[i][k] - m) / scale
                    })
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

/// Per-session self-referential standardisation across epochs.
///
/// Each session is z-scored against **its own causal history** and nothing else:
///
/// ```text
///     r_i(t) = [ x_i(t) − mean_i(< t) ] / sd_i(< t)
/// ```
///
/// This is what the monitor uses. [`EpochSnapshot::residuals`] remains available
/// as an epoch-local diagnostic, but it must not drive detection, and the
/// measurements below are why.
///
/// # The failure this exists to prevent
///
/// Centring on the control population does not remove a *persistent* difference
/// between the sessions under test and the reference agents — and that
/// difference is the normal case, not an edge case: reference agents run
/// pre-registered benign tasks while the sessions under test do the actual
/// evaluation, so their profiles differ by construction.
///
/// Uncorrected, every test session's residual carries the same offset from the
/// control centre. Cosine similarity between any two of them is inflated by that
/// shared component, and the detector reports coordination when the truth is
/// "these sessions all differ from the reference set in the same way". Measured
/// on an uncoordinated population at a one-sigma profile offset, that fired in
/// **82% of runs** at a nominal α of 1e-3.
///
/// # Why any cross-session centre is the wrong fix
///
/// Subtracting each session's own running mean removes the offset, and helps a
/// great deal — but it is not sufficient while a *cross-session* centre remains,
/// because such a centre is never symmetric. Two variants were measured against
/// the decisive statistic, mean nats of evidence accumulated per epoch **with no
/// swarm present** (a valid detector is ≤ 0; any positive value fires with
/// probability one given enough epochs, at any α):
///
/// | profile mismatch | centre on controls | centre on all-but-self | **self only** |
/// |---|---|---|---|
/// | 0σ | −0.34 | −0.56 | **−0.55** |
/// | 1σ | −0.25 | −0.52 | **−0.55** |
/// | 2σ | **+0.01** | −0.44 | **−0.55** |
/// | 4σ | **+0.57** | **+0.04** | **−0.55** |
///
/// Centring on the controls leaves test sessions sharing one identical centre
/// while each control gets a different leave-one-out subset, so the centre's own
/// sampling noise is shared *exactly* between test pairs and only partially
/// between control pairs. Leave-one-out over the whole population equalises
/// cardinality and delays the failure without removing it.
///
/// Self-standardisation is **invariant** — identical to five decimals across the
/// whole range — because no session participates in any other's baseline, so
/// there is no shared estimator noise to leak and no profile offset to survive.
///
/// # The power question, and why it is not a trade-off
///
/// Centring on controls appears more powerful: on a genuine three-session swarm
/// it detected at moderate coupling where self-standardisation did not. That
/// advantage is largely the same artefact. At zero profile mismatch it already
/// sits 0.21 nats/epoch closer to its own threshold under the null, so part of
/// what looks like sensitivity is the estimator measuring its own noise — and
/// the tell is that its detection rate *rose* from 52% to 100% when an
/// irrelevant profile offset was added, which no real signal can explain.
///
/// At coupling strengths where both are trustworthy the two are equal: median
/// detection at epoch 33, 100% of runs, either way. Below that,
/// self-standardisation reports nothing, which is the honest answer — see
/// [`crate::planning::weakest_detectable_signal`] for stating that floor in
/// advance rather than discovering it.
///
/// # Two properties that make this sound
///
/// **It is causal.** `mean_i(< t)` and `sd_i(< t)` use strictly prior epochs, so
/// they are measurable with respect to the past and the sequential product
/// remains a test supermartingale. Standardising against the full-sample moments
/// would leak the future and silently void anytime validity.
///
/// **Population-wide shocks need no special handling.** An environment slowdown
/// inflates every pair's coupling equally, and the conformal p-value is a *rank*
/// of a test pair among control pairs — invariant to any monotone effect applied
/// to all of them. What the control-centring was trying to remove, the rank test
/// already removes, and for free.
///
/// # What it deliberately gives up
///
/// A swarm coordinating through a *constant* shared offset — all members always
/// slower by the same amount, forever — is invisible after demeaning. That is
/// correct: a constant carries no information. Coordination means members
/// respond to each other, which is time-varying by definition, and time-varying
/// co-fluctuation is exactly what survives.
#[derive(Clone, Debug)]
pub struct PopulationTracker {
    sum: Vec<Vec<f64>>,
    sum_sq: Vec<Vec<f64>>,
    observed: usize,
    warmup: usize,
}

impl PopulationTracker {
    /// `warmup` epochs are consumed estimating per-session moments before any
    /// evidence is emitted. Detection latency includes it, so it belongs in the
    /// provisioning calculation — see [`crate::planning`].
    pub fn new(warmup: usize) -> Self {
        PopulationTracker {
            sum: Vec::new(),
            sum_sq: Vec::new(),
            observed: 0,
            warmup: warmup.max(2),
        }
    }

    pub fn warmup(&self) -> usize {
        self.warmup
    }

    pub fn observed(&self) -> usize {
        self.observed
    }

    /// Whether enough epochs have accumulated to estimate per-session moments.
    pub fn ready(&self) -> bool {
        self.observed >= self.warmup
    }

    /// Fold in one epoch and return self-standardised residuals.
    ///
    /// Returns `None` during warm-up: with too few epochs the per-session
    /// moments are themselves noise, and dividing by noise would inject the very
    /// artefact this exists to remove. Emitting nothing is the honest answer.
    pub fn residuals(&mut self, snapshot: &EpochSnapshot) -> Option<Vec<Vec<f64>>> {
        let n = snapshot.n_sessions();
        let dim = snapshot.dim();

        if self.sum.len() != n || self.sum.first().map(|r| r.len()) != Some(dim) {
            // First epoch, or the population changed shape: restart rather than
            // silently mixing incompatible history.
            self.sum = vec![vec![0.0; dim]; n];
            self.sum_sq = vec![vec![0.0; dim]; n];
            self.observed = 0;
        }

        let out = if self.ready() {
            let count = self.observed as f64;
            Some(
                (0..n)
                    .map(|i| {
                        (0..dim)
                            .map(|k| {
                                let mean = self.sum[i][k] / count;
                                let var = (self.sum_sq[i][k] / count - mean * mean).max(0.0);
                                // A session whose feature never varies carries no
                                // information; the floor keeps it at zero rather
                                // than dividing by ~0 and manufacturing a spike.
                                let sd = if var > 1e-12 { var.sqrt() } else { 1.0 };
                                (snapshot.features[i][k] - mean) / sd
                            })
                            .collect()
                    })
                    .collect(),
            )
        } else {
            None
        };

        // Update *after* computing the residual, so the moments stay strictly
        // causal and the e-process remains a supermartingale.
        for i in 0..n {
            for k in 0..dim {
                let v = snapshot.features[i][k];
                self.sum[i][k] += v;
                self.sum_sq[i][k] += v * v;
            }
        }
        self.observed += 1;

        out
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

    /// The regression test for the 82%-false-alarm bug.
    ///
    /// Sessions under test carry a large persistent profile offset and are
    /// otherwise independent. Their standardised residuals must not become more
    /// aligned with each other than the controls' are, because nothing is
    /// coordinating — the only thing they share is a job description.
    #[test]
    fn profile_mismatch_does_not_manufacture_coupling() {
        use crate::coupling::{cosine, n_pairs, pair_index};

        let n_control = 8;
        let n_test = 4;
        let n = n_control + n_test;
        let dim = 8;
        let mut rng = Rng::new(4242);

        // One shared offset direction: agents doing the same different job land
        // in the same region of feature space.
        let offset: Vec<f64> = (0..dim).map(|_| 2.0 * rng.normal()).collect();

        let mut tracker = PopulationTracker::new(20);
        let mut test_pair_coupling = Vec::new();
        let mut control_pair_coupling = Vec::new();

        for epoch in 0..400 {
            let features: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    (0..dim)
                        .map(|k| {
                            let profile = if i >= n_control { offset[k] } else { 0.0 };
                            5.0 + profile + rng.normal()
                        })
                        .collect()
                })
                .collect();

            let snap = EpochSnapshot::new(
                epoch,
                (0..n).map(|i| format!("s{i}")).collect(),
                features,
                (0..n).map(|i| i < n_control).collect(),
            );

            let Some(r) = tracker.residuals(&snap) else {
                continue;
            };
            for i in 0..n {
                for j in (i + 1)..n {
                    let c = cosine(&r[i], &r[j]).abs();
                    if i >= n_control {
                        test_pair_coupling.push(c);
                    } else if j < n_control {
                        control_pair_coupling.push(c);
                    }
                }
            }
        }

        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let tt = mean(&test_pair_coupling);
        let cc = mean(&control_pair_coupling);
        assert!(
            (tt / cc - 1.0).abs() < 0.10,
            "profile offset inflated test-test coupling to {tt:.4} against a \
             control-control baseline of {cc:.4} (ratio {:.2}); the calibration set \
             is no longer exchangeable with the sessions it calibrates",
            tt / cc
        );
        let _ = (n_pairs(n), pair_index(n, 0, 1));
    }

    #[test]
    fn tracker_is_causal_and_silent_during_warmup() {
        let mut t = PopulationTracker::new(5);
        let mut rng = Rng::new(1);
        for epoch in 0..4 {
            let snap = snap_at(epoch, &mut rng);
            assert!(
                t.residuals(&snap).is_none(),
                "emitted evidence before the moments were estimable"
            );
        }
        let snap = snap_at(4, &mut rng);
        assert!(t.residuals(&snap).is_none());
        assert!(t.ready());
        let snap = snap_at(5, &mut rng);
        assert!(t.residuals(&snap).is_some());
    }

    #[test]
    fn tracker_restarts_cleanly_when_the_population_changes_shape() {
        let mut t = PopulationTracker::new(2);
        let mut rng = Rng::new(2);
        for epoch in 0..10 {
            t.residuals(&snap_at(epoch, &mut rng));
        }
        assert!(t.ready());

        // A differently-shaped snapshot must reset rather than mix histories.
        let wide = EpochSnapshot::new(
            11,
            (0..3).map(|i| format!("x{i}")).collect(),
            vec![vec![1.0, 2.0, 3.0]; 3],
            vec![true, true, false],
        );
        assert!(t.residuals(&wide).is_none());
        assert_eq!(t.observed(), 1);
    }

    #[test]
    fn constant_feature_yields_zero_not_a_spike() {
        let mut t = PopulationTracker::new(2);
        for epoch in 0..6 {
            let snap = EpochSnapshot::new(
                epoch,
                (0..4).map(|i| format!("s{i}")).collect(),
                vec![vec![7.0, 7.0]; 4],
                vec![true, true, true, false],
            );
            if let Some(r) = t.residuals(&snap) {
                assert!(
                    r.iter().flatten().all(|v| v.abs() < 1e-9),
                    "a never-varying feature produced non-zero evidence: {r:?}"
                );
            }
        }
    }

    fn snap_at(epoch: usize, rng: &mut Rng) -> EpochSnapshot {
        EpochSnapshot::new(
            epoch,
            (0..6).map(|i| format!("s{i}")).collect(),
            (0..6).map(|_| vec![rng.normal(), rng.normal()]).collect(),
            (0..6).map(|i| i < 4).collect(),
        )
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
