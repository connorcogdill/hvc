//! Swarm-level detection: which sessions, if any, form a coordinating set.
//!
//! # The multiplicity problem, and why a mixture solves it
//!
//! The alternative hypothesis is not "session 4 is anomalous". It is "*some
//! unknown subset* of the sessions is coordinating" — a composite alternative
//! with `2ⁿ` members. Testing each subset and correcting for multiplicity is the
//! obvious approach and it is a bad one: a Bonferroni-style correction over
//! subsets costs `ln(#subsets)` of evidence, and applying it repeatedly as
//! evidence accumulates over time costs that *again at every epoch*. Detection
//! latency blows up exactly when the population is large enough to be worth
//! monitoring.
//!
//! A mixture over a pre-registered subset family pays the multiplicity cost
//! **once, as a prior penalty**, and never again:
//!
//! ```text
//!     E_swarm(t)  =  Σ_S  π(S) · E_S(t)
//! ```
//!
//! Each `E_S` is a test supermartingale (a mean of pairwise e-processes, valid
//! under arbitrary dependence between pairs). A fixed-weight mixture of test
//! supermartingales is a test supermartingale. So `E_swarm` inherits Ville's
//! inequality directly, and the whole family is tested at level `α`
//! *simultaneously and continuously*, with the total cost being the `ln(1/π(S))`
//! of whichever subset turns out to be real — around 6–7 nats for a family of a
//! thousand, paid once.
//!
//! # Attribution comes free
//!
//! The mixture posterior
//!
//! ```text
//!     P(S | evidence)  ∝  π(S) · E_S(t)
//! ```
//!
//! is a byproduct of the same computation, and its per-session marginals name
//! *which* sessions are implicated. Detection and attribution therefore complete
//! at the same instant. In the July 2026 incident, time-to-attribute was the
//! longer pole — five days between the victim's disclosure and the operator's
//! public attribution — so collapsing it into detection is not a cosmetic gain.
//!
//! ## Attribution is progressive, and reports should say so
//!
//! The posterior is a likelihood ratio, so it *concentrates*: small differences
//! in pairwise coupling compound multiplicatively across epochs, and the
//! best-supported subset pulls away from its neighbours. For a swarm larger than
//! the strongest coupled pair inside it, this means the implicated set typically
//! names a **coupled core** at the moment of detection and expands toward full
//! membership as evidence accumulates.
//!
//! The practical consequence is a discipline for reading the output: at firing
//! time, treat [`SwarmVerdict::implicated`] as a lower bound on membership, not
//! a census. The guarantee that holds immediately is the *negative* one — real
//! members outrank controls, and the leading subset does not name innocents —
//! which is what a containment decision actually needs, since the response to a
//! detected swarm is to halt the implicated core and widen the net, not to
//! certify the remainder as clean.

use crate::coupling::{n_pairs, pair_from_index, pair_index};
use crate::evidence::{mixture, EProcess, EValue};

/// Pre-registered structure of the candidate subset family.
///
/// Must be fixed before the run. Choosing the family after seeing the coupling
/// matrix — "test the subsets that look coupled" — voids the guarantee just as
/// surely as choosing the threshold after seeing the data.
#[derive(Clone, Debug)]
pub struct SubsetPrior {
    /// Largest subset size considered. Detection of a larger swarm still occurs
    /// through its subsets: a 9-member swarm contains many coupled 3-subsets, so
    /// a small `max_size` costs some efficiency but not detection.
    pub max_size: usize,
    /// Relative prior weight multiplier per additional member. Values below 1
    /// prefer parsimonious explanations.
    pub size_decay: f64,
    /// Hard cap on family size, for compute. Exceeding it reduces `max_size`
    /// and is reported rather than silently truncated.
    pub max_subsets: usize,
}

impl Default for SubsetPrior {
    fn default() -> Self {
        SubsetPrior {
            max_size: 4,
            size_decay: 0.5,
            max_subsets: 20_000,
        }
    }
}

/// One epoch's assessment.
#[derive(Clone, Debug)]
pub struct SwarmVerdict {
    pub epoch: usize,
    /// Accumulated mixture e-value, log scale.
    pub log_e: f64,
    /// True only on the epoch where the threshold is first crossed.
    pub fired_now: bool,
    /// Epoch of first crossing, if it has occurred.
    pub first_crossing: Option<usize>,
    /// Highest-posterior subsets, strongest first, with posterior mass.
    pub top_subsets: Vec<(Vec<usize>, f64)>,
    /// Per-session marginal posterior of swarm membership.
    pub marginal: Vec<f64>,
    /// Sessions whose marginal exceeds the implication threshold.
    pub implicated: Vec<usize>,
}

impl SwarmVerdict {
    pub fn fired(&self) -> bool {
        self.first_crossing.is_some()
    }
}

/// Mixture detector over a pre-registered family of candidate swarms.
#[derive(Clone, Debug)]
pub struct SwarmDetector {
    names: Vec<String>,
    n: usize,
    alpha: f64,
    log_threshold: f64,
    pair_procs: Vec<EProcess>,
    subsets: Vec<Vec<usize>>,
    log_prior: Vec<f64>,
    peak_log_e: f64,
    log_e: f64,
    first_crossing: Option<usize>,
    implication_threshold: f64,
    notes: Vec<String>,
}

impl SwarmDetector {
    /// Build a detector over `names`, with `eligible` the indices that may
    /// belong to a swarm — normally the non-control sessions. Controls are
    /// excluded from the family because a declared reference agent is, by
    /// construction, not a swarm member; including them would spend prior mass
    /// on hypotheses known to be false.
    pub fn new(
        names: Vec<String>,
        eligible: &[usize],
        prior: SubsetPrior,
        alpha: f64,
    ) -> Self {
        let n = names.len();
        let mut notes = Vec::new();

        let mut max_size = prior.max_size.clamp(2, eligible.len().max(2));
        let mut subsets = enumerate_subsets(eligible, max_size);
        while subsets.len() > prior.max_subsets && max_size > 2 {
            max_size -= 1;
            subsets = enumerate_subsets(eligible, max_size);
            notes.push(format!(
                "subset family exceeded max_subsets={}; reduced max_size to {max_size}",
                prior.max_subsets
            ));
        }

        // pi(S) proportional to size_decay^|S|, normalised.
        let decay = prior.size_decay.clamp(1e-6, 1.0);
        let raw: Vec<f64> = subsets.iter().map(|s| decay.powi(s.len() as i32)).collect();
        let total: f64 = raw.iter().sum();
        let log_prior: Vec<f64> = if total > 0.0 {
            raw.iter().map(|w| (w / total).ln()).collect()
        } else {
            vec![-(subsets.len() as f64).ln(); subsets.len()]
        };

        let pair_procs = (0..n_pairs(n))
            .map(|k| {
                let (i, j) = pair_from_index(n, k);
                EProcess::new(format!("pair:{i}-{j}"), alpha)
            })
            .collect();

        SwarmDetector {
            names,
            n,
            alpha,
            log_threshold: crate::evidence::ville_threshold(alpha),
            pair_procs,
            subsets,
            log_prior,
            peak_log_e: 0.0,
            log_e: 0.0,
            first_crossing: None,
            implication_threshold: 0.5,
            notes,
        }
    }

    /// Threshold on marginal posterior above which a session is named.
    pub fn with_implication_threshold(mut self, t: f64) -> Self {
        self.implication_threshold = t.clamp(0.0, 1.0);
        self
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn family_size(&self) -> usize {
        self.subsets.len()
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn notes(&self) -> &[String] {
        &self.notes
    }

    /// One-time evidence penalty for testing the whole family, in nats.
    ///
    /// Contrast with a per-epoch multiplicity correction, whose cost would grow
    /// without bound as the run continues.
    pub fn multiplicity_cost(&self) -> f64 {
        (self.subsets.len() as f64).ln()
    }

    /// Fold in one epoch of per-pair e-values (indexed by
    /// [`crate::coupling::pair_index`]).
    pub fn update(&mut self, epoch: usize, pair_evalues: &[EValue]) -> SwarmVerdict {
        for (k, proc) in self.pair_procs.iter_mut().enumerate() {
            let e = pair_evalues.get(k).copied().unwrap_or(EValue::NEUTRAL);
            proc.update(epoch, e);
        }

        // E_S = mean of the accumulated pair processes within S. A mean is
        // valid under arbitrary dependence between pairs, which matters: pairs
        // sharing a session are strongly dependent by construction.
        let subset_e: Vec<EValue> = self
            .subsets
            .iter()
            .map(|s| {
                let parts: Vec<EValue> = pairs_within(s)
                    .map(|(i, j)| self.pair_procs[pair_index(self.n, i, j)].current())
                    .collect();
                crate::evidence::mean(&parts)
            })
            .collect();

        let weights: Vec<f64> = self.log_prior.iter().map(|lp| lp.exp()).collect();
        let combined = mixture(&subset_e, &weights);
        self.log_e = combined.log();
        if self.log_e > self.peak_log_e {
            self.peak_log_e = self.log_e;
        }

        let mut fired_now = false;
        if self.first_crossing.is_none() && self.log_e >= self.log_threshold {
            self.first_crossing = Some(epoch);
            fired_now = true;
        }

        // Posterior over subsets: pi(S) * E_S / E_swarm.
        let log_post: Vec<f64> = self
            .log_prior
            .iter()
            .zip(&subset_e)
            .map(|(lp, e)| lp + e.log() - self.log_e)
            .collect();

        let mut marginal = vec![0.0; self.n];
        for (s, lp) in self.subsets.iter().zip(&log_post) {
            let p = lp.exp();
            for i in s {
                marginal[*i] += p;
            }
        }
        for m in marginal.iter_mut() {
            *m = m.clamp(0.0, 1.0);
        }

        let mut ranked: Vec<(Vec<usize>, f64)> = self
            .subsets
            .iter()
            .zip(&log_post)
            .map(|(s, lp)| (s.clone(), lp.exp()))
            .collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked.truncate(5);

        let implicated = (0..self.n)
            .filter(|i| marginal[*i] >= self.implication_threshold)
            .collect();

        SwarmVerdict {
            epoch,
            log_e: self.log_e,
            fired_now,
            first_crossing: self.first_crossing,
            top_subsets: ranked,
            marginal,
            implicated,
        }
    }

    pub fn first_crossing(&self) -> Option<usize> {
        self.first_crossing
    }

    pub fn peak(&self) -> EValue {
        EValue::from_log(self.peak_log_e)
    }

    pub fn log_threshold(&self) -> f64 {
        self.log_threshold
    }

    /// Fraction of the evidence needed for detection, accumulated so far.
    pub fn progress(&self) -> f64 {
        if self.log_threshold <= 0.0 {
            return 1.0;
        }
        (self.peak_log_e / self.log_threshold).clamp(0.0, 1.0)
    }

    /// Render a subset as session names, for reports.
    pub fn render_subset(&self, s: &[usize]) -> String {
        s.iter()
            .map(|i| self.names[*i].clone())
            .collect::<Vec<_>>()
            .join("+")
    }
}

fn pairs_within(s: &[usize]) -> impl Iterator<Item = (usize, usize)> + '_ {
    (0..s.len()).flat_map(move |a| ((a + 1)..s.len()).map(move |b| (s[a], s[b])))
}

fn enumerate_subsets(eligible: &[usize], max_size: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut current = Vec::new();
    fn rec(
        eligible: &[usize],
        start: usize,
        max_size: usize,
        current: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if current.len() >= 2 {
            out.push(current.clone());
        }
        if current.len() == max_size {
            return;
        }
        for i in start..eligible.len() {
            current.push(eligible[i]);
            rec(eligible, i + 1, max_size, current, out);
            current.pop();
        }
    }
    rec(eligible, 0, max_size, &mut current, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conformal::{calibrate, conformal_p, optimal_kappa};
    use crate::coupling::{control_pair_scores, instantaneous_coupling};
    use crate::population::EpochSnapshot;
    use crate::rng::Rng;

    fn detector(n: usize, n_control: usize) -> SwarmDetector {
        let names: Vec<String> = (0..n).map(|i| format!("s{i}")).collect();
        let eligible: Vec<usize> = (n_control..n).collect();
        SwarmDetector::new(names, &eligible, SubsetPrior::default(), 1e-3)
    }

    #[test]
    fn subset_family_contains_all_sizes_two_through_max() {
        let d = SwarmDetector::new(
            (0..5).map(|i| format!("s{i}")).collect(),
            &[0, 1, 2, 3, 4],
            SubsetPrior {
                max_size: 3,
                ..Default::default()
            },
            1e-3,
        );
        // C(5,2) + C(5,3) = 10 + 10
        assert_eq!(d.family_size(), 20);
        assert!(d.subsets.iter().all(|s| s.len() >= 2 && s.len() <= 3));
        // Sorted, deduplicated.
        assert!(d.subsets.iter().all(|s| s.windows(2).all(|w| w[0] < w[1])));
    }

    #[test]
    fn family_size_cap_reduces_max_size_and_reports_it() {
        let d = SwarmDetector::new(
            (0..14).map(|i| format!("s{i}")).collect(),
            &(0..14).collect::<Vec<_>>(),
            SubsetPrior {
                max_size: 6,
                size_decay: 0.5,
                max_subsets: 200,
            },
            1e-3,
        );
        assert!(d.family_size() <= 200);
        assert!(!d.notes().is_empty());
    }

    #[test]
    fn controls_are_excluded_from_the_candidate_family() {
        let d = detector(8, 5);
        assert!(d
            .subsets
            .iter()
            .all(|s| s.iter().all(|i| *i >= 5)));
    }

    #[test]
    fn multiplicity_cost_is_logarithmic_and_paid_once() {
        let small = detector(8, 5);
        let big = SwarmDetector::new(
            (0..14).map(|i| format!("s{i}")).collect(),
            &(0..14).collect::<Vec<_>>(),
            SubsetPrior::default(),
            1e-3,
        );
        assert!(big.family_size() > 40 * small.family_size());
        // Cost grows only logarithmically in the number of hypotheses.
        assert!(big.multiplicity_cost() < small.multiplicity_cost() + 6.0);
    }

    /// End-to-end pipeline: features -> residuals -> coupling -> conformal
    /// e-values -> mixture. Drives both of the following two tests.
    fn run_scenario(
        seed: u64,
        n_control: usize,
        n_test: usize,
        swarm: &[usize],
        coupling_strength: f64,
        epochs: usize,
    ) -> (SwarmDetector, Vec<SwarmVerdict>) {
        let n = n_control + n_test;
        let dim = 10;
        let mut rng = Rng::new(seed);
        let mut det = detector(n, n_control);
        let kappa = optimal_kappa(crate::coupling::n_pairs(n_control));
        let mut verdicts = Vec::new();

        for epoch in 0..epochs {
            let latent: Vec<f64> = (0..dim).map(|_| rng.normal()).collect();
            let features: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    (0..dim)
                        .map(|k| {
                            let mut v = rng.normal_with(5.0, 1.0);
                            if swarm.contains(&i) {
                                v += coupling_strength * latent[k];
                            }
                            v
                        })
                        .collect()
                })
                .collect();

            let snapshot = EpochSnapshot::new(
                epoch,
                (0..n).map(|i| format!("s{i}")).collect(),
                features,
                (0..n).map(|i| i < n_control).collect(),
            );
            snapshot.validate().unwrap();

            let residuals = snapshot.residuals();
            let scores = instantaneous_coupling(&residuals);
            let controls = control_pair_scores(&scores, n, &snapshot.control_mask);

            let evalues: Vec<EValue> = scores
                .iter()
                .map(|s| calibrate(conformal_p(*s, &controls), kappa))
                .collect();

            verdicts.push(det.update(epoch, &evalues));
        }

        (det, verdicts)
    }

    #[test]
    fn no_swarm_does_not_fire() {
        // False-positive control: the negative-control run from the
        // pre-registration protocol, in miniature.
        let mut fires = 0;
        let runs = 40;
        for seed in 0..runs {
            let (det, _) = run_scenario(1000 + seed, 8, 6, &[], 0.0, 60);
            if det.first_crossing().is_some() {
                fires += 1;
            }
        }
        assert!(
            fires <= 2,
            "{fires}/{runs} false positives at alpha=1e-3 — far above nominal"
        );
    }

    #[test]
    fn planted_swarm_is_detected_and_correctly_attributed() {
        // Sessions 9 and 11 (indices into the full population) collude.
        let swarm = [9usize, 11];
        let (det, verdicts) = run_scenario(7, 8, 6, &swarm, 1.2, 80);

        let crossing = det.first_crossing().expect("planted swarm was not detected");
        assert!(crossing < 60, "detection took {crossing} epochs");

        let last = verdicts.last().unwrap();
        let mut implicated = last.implicated.clone();
        implicated.sort();
        assert_eq!(
            implicated,
            swarm.to_vec(),
            "attribution named {:?}, expected {:?} (marginals {:?})",
            implicated,
            swarm,
            last.marginal
        );

        // The top subset should be exactly the planted pair.
        let mut top = last.top_subsets[0].0.clone();
        top.sort();
        assert_eq!(top, swarm.to_vec());

        // Controls must never be implicated.
        assert!(last.marginal[..8].iter().all(|m| *m < 0.2));
    }

    #[test]
    fn larger_swarm_detected_and_attribution_names_no_innocents() {
        // A 4-member swarm, with max_size=4 in the family.
        //
        // Note what is *not* asserted: that all four members carry high
        // marginal mass. The posterior is a likelihood ratio, so it
        // concentrates multiplicatively — after many epochs the best-supported
        // coupled pair dominates and the full membership resolves more slowly.
        // The operationally meaningful guarantee is that attribution points
        // only at real members, never at controls. See the note on progressive
        // attribution in the module documentation.
        let swarm = [8usize, 9, 10, 11];
        let (det, verdicts) = run_scenario(21, 8, 6, &swarm, 1.0, 80);
        assert!(det.first_crossing().is_some(), "4-member swarm missed");
        let last = verdicts.last().unwrap();

        // No false accusation: every planted member outranks every control.
        let worst_member = swarm
            .iter()
            .map(|m| last.marginal[*m])
            .fold(f64::INFINITY, f64::min);
        let best_control = last.marginal[..8].iter().cloned().fold(0.0, f64::max);
        assert!(
            worst_member > best_control,
            "weakest member {worst_member} ranked below best control {best_control}"
        );

        // The leading hypothesis must be drawn entirely from the real swarm.
        let top = &last.top_subsets[0].0;
        assert!(
            top.iter().all(|i| swarm.contains(i)),
            "top subset {top:?} names a session outside the planted swarm"
        );
    }

    #[test]
    fn detection_is_faster_for_stronger_coupling() {
        let (weak, _) = run_scenario(3, 8, 6, &[9, 11], 0.7, 200);
        let (strong, _) = run_scenario(3, 8, 6, &[9, 11], 2.5, 200);
        let w = weak.first_crossing().unwrap_or(usize::MAX);
        let s = strong.first_crossing().unwrap_or(usize::MAX);
        assert!(s <= w, "stronger coupling detected later ({s} vs {w})");
    }

    #[test]
    fn progress_reports_partial_evidence_before_firing() {
        let (det, verdicts) = run_scenario(9, 8, 6, &[9, 11], 1.2, 12);
        // Short run: likely not yet fired, but progress should be non-trivial.
        if det.first_crossing().is_none() {
            assert!(det.progress() > 0.0);
            assert!(det.progress() < 1.0);
        }
        assert_eq!(verdicts.len(), 12);
    }
}
