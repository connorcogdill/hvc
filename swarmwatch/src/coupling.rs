//! Pairwise and directed coupling between sessions.
//!
//! # Why coupling, and not per-session anomaly
//!
//! The sample-complexity pessimism bites hardest on single-surface, single-agent
//! anomaly detection: a patient adversary dilutes its per-observation signal
//! until the detector starves. Each swarm member individually can be made
//! arbitrarily unremarkable.
//!
//! But a swarm is not a set of agents. A swarm is a set of agents **plus a
//! relation between them**, and the relation is the thing that cannot be
//! diluted. Coordination is not a message that can be sent slowly; it is a
//! constraint that members' behaviour must jointly satisfy. Two agents that
//! coordinate must, at some point, be more predictable from each other than two
//! agents that do not — otherwise nothing has been coordinated.
//!
//! So the statistic is relational, and it lives on the *residuals* from
//! [`crate::population`], which have already had everything the whole population
//! shares subtracted out. What remains is the part of each session's behaviour
//! that is specific to it, and coupling asks whether those specific parts move
//! together.
//!
//! # Instantaneous versus directed
//!
//! Instantaneous coupling (same-epoch residual alignment) detects *that* two
//! sessions are related. It cannot distinguish coordination from common cause —
//! two agents handed similar subtasks will align without colluding.
//!
//! Directed coupling exploits the arrow of time: if A's residual at `t-1`
//! predicts B's at `t` better than the reverse, information flowed A→B.
//! A common cause acting on both sessions is symmetric and cancels in the
//! difference. This asymmetry is what turns detection into **attribution** —
//! naming the donor rather than just the cluster — and time-to-attribute is
//! the metric the July 2026 incident showed to be the longer pole.
//!
//! Neither statistic requires reading content. Both operate on numeric feature
//! residuals, so they remain available when transcripts are unavailable,
//! untrusted (Tier 2), or encrypted.

/// Number of unordered pairs among `n` items.
pub fn n_pairs(n: usize) -> usize {
    n * n.saturating_sub(1) / 2
}

/// Flat index of the unordered pair `(i, j)`, `i != j`, in row-major upper-
/// triangular order.
pub fn pair_index(n: usize, i: usize, j: usize) -> usize {
    let (a, b) = if i < j { (i, j) } else { (j, i) };
    // Pairs before row a, plus offset within row a.
    a * n - a * (a + 1) / 2 + (b - a - 1)
}

/// Inverse of [`pair_index`].
pub fn pair_from_index(n: usize, idx: usize) -> (usize, usize) {
    let mut offset = 0;
    for a in 0..n {
        let row = n - a - 1;
        if idx < offset + row {
            return (a, a + 1 + (idx - offset));
        }
        offset += row;
    }
    (0, 0)
}

/// Cosine similarity, in `[-1, 1]`. Zero vectors give 0 — no direction, hence
/// no alignment, rather than a division by zero.
pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for k in 0..n {
        dot += a[k] * b[k];
        na += a[k] * a[k];
        nb += b[k] * b[k];
    }
    if na <= 1e-18 || nb <= 1e-18 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())).clamp(-1.0, 1.0)
}

/// Instantaneous coupling for every pair, indexed by [`pair_index`].
///
/// Magnitude rather than signed similarity: two sessions moving in *opposite*
/// directions in residual space are as coupled as two moving together. A swarm
/// dividing labour — one agent taking the work another declines — produces
/// anti-correlation, and treating that as innocent would be a straightforward
/// gap for an adversary to walk through.
pub fn instantaneous_coupling(residuals: &[Vec<f64>]) -> Vec<f64> {
    let n = residuals.len();
    let mut out = vec![0.0; n_pairs(n)];
    for i in 0..n {
        for j in (i + 1)..n {
            out[pair_index(n, i, j)] = cosine(&residuals[i], &residuals[j]).abs();
        }
    }
    out
}

/// Signed directed coupling for every pair, indexed by [`pair_index`].
///
/// `d[(i,j)] = cos(prev[i], cur[j]) - cos(prev[j], cur[i])`
///
/// Positive means information flowed `i → j` (i is the donor / parent);
/// negative means `j → i`; near zero means symmetric, i.e. common cause or no
/// relation. Returns an empty vector when `prev` and `cur` disagree in shape.
pub fn directed_coupling(prev: &[Vec<f64>], cur: &[Vec<f64>]) -> Vec<f64> {
    let n = prev.len();
    if n != cur.len() {
        return Vec::new();
    }
    let mut out = vec![0.0; n_pairs(n)];
    for i in 0..n {
        for j in (i + 1)..n {
            let fwd = cosine(&prev[i], &cur[j]);
            let rev = cosine(&prev[j], &cur[i]);
            out[pair_index(n, i, j)] = fwd - rev;
        }
    }
    out
}

/// Magnitude of directed coupling — "there is a direction here", regardless of
/// which way. Used as the *detection* statistic; the sign is used for
/// attribution.
pub fn directed_strength(prev: &[Vec<f64>], cur: &[Vec<f64>]) -> Vec<f64> {
    directed_coupling(prev, cur)
        .into_iter()
        .map(f64::abs)
        .collect()
}

/// The coupling scores among declared control sessions only.
///
/// This is the calibration set: pairs both of whose members are uncontaminable
/// reference agents, and which are therefore guaranteed swarm-free. Under the
/// null every pair is exchangeable with these, which is what licenses the
/// conformal p-value.
pub fn control_pair_scores(scores: &[f64], n: usize, control_mask: &[bool]) -> Vec<f64> {
    let mut out = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if *control_mask.get(i).unwrap_or(&false) && *control_mask.get(j).unwrap_or(&false) {
                out.push(scores[pair_index(n, i, j)]);
            }
        }
    }
    out
}

/// A directed edge implicated in coordination, ordered strongest first.
#[derive(Clone, Debug, PartialEq)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub strength: f64,
}

/// Rank directed edges by strength, orienting each by the sign of the directed
/// coupling. Feeds attribution: the source of the strongest edge is the donor
/// candidate for a replication event.
pub fn rank_edges(directed: &[f64], n: usize, min_strength: f64) -> Vec<Edge> {
    let mut edges: Vec<Edge> = (0..directed.len())
        .filter(|k| directed[*k].abs() >= min_strength)
        .map(|k| {
            let (i, j) = pair_from_index(n, k);
            if directed[k] >= 0.0 {
                Edge {
                    from: i,
                    to: j,
                    strength: directed[k],
                }
            } else {
                Edge {
                    from: j,
                    to: i,
                    strength: -directed[k],
                }
            }
        })
        .collect();
    edges.sort_by(|a, b| {
        b.strength
            .partial_cmp(&a.strength)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    edges
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn pair_index_is_a_bijection() {
        for n in 2..12 {
            let mut seen = vec![false; n_pairs(n)];
            for i in 0..n {
                for j in (i + 1)..n {
                    let k = pair_index(n, i, j);
                    assert!(k < n_pairs(n), "index {k} out of range for n={n}");
                    assert!(!seen[k], "collision at n={n} pair ({i},{j})");
                    seen[k] = true;
                    assert_eq!(pair_from_index(n, k), (i, j));
                    // Order-independent.
                    assert_eq!(pair_index(n, j, i), k);
                }
            }
            assert!(seen.iter().all(|s| *s));
        }
    }

    #[test]
    fn cosine_basics() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-12);
        assert!((cosine(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-12);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-12);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn independent_sessions_are_weakly_coupled() {
        let mut rng = Rng::new(1);
        let residuals: Vec<Vec<f64>> = (0..8)
            .map(|_| (0..12).map(|_| rng.normal()).collect())
            .collect();
        let c = instantaneous_coupling(&residuals);
        let mean: f64 = c.iter().sum::<f64>() / c.len() as f64;
        // Random 12-dim unit vectors align at ~sqrt(2/(pi*d)) in expectation.
        assert!(mean < 0.35, "independent sessions coupled at {mean}");
    }

    #[test]
    fn coordinated_sessions_are_strongly_coupled() {
        let mut rng = Rng::new(2);
        let shared: Vec<f64> = (0..12).map(|_| rng.normal()).collect();
        let mut residuals: Vec<Vec<f64>> = (0..8)
            .map(|_| (0..12).map(|_| rng.normal()).collect())
            .collect();
        // Sessions 0 and 1 carry a common latent component.
        for row in residuals.iter_mut().take(2) {
            for (k, v) in row.iter_mut().enumerate() {
                *v += 2.0 * shared[k];
            }
        }
        let c = instantaneous_coupling(&residuals);
        let coupled = c[pair_index(8, 0, 1)];
        let others: f64 = (2..8)
            .flat_map(|i| ((i + 1)..8).map(move |j| (i, j)))
            .map(|(i, j)| c[pair_index(8, i, j)])
            .sum::<f64>()
            / n_pairs(6) as f64;
        assert!(coupled > 3.0 * others, "coupled {coupled} vs others {others}");
    }

    #[test]
    fn anticorrelated_division_of_labour_is_still_coupling() {
        let mut rng = Rng::new(3);
        let shared: Vec<f64> = (0..12).map(|_| rng.normal()).collect();
        let a: Vec<f64> = shared.iter().map(|v| 3.0 * v).collect();
        let b: Vec<f64> = shared.iter().map(|v| -3.0 * v).collect();
        let residuals = vec![a, b];
        let c = instantaneous_coupling(&residuals);
        assert!(
            c[0] > 0.9,
            "opposed-but-locked sessions must register as coupled, got {}",
            c[0]
        );
    }

    #[test]
    fn directed_coupling_recovers_the_donor() {
        // An orthogonal basis, so the only non-zero alignment anywhere is the
        // planted one. Hand-picked vectors are a trap here: an accidental
        // *anti*-alignment on some other pair inflates that pair's asymmetry
        // just as much as a real copy would, and can outrank the true edge.
        let prev = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        let cur = vec![
            vec![0.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 0.0], // copied from prev[0]: flow 0 -> 1
            vec![0.0, 0.0, 0.0, 1.0],
        ];

        let d = directed_coupling(&prev, &cur);
        assert!((d[pair_index(3, 0, 1)] - 1.0).abs() < 1e-12);
        assert!(d[pair_index(3, 0, 2)].abs() < 1e-12);
        assert!(d[pair_index(3, 1, 2)].abs() < 1e-12);

        let edges = rank_edges(&d, 3, 0.3);
        assert_eq!(edges.len(), 1);
        assert_eq!((edges[0].from, edges[0].to), (0, 1));
    }

    #[test]
    fn common_cause_produces_no_direction() {
        // Both sessions driven by the same external signal, symmetrically.
        let mut rng = Rng::new(4);
        let drive: Vec<f64> = (0..10).map(|_| rng.normal()).collect();
        let prev = vec![drive.clone(), drive.clone()];
        let cur = vec![drive.clone(), drive.clone()];
        let d = directed_coupling(&prev, &cur);
        assert!(
            d[0].abs() < 1e-9,
            "symmetric common cause leaked a direction: {}",
            d[0]
        );
    }

    #[test]
    fn directed_coupling_rejects_shape_mismatch() {
        let prev = vec![vec![1.0], vec![2.0]];
        let cur = vec![vec![1.0]];
        assert!(directed_coupling(&prev, &cur).is_empty());
    }

    #[test]
    fn control_pair_scores_selects_only_control_control_pairs() {
        let n = 4;
        let mask = vec![true, true, false, false];
        let scores: Vec<f64> = (0..n_pairs(n)).map(|k| k as f64).collect();
        let got = control_pair_scores(&scores, n, &mask);
        assert_eq!(got, vec![scores[pair_index(n, 0, 1)]]);
    }

    #[test]
    fn rank_edges_orders_by_strength_and_filters() {
        let n = 3;
        let mut d = vec![0.0; n_pairs(n)];
        d[pair_index(n, 0, 1)] = 0.9;
        d[pair_index(n, 0, 2)] = -0.5;
        d[pair_index(n, 1, 2)] = 0.05;
        let e = rank_edges(&d, n, 0.1);
        assert_eq!(e.len(), 2);
        assert!((e[0].strength - 0.9).abs() < 1e-12);
        assert_eq!((e[1].from, e[1].to), (2, 0)); // sign flipped
    }
}
