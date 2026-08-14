//! Calibration probe — measure the detector's null drift before trusting it.
//!
//! The pre-registration protocol says to run negative controls, because a
//! nominal false-positive rate is a claim and a measured one is evidence. This
//! binary is the sharper version of that check.
//!
//! A clean negative-control run proves less than it appears to. What matters is
//! **null drift**: the mean nats of evidence the detector accumulates per epoch
//! when nothing is coordinating. A valid detector drifts `≤ 0`. Any positive
//! value fires with probability one given enough epochs, at any `α` — so a short
//! run can look spotless while the detector is guaranteed to alarm on a clean
//! population eventually. Drift is the quantity to watch; a false-positive count
//! is a lagging indicator of it.
//!
//! The stressor here is **profile mismatch**: sessions under test whose typical
//! behaviour differs from the reference agents', with no coordination
//! whatsoever. That is the normal deployment condition rather than an edge case —
//! reference agents run pre-registered benign tasks while the sessions under test
//! do the actual evaluation — and it is what broke an earlier version of this
//! detector, which fired on 82% of clean runs at a one-sigma offset while every
//! unit test passed.
//!
//! Run this whenever the feature set, the control population, or the
//! standardisation changes. It exits non-zero on a calibration failure, so it
//! can gate a deployment.
//!
//! ```sh
//! cargo run --release --bin swarmwatch-probe
//! ```

use swarmwatch::conformal::optimal_kappa;
use swarmwatch::coupling::{instantaneous_coupling, n_pairs};
use swarmwatch::population::{EpochSnapshot, PopulationTracker};
use swarmwatch::rng::Rng;
use swarmwatch::swarm::{MultiScaleDetector, ScaleLadder, SubsetPrior};

const N_CONTROL: usize = 8;
const N_TEST: usize = 4;
const DIM: usize = 8;
const WARMUP: usize = 20;
const ALPHA: f64 = 1e-3;

struct Outcome {
    fired_at: Option<usize>,
    /// Mean nats of evidence gained per emitting epoch.
    drift: f64,
}

/// One run through the shipped pipeline.
///
/// `profile_shift` scales a fixed offset applied to the sessions under test —
/// "these agents are doing a different job", not "these agents are colluding".
/// `coupling` adds a genuine per-epoch shared latent to the first `n_members`
/// test sessions, which is coordination.
fn run(
    seed: u64,
    epochs: usize,
    profile_shift: f64,
    coupling: f64,
    n_members: usize,
    ladder: ScaleLadder,
) -> Outcome {
    let n = N_CONTROL + N_TEST;
    let mut rng = Rng::new(seed ^ 0xBEEF);
    let names: Vec<String> = (0..n).map(|i| format!("s{i}")).collect();
    let eligible: Vec<usize> = (N_CONTROL..n).collect();

    let mut det = MultiScaleDetector::new(
        names.clone(),
        &eligible,
        (0..n).map(|i| i < N_CONTROL).collect(),
        ladder,
        SubsetPrior::default(),
        ALPHA,
        optimal_kappa(n_pairs(N_CONTROL)),
    );
    let mut tracker = PopulationTracker::new(WARMUP);

    let offset: Vec<f64> = (0..DIM).map(|_| rng.normal()).collect();
    let members: Vec<usize> = (N_CONTROL..N_CONTROL + n_members).collect();

    let mut emitting = 0usize;
    let mut last_log_e = 0.0;
    let mut fired_at = None;

    for epoch in 0..epochs {
        let latent: Vec<f64> = (0..DIM).map(|_| rng.normal()).collect();
        let features: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let is_test = i >= N_CONTROL;
                let is_member = members.contains(&i);
                (0..DIM)
                    .map(|k| {
                        5.0 + rng.normal()
                            + if is_test { profile_shift * offset[k] } else { 0.0 }
                            + if is_member { coupling * latent[k] } else { 0.0 }
                    })
                    .collect()
            })
            .collect();

        let snapshot = EpochSnapshot::new(
            epoch,
            names.clone(),
            features,
            (0..n).map(|i| i < N_CONTROL).collect(),
        );

        let Some(residuals) = tracker.residuals(&snapshot) else {
            continue; // warm-up
        };

        let Some(verdict) = det.update(epoch, &instantaneous_coupling(&residuals)) else {
            continue; // no scale closed a block this epoch
        };
        last_log_e = verdict.log_e;
        emitting += 1;

        if verdict.fired_now && fired_at.is_none() {
            fired_at = Some(epoch);
        }
    }

    Outcome {
        fired_at,
        drift: if emitting > 0 {
            last_log_e / emitting as f64
        } else {
            0.0
        },
    }
}

fn main() {
    let seeds = 40u64;
    let mut worst_drift = f64::NEG_INFINITY;
    let mut failures = Vec::new();

    println!("{}", "=".repeat(78));
    println!("CALIBRATION PROBE — null drift under profile mismatch");
    println!("{}", "=".repeat(78));
    println!(
        "{N_CONTROL} reference agents, {N_TEST} sessions under test, alpha = {ALPHA:.0e}, \
         warm-up {WARMUP} epochs"
    );
    println!(
        "scale ladder {:?}, {seeds} runs per row.\n",
        ScaleLadder::default().blocks
    );

    println!("NULL — nothing is coordinating. Drift must be <= 0.");
    println!(
        "  {:>10}  {:>16}  {:>14}  {:>10}",
        "mismatch", "drift (nats/ep)", "fired", "verdict"
    );
    println!("  {}", "-".repeat(58));

    for shift in [0.0f64, 0.5, 1.0, 2.0, 4.0, 8.0] {
        let mut drift = 0.0;
        let mut fired = 0;
        for seed in 0..seeds {
            let o = run(seed, 900, shift, 0.0, 0, ScaleLadder::default());
            drift += o.drift;
            if o.fired_at.is_some() {
                fired += 1;
            }
        }
        drift /= seeds as f64;
        worst_drift = worst_drift.max(drift);

        let ok = drift <= 0.0 && fired == 0;
        if !ok {
            failures.push(format!(
                "drift {drift:+.4} nats/epoch, {fired} false alarm(s) at \
                 {shift:.1}-sigma profile mismatch"
            ));
        }
        println!(
            "  {:>9.1}s  {:>+16.5}  {:>7} ({:>3.0}%)  {:>10}",
            shift,
            drift,
            fired,
            100.0 * fired as f64 / seeds as f64,
            if ok { "ok" } else { "DRIFTING" }
        );
    }

    println!("\nALTERNATIVE — a genuine coordinating subset, 3 members, 1200 epoch horizon.");
    println!("Per-epoch calibration alone against the full ladder, median epoch / rate.");
    println!(
        "  {:>10}  {:>10}  {:>18}  {:>18}",
        "mismatch", "coupling", "per-epoch only", "ladder [1,5,25]"
    );
    println!("  {}", "-".repeat(62));

    let fmt = |mut h: Vec<usize>| {
        h.sort();
        if h.is_empty() {
            "never".to_string()
        } else {
            format!("{} / {:.0}%", h[h.len() / 2], 100.0 * h.len() as f64 / seeds as f64)
        }
    };

    for (shift, coupling) in [
        (0.0, 1.4),
        (8.0, 1.4),
        (0.0, 0.8),
        (0.0, 0.6),
        (0.0, 0.4),
    ] {
        let fast: Vec<usize> = (0..seeds)
            .filter_map(|s| run(s, 1200, shift, coupling, 3, ScaleLadder::new([1])).fired_at)
            .collect();
        let full: Vec<usize> = (0..seeds)
            .filter_map(|s| run(s, 1200, shift, coupling, 3, ScaleLadder::default()).fired_at)
            .collect();
        println!(
            "  {:>9.1}s  {:>10.1}  {:>18}  {:>18}",
            shift,
            coupling,
            fmt(fast),
            fmt(full)
        );
    }

    println!("\n{}", "=".repeat(78));
    if failures.is_empty() {
        println!("VERDICT: calibration holds.");
        println!("  Worst null drift {worst_drift:+.5} nats/epoch, across an 8-sigma range of");
        println!("  profile mismatch. Validity does not degrade with mismatch because each");
        println!("  session is standardised against its own history and no session takes part");
        println!("  in another's baseline — see population::PopulationTracker.");
        println!("\n  The ladder mixes several block lengths concurrently, so no single time");
        println!("  scale has to be guessed. It roughly halves the weakest detectable coupling");
        println!("  while costing only ln(#scales) nats on signals the fast scale already sees.");
        println!("\n  Rows still reading 'never' are the honest exposure statement, not a");
        println!("  defect: coupling below the provisioned floor is out of reach by design.");
        println!("  State it in advance with planning::weakest_detectable_signal.");
    } else {
        println!("VERDICT: CALIBRATION FAILURE — do not deploy this configuration.");
        for f in &failures {
            println!("  - {f}");
        }
        println!("\n  Positive null drift means the detector alarms on a clean population given");
        println!("  enough epochs, at any alpha. A short negative-control run will not show it.");
        std::process::exit(1);
    }
    println!("{}", "=".repeat(78));
}
