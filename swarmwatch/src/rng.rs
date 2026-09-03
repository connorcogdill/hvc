//! Deterministic PRNG (splitmix64) for simulation, canary nonces, and tests.
//!
//! Not cryptographic. Canary nonce *material* is derived through HMAC in
//! [`crate::canary`]; this generator only supplies the seed diversity and the
//! synthetic scenarios in [`crate::sim`]. Every simulation in this crate is
//! reproducible from its seed — a monitor whose false-positive rate cannot be
//! reproduced cannot be audited.

/// splitmix64.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
    spare_normal: Option<f64>,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng {
            state: seed.wrapping_add(0x9E3779B97F4A7C15),
            spare_normal: None,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform on `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        // Top 53 bits — the exactly-representable mantissa width.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform on `[lo, hi)`.
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// Uniform on `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }

    /// Bernoulli draw with probability `p`.
    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }

    /// Standard normal, via Box–Muller with the second variate cached.
    pub fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare_normal.take() {
            return z;
        }
        // Guard against u1 == 0, which would give ln(0).
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        self.spare_normal = Some(r * theta.sin());
        r * theta.cos()
    }

    /// Normal with the given mean and standard deviation.
    pub fn normal_with(&mut self, mean: f64, sd: f64) -> f64 {
        mean + sd * self.normal()
    }

    /// Fisher–Yates shuffle.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reproducible_from_seed() {
        let a: Vec<u64> = (0..8).scan(Rng::new(7), |r, _| Some(r.next_u64())).collect();
        let b: Vec<u64> = (0..8).scan(Rng::new(7), |r, _| Some(r.next_u64())).collect();
        assert_eq!(a, b);
        let c: Vec<u64> = (0..8).scan(Rng::new(8), |r, _| Some(r.next_u64())).collect();
        assert_ne!(a, c);
    }

    #[test]
    fn uniform_in_range() {
        let mut r = Rng::new(1);
        for _ in 0..10_000 {
            let x = r.next_f64();
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn normal_moments_are_close() {
        let mut r = Rng::new(42);
        let n = 200_000;
        let mut sum = 0.0;
        let mut sq = 0.0;
        for _ in 0..n {
            let z = r.normal();
            sum += z;
            sq += z * z;
        }
        let mean = sum / n as f64;
        let var = sq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "var {var}");
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut r = Rng::new(3);
        let mut v: Vec<usize> = (0..64).collect();
        r.shuffle(&mut v);
        let mut sorted = v.clone();
        sorted.sort();
        assert_eq!(sorted, (0..64).collect::<Vec<_>>());
        assert_ne!(v, sorted);
    }
}
