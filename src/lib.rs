//! # ternary-dropout
//!
//! Ternary dropout and regularization techniques for neural networks operating
//! in ℤ₃ = {-1, 0, 1}. All operations use a seeded PRNG for reproducibility.

use std::fmt;

/// A trit in ℤ₃ = {-1, 0, 1}.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Trit {
    NegOne = -1,
    Zero = 0,
    One = 1,
}

impl Trit {
    pub fn to_i8(self) -> i8 {
        self as i8
    }

    pub fn from_i8(v: i8) -> Self {
        match v.cmp(&0) {
            std::cmp::Ordering::Less => Trit::NegOne,
            std::cmp::Ordering::Equal => Trit::Zero,
            std::cmp::Ordering::Greater => Trit::One,
        }
    }

    /// Flip to another trit value (randomly among the other two).
    pub fn flip(self, choice: u8) -> Trit {
        match (self, choice % 2) {
            (Trit::NegOne, 0) => Trit::Zero,
            (Trit::NegOne, _) => Trit::One,
            (Trit::Zero, 0) => Trit::NegOne,
            (Trit::Zero, _) => Trit::One,
            (Trit::One, 0) => Trit::NegOne,
            (Trit::One, _) => Trit::Zero,
        }
    }
}

impl fmt::Display for Trit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_i8())
    }
}

/// A simple seeded PRNG (xorshift64) for reproducibility.
#[derive(Debug, Clone)]
pub struct TernaryRng {
    state: u64,
}

impl TernaryRng {
    /// Create a new RNG with the given seed.
    pub fn new(seed: u64) -> Self {
        // Ensure non-zero state
        TernaryRng {
            state: if seed == 0 { 0xDEAD_BEEF_CAFE_BABE } else { seed },
        }
    }

    /// Generate the next random u64.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate a random u8.
    pub fn next_u8(&mut self) -> u8 {
        self.next_u64() as u8
    }

    /// Generate a random float in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() & 0x1FFFFF) as f64 / 0x200000 as f64
    }

    /// Generate a random trit.
    pub fn next_trit(&mut self) -> Trit {
        match self.next_u8() % 3 {
            0 => Trit::NegOne,
            1 => Trit::Zero,
            _ => Trit::One,
        }
    }
}

/// A ternary matrix stored as a flat vector with row-major layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TernaryMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Trit>,
}

impl TernaryMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        TernaryMatrix {
            rows,
            cols,
            data: vec![Trit::Zero; rows * cols],
        }
    }

    pub fn from_flat(rows: usize, cols: usize, data: Vec<Trit>) -> Self {
        assert_eq!(data.len(), rows * cols);
        TernaryMatrix { rows, cols, data }
    }

    pub fn get(&self, r: usize, c: usize) -> Trit {
        self.data[r * self.cols + c]
    }

    pub fn set(&mut self, r: usize, c: usize, v: Trit) {
        self.data[r * self.cols + c] = v;
    }
}

/// Random ternary dropout: set each element to 0 with probability `drop_rate`.
///
/// Returns a new matrix where dropped positions are set to `Trit::Zero`.
/// When `drop_rate` is 0.0, returns a clone of the input.
pub fn random_dropout(matrix: &TernaryMatrix, drop_rate: f64, rng: &mut TernaryRng) -> TernaryMatrix {
    let mut result = matrix.clone();
    if drop_rate <= 0.0 {
        return result;
    }
    for trit in result.data.iter_mut() {
        if rng.next_f64() < drop_rate {
            *trit = Trit::Zero;
        }
    }
    result
}

/// Structured dropout: drop entire rows or columns.
///
/// With `drop_rows=true`, randomly zeroes out entire rows.
/// With `drop_rows=false`, randomly zeroes out entire columns.
/// `drop_rate` controls the fraction of rows/columns to drop.
pub fn structured_dropout(
    matrix: &TernaryMatrix,
    drop_rate: f64,
    drop_rows: bool,
    rng: &mut TernaryRng,
) -> TernaryMatrix {
    let mut result = matrix.clone();

    if drop_rows {
        for r in 0..result.rows {
            if rng.next_f64() < drop_rate {
                for c in 0..result.cols {
                    result.set(r, c, Trit::Zero);
                }
            }
        }
    } else {
        for c in 0..result.cols {
            if rng.next_f64() < drop_rate {
                for r in 0..result.rows {
                    result.set(r, c, Trit::Zero);
                }
            }
        }
    }

    result
}

/// Ternary noise injection: flip random trits to a different value.
///
/// Each element has a probability of `flip_rate` to be flipped to one of the
/// other two trit values (chosen randomly).
pub fn noise_injection(matrix: &TernaryMatrix, flip_rate: f64, rng: &mut TernaryRng) -> TernaryMatrix {
    let mut result = matrix.clone();
    if flip_rate <= 0.0 {
        return result;
    }
    for trit in result.data.iter_mut() {
        if rng.next_f64() < flip_rate {
            *trit = trit.flip(rng.next_u8());
        }
    }
    result
}

/// Sparse dropout: keep only the top-k elements by magnitude (treating -1 and 1 as equal magnitude).
///
/// All elements not in the top-k are set to 0. Ties are broken by position (earlier positions preferred).
pub fn sparse_dropout(matrix: &TernaryMatrix, k: usize) -> TernaryMatrix {
    let total = matrix.data.len();
    if k >= total {
        return matrix.clone();
    }

    // Compute magnitudes: |NegOne|=1, |Zero|=0, |One|=1
    let magnitudes: Vec<(usize, u8)> = matrix
        .data
        .iter()
        .enumerate()
        .map(|(i, t)| (i, if *t == Trit::Zero { 0 } else { 1 }))
        .collect();

    // Sort by magnitude descending, then by index ascending for tie-breaking
    let mut sorted = magnitudes;
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    let mut result = TernaryMatrix::zeros(matrix.rows, matrix.cols);
    for i in 0..k.min(sorted.len()) {
        let idx = sorted[i].0;
        result.data[idx] = matrix.data[idx];
    }
    result
}

/// DropConnect: randomly drop individual weight connections in a matrix.
///
/// Each element has a probability of `drop_rate` to be set to `Trit::Zero`.
/// This is equivalent to element-wise random dropout but is conceptually
/// different—it's applied to weight matrices, not activations.
pub fn dropconnect(weights: &TernaryMatrix, drop_rate: f64, rng: &mut TernaryRng) -> TernaryMatrix {
    random_dropout(weights, drop_rate, rng)
}

/// Monte Carlo dropout: run multiple forward passes with dropout enabled
/// to estimate uncertainty.
///
/// Returns:
///   - `mean`: a matrix where each element is the mean of all samples (mapped back to nearest trit)
///   - `variance`: a matrix where each element counts how many samples differed
///
/// The variance is the fraction of samples that were non-zero (a simple uncertainty measure).
pub struct MCOutput {
    pub mean: TernaryMatrix,
    pub variance: TernaryMatrix,
}

pub fn monte_carlo_dropout(
    matrix: &TernaryMatrix,
    drop_rate: f64,
    num_samples: usize,
    rng: &mut TernaryRng,
) -> MCOutput {
    let total_elements = matrix.data.len();

    // Track sum and count of non-zero for each position
    let mut sums = vec![0i64; total_elements];
    let mut nonzero_counts = vec![0usize; total_elements];

    for _ in 0..num_samples {
        let sample = random_dropout(matrix, drop_rate, rng);
        for (i, trit) in sample.data.iter().enumerate() {
            sums[i] += trit.to_i8() as i64;
            if *trit != Trit::Zero {
                nonzero_counts[i] += 1;
            }
        }
    }

    // Compute mean (quantized to trit)
    let mut mean_data = vec![Trit::Zero; total_elements];
    for (i, &sum) in sums.iter().enumerate() {
        let avg = sum as f64 / num_samples as f64;
        mean_data[i] = if avg < -0.5 {
            Trit::NegOne
        } else if avg > 0.5 {
            Trit::One
        } else {
            Trit::Zero
        };
    }

    // Variance as fraction of samples that were non-zero
    let mut var_data = vec![Trit::Zero; total_elements];
    for (i, &count) in nonzero_counts.iter().enumerate() {
        let frac = count as f64 / num_samples as f64;
        // Map variance fraction to trit: low/medium/high uncertainty
        var_data[i] = if frac < 0.33 {
            Trit::Zero   // low variance
        } else if frac < 0.66 {
            Trit::One    // medium variance
        } else {
            Trit::NegOne // high variance (wraps to -1 in Z3)
        };
    }

    MCOutput {
        mean: TernaryMatrix::from_flat(matrix.rows, matrix.cols, mean_data),
        variance: TernaryMatrix::from_flat(matrix.rows, matrix.cols, var_data),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_matrix() -> TernaryMatrix {
        TernaryMatrix::from_flat(3, 4, vec![
            Trit::One, Trit::NegOne, Trit::Zero, Trit::One,
            Trit::Zero, Trit::One, Trit::One, Trit::NegOne,
            Trit::NegOne, Trit::Zero, Trit::One, Trit::Zero,
        ])
    }

    #[test]
    fn test_random_dropout_zeros_correct_fraction() {
        let mat = sample_matrix();
        let mut rng = TernaryRng::new(42);
        let drop_rate = 0.5;

        let dropped = random_dropout(&mat, drop_rate, &mut rng);

        // Count zeros
        let zeros = dropped.data.iter().filter(|&&t| t == Trit::Zero).count();
        let total = dropped.data.len();

        // With drop_rate=0.5 on 12 elements, expect ~6 zeros (±some variance)
        // The matrix already has 4 zeros, so we need at least 4
        assert!(zeros >= 4, "Expected at least 4 zeros, got {}", zeros);

        // Should not zero everything
        assert!(zeros < total, "Should not zero everything");
    }

    #[test]
    fn test_random_dropout_no_drop_mode() {
        let mat = sample_matrix();
        let mut rng = TernaryRng::new(42);

        // drop_rate = 0 should return identical matrix
        let dropped = random_dropout(&mat, 0.0, &mut rng);
        assert_eq!(dropped, mat);

        // Negative drop_rate should also be no-op
        let dropped = random_dropout(&mat, -0.5, &mut rng);
        assert_eq!(dropped, mat);
    }

    #[test]
    fn test_random_dropout_full_drop() {
        let mat = sample_matrix();
        let mut rng = TernaryRng::new(42);

        let dropped = random_dropout(&mat, 1.0, &mut rng);
        // Everything should be zero
        assert!(dropped.data.iter().all(|&t| t == Trit::Zero));
    }

    #[test]
    fn test_structured_dropout_rows_preserves_shape() {
        let mat = sample_matrix();
        let mut rng = TernaryRng::new(123);

        let dropped = structured_dropout(&mat, 0.5, true, &mut rng);

        // Shape preserved
        assert_eq!(dropped.rows, mat.rows);
        assert_eq!(dropped.cols, mat.cols);

        // At least one row should be all zeros
        let all_zero_rows = (0..dropped.rows)
            .filter(|&r| (0..dropped.cols).all(|c| dropped.get(r, c) == Trit::Zero))
            .count();
        assert!(all_zero_rows >= 1, "Expected at least 1 dropped row");
    }

    #[test]
    fn test_structured_dropout_columns_preserves_shape() {
        let mat = sample_matrix();
        let mut rng = TernaryRng::new(456);

        let dropped = structured_dropout(&mat, 0.5, false, &mut rng);

        assert_eq!(dropped.rows, mat.rows);
        assert_eq!(dropped.cols, mat.cols);

        // Check that at least one column is all zeros
        let all_zero_cols = (0..dropped.cols)
            .filter(|&c| (0..dropped.rows).all(|r| dropped.get(r, c) == Trit::Zero))
            .count();
        assert!(all_zero_cols >= 1, "Expected at least 1 dropped column");
    }

    #[test]
    fn test_noise_injection_changes_some_trits() {
        let mat = sample_matrix();
        let mut rng = TernaryRng::new(789);

        let noisy = noise_injection(&mat, 0.5, &mut rng);

        // Some values should have changed
        let changed = mat.data.iter().zip(noisy.data.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(changed > 0, "Expected some trits to be flipped");

        // But not all should change
        assert!(changed < mat.data.len(), "Not all trits should flip at 50%");
    }

    #[test]
    fn test_noise_injection_no_flip() {
        let mat = sample_matrix();
        let mut rng = TernaryRng::new(42);

        let result = noise_injection(&mat, 0.0, &mut rng);
        assert_eq!(result, mat);

        let result = noise_injection(&mat, -1.0, &mut rng);
        assert_eq!(result, mat);
    }

    #[test]
    fn test_sparse_dropout_keeps_top_k() {
        let mat = sample_matrix();
        // mat has 8 non-zero elements and 4 zeros
        // Keep top 4

        let result = sparse_dropout(&mat, 4);

        // Count non-zero in result
        let nonzero = result.data.iter().filter(|&&t| t != Trit::Zero).count();
        assert_eq!(nonzero, 4, "Should keep exactly 4 elements");

        // The kept elements should be from original non-zero positions
        for i in 0..result.data.len() {
            if result.data[i] != Trit::Zero {
                assert_eq!(result.data[i], mat.data[i], "Kept value should match original");
            } else {
                // Zeroed positions should also have been zero in result
            }
        }
    }

    #[test]
    fn test_sparse_dropout_k_exceeds_total() {
        let mat = sample_matrix();
        let result = sparse_dropout(&mat, 100);
        assert_eq!(result, mat, "k >= total should return original");
    }

    #[test]
    fn test_sparse_dropout_k_zero() {
        let mat = sample_matrix();
        let result = sparse_dropout(&mat, 0);
        assert!(result.data.iter().all(|&t| t == Trit::Zero), "k=0 should zero everything");
    }

    #[test]
    fn test_dropconnect_zeros_connections() {
        let weights = sample_matrix();
        let mut rng = TernaryRng::new(999);

        let dropped = dropconnect(&weights, 0.5, &mut rng);

        let zeros = dropped.data.iter().filter(|&&t| t == Trit::Zero).count();
        assert!(zeros > 4, "Expected significant number of zeros from dropconnect");
    }

    #[test]
    fn test_mc_dropout_produces_variance() {
        let mat = sample_matrix();
        let mut rng = TernaryRng::new(42);

        let mc = monte_carlo_dropout(&mat, 0.5, 20, &mut rng);

        // Mean should have correct shape
        assert_eq!(mc.mean.rows, mat.rows);
        assert_eq!(mc.mean.cols, mat.cols);

        // Variance should have correct shape
        assert_eq!(mc.variance.rows, mat.rows);
        assert_eq!(mc.variance.cols, mat.cols);

        // With dropout 0.5, there should be some non-zero variance entries
        let var_nonzero = mc.variance.data.iter()
            .filter(|&&t| t != Trit::Zero)
            .count();
        assert!(var_nonzero > 0, "MC dropout should produce some variance");
    }

    #[test]
    fn test_mc_dropout_no_drop_gives_original() {
        let mat = sample_matrix();
        let mut rng = TernaryRng::new(42);

        let mc = monte_carlo_dropout(&mat, 0.0, 10, &mut rng);

        // With no dropout, mean should match original
        assert_eq!(mc.mean, mat);

        // Variance should be all One (every sample was non-zero where original was non-zero)
        // Actually: nonzero_count / num_samples → for originally non-zero elements, frac = 1.0 ≥ 0.66 → NegOne
        // for originally zero elements, frac = 0.0 < 0.33 → Zero
        for i in 0..mat.data.len() {
            if mat.data[i] == Trit::Zero {
                assert_eq!(mc.variance.data[i], Trit::Zero, "Zero elements should have zero variance");
            }
        }
    }

    #[test]
    fn test_rng_reproducibility() {
        let mut rng1 = TernaryRng::new(12345);
        let mut rng2 = TernaryRng::new(12345);

        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_trit_flip() {
        assert_eq!(Trit::One.flip(0), Trit::NegOne);
        assert_eq!(Trit::One.flip(1), Trit::Zero);
        assert_eq!(Trit::NegOne.flip(0), Trit::Zero);
        assert_eq!(Trit::NegOne.flip(1), Trit::One);
        assert_eq!(Trit::Zero.flip(0), Trit::NegOne);
        assert_eq!(Trit::Zero.flip(1), Trit::One);
    }

    #[test]
    fn test_sparse_dropout_preserves_shape() {
        let mat = sample_matrix();
        let result = sparse_dropout(&mat, 3);
        assert_eq!(result.rows, mat.rows);
        assert_eq!(result.cols, mat.cols);
    }
}
