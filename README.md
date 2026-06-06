# ternary-dropout

[![crates.io](https://img.shields.io/crates/v/ternary-dropout.svg)](https://crates.io/crates/ternary-dropout)
[![docs.rs](https://docs.rs/ternary-dropout/badge.svg)](https://docs.rs/ternary-dropout)

**Ternary dropout and regularization for neural networks in ℤ₃ = {-1, 0, 1}.**

A Rust library implementing dropout and regularization techniques adapted for ternary neural networks. All operations use a **seeded PRNG** (xorshift64) for deterministic, reproducible behavior—essential for testing, debugging, and reproducible research.

## Why Ternary Dropout?

Dropout is a foundational regularization technique. In ternary networks, dropout takes on unique characteristics:

- **Zeroing is natural** — Trit::Zero is a first-class value, not an artificial mask
- **Noise injection is structurally different** — flipping a trit means switching to one of two other values
- **Sparsity is free** — The zero trit already represents "no signal"
- **Uncertainty quantification** — Monte Carlo dropout with ternary values has interesting statistical properties

This library provides six regularization strategies, each with distinct inductive biases.

## Regularization Methods

### 1. Random Ternary Dropout
Classic dropout: set each element to `Trit::Zero` with probability `p`.

```rust
let result = random_dropout(&matrix, 0.5, &mut rng);
```

**Effect**: Introduces sparsity by randomly silencing trit connections. In ℤ₃, this is particularly effective because zero is a meaningful "neutral" element in ternary addition.

**When to use**: Standard regularization during training, just like traditional dropout.

### 2. Structured Dropout
Drop entire rows or columns instead of individual elements.

```rust
// Drop rows
let result = structured_dropout(&matrix, 0.3, true, &mut rng);
// Drop columns
let result = structured_dropout(&matrix, 0.3, false, &mut rng);
```

**Effect**: Forces the network to be robust to missing entire features or sequence positions. More aggressive than element-wise dropout.

**When to use**: When you want to prevent co-adaptation of entire feature groups or attention heads.

### 3. Ternary Noise Injection
Flip random trits to one of the other two trit values.

```rust
let result = noise_injection(&matrix, 0.1, &mut rng);
```

**Effect**: Unlike dropout (which only silences), noise injection can *change the sign* of a trit. This is a stronger perturbation that tests the network's resilience to adversarial-like input changes.

**When to use**: Data augmentation, adversarial robustness training, or when dropout alone isn't providing enough regularization.

### 4. Sparse Dropout
Keep only the top-k elements by magnitude (|{-1, +1}| = 1 > |{0}| = 0).

```rust
let result = sparse_dropout(&matrix, 10);
```

**Effect**: Enforces sparsity by keeping only the k most "active" positions. In ternary networks, this means keeping the non-zero trits preferentially.

**When to use**: When you want a hard sparsity constraint, similar to top-k activation in sparse transformers.

### 5. DropConnect
Randomly zero individual weight connections.

```rust
let result = dropconnect(&weights, 0.3, &mut rng);
```

**Effect**: Unlike dropout (which zeros activations), DropConnect zeros *weights*. This is structurally different—it changes the network's function rather than its input.

**When to use**: When you need stronger regularization than dropout, especially for overparameterized ternary networks.

### 6. Monte Carlo Dropout
Run multiple forward passes with dropout enabled to estimate uncertainty.

```rust
let mc = monte_carlo_dropout(&matrix, 0.5, 50, &mut rng);
// mc.mean — averaged output (quantized back to trit)
// mc.variance — uncertainty measure per element
```

**Effect**: Provides uncertainty estimates at each position. High variance means the output is sensitive to which connections are dropped—a useful signal for:
- Active learning (query uncertain predictions)
- Out-of-distribution detection
- Ensemble-like predictions without multiple models

**When to use**: Inference-time uncertainty estimation, especially for safety-critical applications.

## Seeded RNG

All methods use `TernaryRng`, a deterministic xorshift64 PRNG:

```rust
let mut rng = TernaryRng::new(42); // seed for reproducibility

// Generate random values
let u64 = rng.next_u64();
let f64 = rng.next_f64();    // [0, 1)
let trit = rng.next_trit();  // random {-1, 0, 1}
```

**Why seeded?**
- **Reproducible experiments**: Same seed → same dropout pattern → comparable results
- **Testability**: Assert exact output, not statistical properties
- **Distributed training**: Synchronize dropout masks across workers using the same seed

## Usage

```rust
use ternary_dropout::*;

let matrix = TernaryMatrix::from_flat(4, 8, /* ... trits ... */);
let mut rng = TernaryRng::new(42);

// Standard dropout
let dropped = random_dropout(&matrix, 0.5, &mut rng);

// Structured (row) dropout
let row_dropped = structured_dropout(&matrix, 0.3, true, &mut rng);

// Noise injection
let noisy = noise_injection(&matrix, 0.1, &mut rng);

// Keep top 10 elements
let sparse = sparse_dropout(&matrix, 10);

// DropConnect on weights
let sparse_weights = dropconnect(&weights, 0.3, &mut rng);

// Monte Carlo uncertainty
let mc = monte_carlo_dropout(&matrix, 0.5, 30, &mut rng);
println!("Mean: {:?}", mc.mean.data);
println!("Uncertainty: {:?}", mc.variance.data);
```

## Theoretical Notes

### Dropout in ℤ₃ vs ℝ

In continuous networks, dropout with rate `p` scales surviving activations by `1/(1-p)` during training (inverted dropout). In ℤ₃:
- No scaling is needed: ternary addition is closed in ℤ₃
- The "expected value" interpretation doesn't directly apply
- Instead, the regularization effect comes from **reducing the number of non-zero terms in each dot product**

### Variance Interpretation

Monte Carlo dropout variance in ℤ₃ measures the fraction of forward passes where each element was non-zero. This is mapped to a ternary uncertainty score:
- Low (< 33%): `Zero` — stable output
- Medium (33–66%): `One` — moderate uncertainty
- High (> 66%): `NegOne` — high uncertainty

### Noise Injection as Data Augmentation

Ternary noise injection is equivalent to a simple form of data augmentation:
- Flipping `1 → 0` or `1 → -1` tests sign sensitivity
- Flipping `0 → ±1` tests robustness to spurious activation
- This is analogous to Gaussian noise injection in continuous networks

## Testing

```bash
cargo test
```

All 16 tests cover:
- Random dropout: correct zeroing fraction, no-drop passthrough, full-drop
- Structured dropout: row and column modes preserve shape, zero correct structures
- Noise injection: changes some trits, no-flip passthrough
- Sparse dropout: keeps exactly top-k, handles k > total, k = 0
- DropConnect: zeros connections
- Monte Carlo: produces variance, no-drop recovers original
- RNG: deterministic reproducibility
- Trit flip: all 6 flip cases

## License

MIT
