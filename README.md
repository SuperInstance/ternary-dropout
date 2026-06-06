# ternary-dropout

**Regularization in {-1, 0, +1} — where zero is a first-class citizen and flipping a trit means something.**

[![crates.io](https://img.shields.io/crates/v/ternary-dropout.svg)](https://crates.io/crates/ternary-dropout)
[![docs.rs](https://docs.rs/ternary-dropout/badge.svg)](https://docs.rs/ternary-dropout)

## Why This Exists

Dropout prevents neural networks from memorizing training data by randomly silencing activations during training. It's embarrassingly simple: set values to zero with probability p. The network can't rely on any single neuron, so it learns redundant, robust representations.

In float networks, dropout introduces artificial zeros. In ternary networks, zero is already a meaningful value — it represents "neutral" or "no signal." Ternary dropout doesn't introduce a foreign concept; it *amplifies an existing one*. This makes dropout particularly natural in Z₃.

But ternary space also enables something float dropout can't do: **trit flipping.** Instead of silencing a value, you can *change its sign*. Flipping a +1 to -1 isn't just noise — it's adversarial training built into the architecture. This crate provides both.

## The Key Insight

Ternary dropout has three operations that float dropout doesn't:

1. **Zeroing** — natural, because zero is already in the alphabet
2. **Flipping** — change +1 to -1 or vice versa, a sign-reversal attack
3. **Voting** — majority pooling after dropout acts as ensemble consensus

These three operations provide a richer regularization landscape than float dropout's single "set to zero" move.

## Quick Start

```toml
[dependencies]
ternary-dropout = "0.1"
```

```rust
use ternary_dropout::*;

let matrix = TernaryMatrix::from_flat(3, 4, vec![
    Trit::One, Trit::NegOne, Trit::Zero, Trit::One,
    Trit::Zero, Trit::One, Trit::One, Trit::NegOne,
    Trit::NegOne, Trit::Zero, Trit::One, Trit::Zero,
]);

let mut rng = TernaryRng::new(42);

// Standard dropout — zero random elements
let dropped = random_dropout(&matrix, 0.5, &mut rng);

// Structured dropout — zero entire rows or columns
let row_dropped = structured_dropout(&matrix, 0.3, true, &mut rng);  // drop rows
let col_dropped = structured_dropout(&matrix, 0.3, false, &mut rng); // drop columns

// Noise injection — flip random trits (stronger than dropout)
let noisy = noise_injection(&matrix, 0.1, &mut rng);

// Sparse dropout — keep only top-k by magnitude
let sparse = sparse_dropout(&matrix, 6);

// DropConnect — zero weights (not activations)
let sparse_weights = dropconnect(&weights, 0.3, &mut rng);

// Monte Carlo dropout — uncertainty estimation
let mc = monte_carlo_dropout(&matrix, 0.5, 30, &mut rng);
println!("Mean:       {:?}", mc.mean.data);
println!("Uncertainty: {:?}", mc.variance.data);
```

## Architecture

```
┌───────────────────────────────────────────────────┐
│                  TernaryMatrix                     │
│  (rows × cols, Vec<Trit>, row-major)              │
└───────────────────────┬───────────────────────────┘
                        │
    ┌───────────┬───────┼───────┬──────────────┐
    │           │       │       │              │
┌───▼───┐ ┌────▼────┐ ┌▼──────┐ ▼──────────┐ ┌▼──────────┐
│Random │ │Structured│ │Noise  │ Sparse     │ │ Monte     │
│Dropout│ │Dropout   │ │Inject │ Dropout    │ │ Carlo     │
│(zero) │ │(row/col) │ │(flip) │ (top-k)    │ │ Dropout   │
└───────┘ └─────────┘ └───────┘ └───────────┘ └───────────┘
```

## Regularization Strategies

### Random Dropout

Set each element to `Trit::Zero` with probability `p`. The ternary equivalent of standard dropout.

**Why it works here:** Zero is the additive identity in Z₃. Dropping an element from a sum is the same as setting it to zero — no scaling adjustment needed (unlike inverted dropout in float networks).

### Structured Dropout

Drop entire rows or columns. Forces the network to be robust to missing feature groups or spatial positions.

**When to use:** Preventing co-adaptation of feature groups, attention heads, or sequence positions.

### Noise Injection (Trit Flipping)

Flip random trits to one of the other two values. Unlike dropout (which only silences), flipping can *reverse the sign* of a signal. This is a stronger perturbation — adversarial training built into the forward pass.

```rust
// Flipping: each trit maps to one of two alternatives
Trit::One.flip(0)    → NegOne
Trit::One.flip(1)    → Zero
Trit::NegOne.flip(0) → Zero
Trit::NegOne.flip(1) → One
Trit::Zero.flip(0)   → NegOne
Trit::Zero.flip(1)   → One
```

**When to use:** Data augmentation, adversarial robustness, when dropout alone isn't enough.

### Sparse Dropout

Keep only the top-k elements by magnitude (|-1| = |+1| = 1 > |0| = 0). Non-zero trits are preserved preferentially.

**When to use:** Hard sparsity constraints, sparse attention mechanisms.

### DropConnect

Randomly zero individual *weight* connections. Structurally different from dropout: it changes the network's function, not its input.

**When to use:** Stronger regularization than dropout, especially for overparameterized ternary networks.

### Monte Carlo Dropout

Run N forward passes with dropout enabled. Each pass produces a different result due to random masking. Aggregate to estimate uncertainty:

- **Mean**: averaged output, mapped back to nearest trit
- **Variance**: fraction of passes where each element was non-zero, mapped to a ternary uncertainty score

```rust
let mc = monte_carlo_dropout(&matrix, 0.5, 50, &mut rng);
// mc.mean: consensus output
// mc.variance: per-element uncertainty (Zero=stable, One=moderate, NegOne=high)
```

**When to use:** Inference-time uncertainty estimation, active learning, out-of-distribution detection.

## Seeded RNG

All methods use `TernaryRng`, a deterministic xorshift64 PRNG:

```rust
let mut rng = TernaryRng::new(42);      // seed for reproducibility
rng.next_u64();                          // random u64
rng.next_f64();                          // random [0, 1)
rng.next_trit();                         // random {-1, 0, +1}
```

**Why seeded?** Reproducible experiments. Same seed → same dropout mask → comparable results across runs. Essential for debugging, ablation studies, and distributed training synchronization.

## API Reference

### Core Types

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Trit { NegOne = -1, Zero = 0, One = 1 }

struct TernaryRng { /* xorshift64 state */ }
impl TernaryRng {
    fn new(seed: u64) -> Self;
    fn next_u64(&mut self) -> u64;
    fn next_f64(&mut self) -> f64;
    fn next_trit(&mut self) -> Trit;
}

struct TernaryMatrix { pub rows: usize, pub cols: usize, pub data: Vec<Trit> }
```

### Regularization Functions

```rust
fn random_dropout(matrix: &TernaryMatrix, drop_rate: f64, rng: &mut TernaryRng) -> TernaryMatrix;
fn structured_dropout(matrix: &TernaryMatrix, drop_rate: f64, drop_rows: bool, rng: &mut TernaryRng) -> TernaryMatrix;
fn noise_injection(matrix: &TernaryMatrix, flip_rate: f64, rng: &mut TernaryRng) -> TernaryMatrix;
fn sparse_dropout(matrix: &TernaryMatrix, k: usize) -> TernaryMatrix;
fn dropconnect(weights: &TernaryMatrix, drop_rate: f64, rng: &mut TernaryRng) -> TernaryMatrix;
fn monte_carlo_dropout(matrix: &TernaryMatrix, drop_rate: f64, num_samples: usize, rng: &mut TernaryRng) -> MCOutput;
```

### MC Output

```rust
struct MCOutput {
    pub mean: TernaryMatrix,      // averaged, quantized to trit
    pub variance: TernaryMatrix,  // uncertainty per element
}
```

## Real-World Example: Safety-Critical Ternary Inference

A medical imaging system uses a ternary network to classify X-rays as "abnormal" (-1), "uncertain" (0), or "normal" (+1). False confidence is dangerous — the system must know when it doesn't know.

```rust
let mut rng = TernaryRng::new(42);
let scan_features = extract_features(&xray);
let mc = monte_carlo_dropout(&scan_features, 0.3, 100, &mut rng);

match mc.mean.data[0] {
    Trit::One => {
        // Check uncertainty
        if mc.variance.data[0] == Trit::NegOne {
            // High variance — model is uncertain
            flag_for_human_review("Normal prediction, but uncertain");
        } else {
            report_normal();
        }
    }
    Trit::NegOne => flag_for_human_review("Abnormal detected"),
    Trit::Zero => flag_for_human_review("Uncertain — needs review"),
}
```

Monte Carlo dropout provides calibrated uncertainty without training a separate uncertainty model. The ternary variance maps directly to clinical decision thresholds.

## Performance Characteristics

- **Random/Structured dropout**: O(n) — one comparison per element or row/column
- **Noise injection**: O(n) — one comparison + one flip per element
- **Sparse dropout**: O(n log n) — sort by magnitude
- **DropConnect**: O(n) — same as random dropout
- **Monte Carlo**: O(N × n) — N forward passes over n elements

Memory: All operations return a new `TernaryMatrix`. Each `Trit` is 1 byte. MC dropout with 100 samples uses ~100n temporary bytes.

## Ecosystem Connections

Dropout/regularization integrates throughout the ternary pipeline:

- [`ternary-activation`](https://github.com/SuperInstance/ternary-activation) — activations are what dropout silences
- [`ternary-conv`](https://github.com/SuperInstance/ternary-conv) — dropout applied after convolution layers
- [`ternary-matmul`](https://github.com/SuperInstance/ternary-matmul) — DropConnect applied to weight matrices
- [`ternary-pool`](https://github.com/SuperInstance/ternary-pool) — majority pooling provides natural ensemble averaging

## Open Questions

- **Curriculum dropout**: Start with low dropout rate, increase during training. Does this help ternary networks as much as it helps float networks?
- **Trit-aware uncertainty**: The MC variance maps to {Zero, One, NegOne} — a 3-level uncertainty scale. A richer encoding (e.g., per-class variance) might be more informative.
- **Layer-wise dropout rates**: Different layers might benefit from different dropout strategies (structured for attention, random for FFN).

## Testing

```bash
cargo test
```

16 tests covering: random dropout zeroing fraction and edge cases (no-drop, full-drop), structured dropout row/column modes, noise injection (some changed, none changed), sparse dropout (exact top-k, k > total, k = 0), DropConnect zeroing, MC dropout variance and no-drop recovery, RNG determinism, trit flip all 6 cases.

## License

MIT
