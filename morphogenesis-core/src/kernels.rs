//! The five bits of maths the whole model is built from.
//!
//! Each one is small, does exactly one job, and can be tested on its own.
//! Everything else in the engine is these five functions called in the right
//! order with the right numbers.

/// Rescale a list of numbers so they're a sensible size, then apply a
/// learned per-slot weight.
///
/// Why it exists: numbers flow through 28 layers, each adding to them. Without
/// rescaling they'd grow or shrink out of control. This pulls them back.
///
/// ```text
///              x[i]
///   y[i] = ---------------- x g[i]
///          sqrt(avg(x*x) + eps)
/// ```
///
/// `g` is a learned weight from the model file (e.g. `input_layernorm.weight`).
/// `eps` is a tiny number (1e-6) that only exists so we never divide by zero.
pub fn rmsnorm(x: &[f32], g: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), g.len(), "rmsnorm: x and g must be the same length");

    // Square everything and take the average.
    let mean_square = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;

    // One division is cheaper than doing it per element, so flip it once here.
    let scale = 1.0 / (mean_square + eps).sqrt();

    x.iter().zip(g).map(|(xi, gi)| xi * scale * gi).collect()
}

/// Turn a list of scores into probabilities that add up to 1.
///
/// Used to decide how much attention each word pays to each other word.
/// Bigger score in, bigger share out.
///
/// Changes the list in place rather than making a new one, because attention
/// calls this constantly and we don't want to allocate every time.
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    // Find the biggest value and subtract it from everything first.
    //
    // This is NOT optional. e^x explodes fast -- e^100 is already too big for
    // an f32 and comes out as infinity. Subtracting the max makes the largest
    // value e^0 = 1, so nothing ever overflows. The final answer is exactly
    // the same either way, because it's a ratio.
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut total = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        total += *v;
    }

    // Divide by the total so everything adds up to 1.
    for v in x.iter_mut() {
        *v /= total;
    }
}

/// A smooth on/off switch: keeps positive numbers, squashes negative ones.
///
///   silu(x) = x * sigmoid(x) = x / (1 + e^-x)
///
/// Used in the part of each layer where every word "thinks" on its own.
/// Smooth rather than a hard cutoff, which helps during training.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Multiply a matrix by a vector: the workhorse of the whole engine.
///
/// This is what a "projection" (q_proj, k_proj, ...) actually does, and it's
/// where ~99% of the time and 100% of the disk reads go.
///
/// `w` is stored as `out_dim` rows, each `in_dim` long, laid end to end --
/// exactly how it sits in the model file, so no rearranging needed.
///
/// Each output number is one row of `w` dot-multiplied with `x`:
/// "how much does x look like the pattern in row i?"
pub fn matvec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    assert_eq!(w.len(), out_dim * in_dim, "matvec: weight size mismatch");
    assert_eq!(x.len(), in_dim, "matvec: input size mismatch");

    (0..out_dim)
        .map(|i| {
            let row = &w[i * in_dim..(i + 1) * in_dim];
            row.iter().zip(x).map(|(a, b)| a * b).sum()
        })
        .collect()
}

/// Stamp "where in the sentence am I" onto a vector, by rotating it.
///
/// Attention on its own has no sense of order -- shuffle the words and it
/// gives the same answer. RoPE fixes that by spinning pairs of numbers by an
/// angle that depends on the position.
///
/// The clever part: after rotating, comparing a word at position 5 with one at
/// position 2 only depends on the GAP (3), not the absolute positions. So the
/// model learns "three words back" rather than "at slot 5".
///
/// Operates on one head's slice (128 numbers here), in place.
///
/// WARNING -- pairing convention. There are two ways to pair up the numbers,
/// and picking the wrong one gives fluent-sounding nonsense with no error.
/// Qwen3 pairs the first half with the second half: slot k with slot k+64.
/// (The other convention pairs neighbours: 0 with 1, 2 with 3.)
pub fn rope(x: &mut [f32], pos: usize, theta_base: f32) {
    let head_dim = x.len();
    let half = head_dim / 2;

    for k in 0..half {
        // Each pair spins at its own speed. Low k spins fast (tracks nearby
        // positions), high k spins slowly (tracks far-apart positions).
        let freq = 1.0 / theta_base.powf(2.0 * k as f32 / head_dim as f32);
        let (sin, cos) = (pos as f32 * freq).sin_cos();

        // Standard 2D rotation of the pair (a, b).
        let a = x[k];
        let b = x[k + half];
        x[k] = a * cos - b * sin;
        x[k + half] = b * cos + a * sin;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Check two lists are equal, allowing for tiny floating-point wobble.
    fn assert_close(got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len(), "different lengths");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() < tol,
                "index {i}: got {g}, wanted {w} (tolerance {tol})"
            );
        }
    }

    #[test]
    fn rmsnorm_matches_hand_calculation() {
        // squares are 1, 4, 9, 16 -> average 7.5 -> sqrt = 2.7386128
        let x = [1.0, 2.0, 3.0, 4.0];
        let g = [1.0; 4];
        let want = [0.36514837, 0.73029674, 1.0954451, 1.4605935];
        assert_close(&rmsnorm(&x, &g, 0.0), &want, 1e-6);
    }

    #[test]
    fn rmsnorm_output_always_has_rms_of_one() {
        // True for ANY input when g is all ones -- no expected values needed.
        for x in [
            vec![1.0, 2.0, 3.0, 4.0],
            vec![-5.0, 0.5, 100.0, -0.001, 7.0],
            vec![0.25; 1024],
        ] {
            let g = vec![1.0; x.len()];
            let y = rmsnorm(&x, &g, 0.0);
            let rms = (y.iter().map(|v| v * v).sum::<f32>() / y.len() as f32).sqrt();
            assert!((rms - 1.0).abs() < 1e-5, "rms was {rms}");
        }
    }

    #[test]
    fn rmsnorm_applies_the_learned_weight() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let g = [2.0, 2.0, 2.0, 2.0];
        let plain = rmsnorm(&x, &[1.0; 4], 0.0);
        let scaled = rmsnorm(&x, &g, 0.0);
        for (p, s) in plain.iter().zip(&scaled) {
            assert!((s - p * 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn softmax_adds_up_to_one() {
        let mut x = [1.0, 2.0, 3.0, 4.0, 5.0];
        softmax(&mut x);
        let total: f32 = x.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "total was {total}");
        assert!(x.iter().all(|v| (0.0..=1.0).contains(v)));
    }

    #[test]
    fn softmax_keeps_the_order() {
        let mut x = [3.0, 1.0, 2.0];
        softmax(&mut x);
        assert!(x[0] > x[2] && x[2] > x[1]);
    }

    #[test]
    fn softmax_survives_huge_numbers() {
        // Without subtracting the max first, this would be inf/inf = NaN.
        let mut x = [1000.0, 1001.0, 1002.0];
        softmax(&mut x);
        assert!(x.iter().all(|v| v.is_finite()), "got {x:?}");
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_equal_scores_split_evenly() {
        let mut x = [5.0; 4];
        softmax(&mut x);
        assert_close(&x, &[0.25; 4], 1e-6);
    }

    #[test]
    fn silu_behaves() {
        assert!((silu(0.0) - 0.0).abs() < 1e-7);
        // large positive -> passes through almost unchanged
        assert!((silu(20.0) - 20.0).abs() < 1e-4);
        // large negative -> squashed to nearly nothing
        assert!(silu(-20.0).abs() < 1e-6);
        // known value: 1 * sigmoid(1) = 0.7310586
        assert!((silu(1.0) - 0.7310586).abs() < 1e-6);
    }

    #[test]
    fn matvec_with_identity_returns_the_input() {
        // 3x3 identity matrix
        let w = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let x = [7.0, 8.0, 9.0];
        assert_close(&matvec(&w, &x, 3, 3), &x, 1e-6);
    }

    #[test]
    fn matvec_matches_hand_calculation() {
        // rows: [1,2,3] and [4,5,6];  x = [1,1,1]
        let w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0, 1.0, 1.0];
        assert_close(&matvec(&w, &x, 2, 3), &[6.0, 15.0], 1e-6);
    }

    #[test]
    fn matvec_is_linear() {
        // W(a+b) should equal W(a) + W(b) -- true for any matrix.
        let w: Vec<f32> = (0..12).map(|i| i as f32 * 0.37 - 2.0).collect();
        let a = [1.0, -2.0, 0.5, 3.0];
        let b = [0.25, 4.0, -1.5, 2.0];
        let sum: Vec<f32> = a.iter().zip(&b).map(|(p, q)| p + q).collect();

        let wa = matvec(&w, &a, 3, 4);
        let wb = matvec(&w, &b, 3, 4);
        let want: Vec<f32> = wa.iter().zip(&wb).map(|(p, q)| p + q).collect();

        assert_close(&matvec(&w, &sum, 3, 4), &want, 1e-4);
    }

    #[test]
    fn rope_does_not_change_length() {
        // Rotating a vector spins it around; it never stretches or shrinks it.
        // This holds at every position, which makes it a strong free check.
        let original: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let length = |v: &[f32]| v.iter().map(|a| a * a).sum::<f32>().sqrt();

        for pos in [0, 1, 7, 100, 4096] {
            let mut x = original.clone();
            rope(&mut x, pos, 1_000_000.0);
            assert!(
                (length(&x) - length(&original)).abs() < 1e-3,
                "position {pos} changed the length"
            );
        }
    }

    #[test]
    fn rope_at_position_zero_changes_nothing() {
        // Angle is 0, so cos=1 and sin=0 -- every number stays put.
        let original: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let mut x = original.clone();
        rope(&mut x, 0, 1_000_000.0);
        assert_close(&x, &original, 1e-6);
    }

    #[test]
    fn rope_only_depends_on_the_gap() {
        // The whole point of RoPE: comparing a word at position m with one at
        // position n gives the same answer for any pair with the same gap.
        let q: Vec<f32> = (0..128).map(|i| (i as f32 * 0.3).cos()).collect();
        let k: Vec<f32> = (0..128).map(|i| (i as f32 * 0.7).sin()).collect();
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(p, q)| p * q).sum::<f32>();

        let score = |m: usize, n: usize| {
            let (mut qq, mut kk) = (q.clone(), k.clone());
            rope(&mut qq, m, 1_000_000.0);
            rope(&mut kk, n, 1_000_000.0);
            dot(&qq, &kk)
        };

        // gap of 3, measured at three different places in the sentence
        let a = score(3, 0);
        let b = score(10, 7);
        let c = score(50, 47);
        assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        assert!((a - c).abs() < 1e-3, "{a} vs {c}");
    }
}
