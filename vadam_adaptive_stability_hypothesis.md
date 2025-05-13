# VADAM and the Adaptive Edge of Stability: Theoretical Analysis

## Background

The paper ["Adaptive Gradient Methods at the Edge of Stability"](https://arxiv.org/abs/2207.14484) by Cohen et al. establishes that Adam with β₁ = 0.9 exhibits an "Adaptive Edge of Stability" (AEoS) phenomenon where the maximum eigenvalue of the preconditioned Hessian equilibrates at approximately 38/η, where η is the learning rate.

This document presents a theoretical analysis of why our Velocity-Adaptive Momentum (VADAM) optimizer might be able to surpass this bound and maintain stability at higher eigenvalues.

## Key Insights from the AEoS Paper

1. Unlike SGD, which exhibits the Edge of Stability with a bound of 2/η, Adam's adaptive nature allows it to operate at a higher eigenvalue threshold of approximately 38/η (for β₁ = 0.9).

2. When adaptive methods reach this threshold, they don't bounce back like SGD but rather adapt their preconditioner to compensate, allowing further advancement into high-curvature regions.

3. The AEoS depends on the momentum parameter β₁, with higher β₁ values resulting in higher stability thresholds.

## VADAM's Mechanism

VADAM extends Adam by introducing:

1. **Velocity-dependent learning rate**: The effective learning rate is modulated by the squared norm of the gradient or momentum buffer.

2. **Beta3 parameter**: Controls the sensitivity of the learning rate to the norm of the gradient/momentum.

3. **Power parameter**: Determines which Lp-norm to use (default is p=2).

4. **lr_cutoff parameter**: Limits the minimum learning rate to η/(lr_cutoff+1).

The VADAM update rule modifies the effective learning rate as:

```
lr_effective = η / (1 + min(β₃ * ||g||^p, lr_cutoff))
```

where g is either the gradient (if normgrad=True) or momentum buffer (if normgrad=False).

## Theoretical Extension of the AEoS Bound

We hypothesize that VADAM can exceed the AEoS bound of 38/η for Adam through the following mechanisms:

### 1. Automatic Learning Rate Adaptation

When approaching regions of high curvature (large eigenvalues), the gradient norm typically increases. VADAM responds by automatically reducing the effective learning rate, potentially allowing stability in regions where Adam would become unstable.

The theoretical AEoS bound can be extended to:

```
AEoS_VADAM ≈ (38/η) * (1 + β₃ * ||g||^p)
```

for appropriate values of β₃ and p.

### 2. Curvature-Aware Optimization

By modulating the learning rate based on the gradient or momentum norm, VADAM implicitly incorporates curvature information into the optimization process, beyond what Adam's second-moment scaling provides.

### 3. Preconditioner Enhancement

VADAM's effective preconditioner combines:
- Adam's standard preconditioner (based on EMA of squared gradients)
- The additional velocity-dependent scaling factor

This dual preconditioning allows VADAM to adapt more efficiently to the local geometry of the loss landscape.

## Hypothesis Testing

Our edge_of_stability_analysis.py script is designed to test this hypothesis by:

1. Measuring the maximum eigenvalue of the preconditioned Hessian for both Adam and VADAM during training.

2. Comparing these values to the theoretical AEoS bound of 38/η for Adam.

3. Examining whether VADAM can maintain stability at eigenvalues exceeding this bound.

4. Testing different configurations of η, β₁, β₃, and p to find optimal settings.

## Potential Parameter Settings to Explore

Based on our theoretical understanding, we suggest exploring:

1. **β₃ values**: Larger values (1.0-5.0) should increase stability by making the learning rate more responsive to gradient magnitudes.

2. **Power parameter (p)**: Higher values might make the method more sensitive to outlier gradients, potentially enhancing stability in some cases.

3. **normgrad setting**: Using the momentum buffer norm (normgrad=False) might provide smoother adaptation compared to using the gradient norm directly.

4. **lr_cutoff**: Lower values allow a wider range of effective learning rates, potentially helping navigate highly non-uniform landscapes.

## Expected Results

If our hypothesis is correct, we should observe:

1. VADAM maintaining stability at eigenvalues significantly higher than 38/η, particularly for larger β₃ values.

2. More consistent training dynamics in the presence of sharp curvature.

3. The ratio of VADAM's max eigenvalue to Adam's theoretical bound exceeding 1.0 while maintaining stable optimization.

The analysis will provide valuable insights into how adaptive learning rate methods can be further enhanced to handle challenging optimization landscapes. 