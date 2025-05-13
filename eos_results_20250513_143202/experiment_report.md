# Edge of Stability Experiments Report

Generated on: 2025-05-13 14:44:46

## Experiment Summary

| Experiment ID | Status | Beta1 | Beta3 | Eta Values | Steps | Plots |
|--------------|--------|-------|-------|------------|-------|-------|
| baseline_comparison | ✅ Success | 0.9 | 1.0 | [0.001, 0.01, 0.05] | 100 | [plots_baseline_comparison](plots_baseline_comparison) |
| high_beta3 | ✅ Success | 0.9 | 3.0 | [0.01] | 100 | [plots_high_beta3](plots_high_beta3) |
| very_high_beta3 | ✅ Success | 0.9 | 10.0 | [0.01] | 100 | [plots_very_high_beta3](plots_very_high_beta3) |
| beta1_variation | ✅ Success | 0.95 | 1.0 | [0.01] | 100 | [plots_beta1_variation](plots_beta1_variation) |
| momentum_norm | ✅ Success | 0.9 | 1.0 | [0.01] | 100 | [plots_momentum_norm](plots_momentum_norm) |
| low_cutoff | ✅ Success | 0.9 | 1.0 | [0.01] | 100 | [plots_low_cutoff](plots_low_cutoff) |
| higher_power | ✅ Success | 0.9 | 1.0 | [0.01] | 100 | [plots_higher_power](plots_higher_power) |
| long_run_analysis | ✅ Success | 0.9 | 1.0 | [0.01] | 500 | [plots_long_run_analysis](plots_long_run_analysis) |

## Experiment Details

### Experiment: baseline_comparison

**Status**: Success

**Configuration**:
- eta_values: [0.001, 0.01, 0.05]
- beta1: 0.9
- beta2: 0.999
- beta3: 1.0
- num_steps: 100
- seed: 42
- use_wandb: False

**Results**: See plots in the [plots_baseline_comparison](plots_baseline_comparison) directory

**Available plots**:

- [eos_ratio_beta1_0.9_beta3_1.0.png](plots_baseline_comparison/eos_ratio_beta1_0.9_beta3_1.0.png)
- [eos_eta_0.001_beta1_0.9_beta3_1.0.png](plots_baseline_comparison/eos_eta_0.001_beta1_0.9_beta3_1.0.png)
- [eos_eta_0.01_beta1_0.9_beta3_1.0.png](plots_baseline_comparison/eos_eta_0.01_beta1_0.9_beta3_1.0.png)
- [eos_all_etas_beta1_0.9_beta3_1.0.png](plots_baseline_comparison/eos_all_etas_beta1_0.9_beta3_1.0.png)
- [eos_eta_0.05_beta1_0.9_beta3_1.0.png](plots_baseline_comparison/eos_eta_0.05_beta1_0.9_beta3_1.0.png)

---

### Experiment: high_beta3

**Status**: Success

**Configuration**:
- eta_values: [0.01]
- beta1: 0.9
- beta2: 0.999
- beta3: 3.0
- num_steps: 100
- seed: 42
- use_wandb: False

**Results**: See plots in the [plots_high_beta3](plots_high_beta3) directory

**Available plots**:

- [eos_ratio_beta1_0.9_beta3_3.0.png](plots_high_beta3/eos_ratio_beta1_0.9_beta3_3.0.png)
- [eos_eta_0.01_beta1_0.9_beta3_3.0.png](plots_high_beta3/eos_eta_0.01_beta1_0.9_beta3_3.0.png)
- [eos_all_etas_beta1_0.9_beta3_3.0.png](plots_high_beta3/eos_all_etas_beta1_0.9_beta3_3.0.png)

---

### Experiment: very_high_beta3

**Status**: Success

**Configuration**:
- eta_values: [0.01]
- beta1: 0.9
- beta2: 0.999
- beta3: 10.0
- num_steps: 100
- seed: 42
- use_wandb: False

**Results**: See plots in the [plots_very_high_beta3](plots_very_high_beta3) directory

**Available plots**:

- [eos_ratio_beta1_0.9_beta3_10.0.png](plots_very_high_beta3/eos_ratio_beta1_0.9_beta3_10.0.png)
- [eos_eta_0.01_beta1_0.9_beta3_10.0.png](plots_very_high_beta3/eos_eta_0.01_beta1_0.9_beta3_10.0.png)
- [eos_all_etas_beta1_0.9_beta3_10.0.png](plots_very_high_beta3/eos_all_etas_beta1_0.9_beta3_10.0.png)

---

### Experiment: beta1_variation

**Status**: Success

**Configuration**:
- eta_values: [0.01]
- beta1: 0.95
- beta2: 0.999
- beta3: 1.0
- num_steps: 100
- seed: 42
- use_wandb: False

**Results**: See plots in the [plots_beta1_variation](plots_beta1_variation) directory

**Available plots**:

- [eos_ratio_beta1_0.95_beta3_1.0.png](plots_beta1_variation/eos_ratio_beta1_0.95_beta3_1.0.png)
- [eos_eta_0.01_beta1_0.95_beta3_1.0.png](plots_beta1_variation/eos_eta_0.01_beta1_0.95_beta3_1.0.png)
- [eos_all_etas_beta1_0.95_beta3_1.0.png](plots_beta1_variation/eos_all_etas_beta1_0.95_beta3_1.0.png)

---

### Experiment: momentum_norm

**Status**: Success

**Configuration**:
- eta_values: [0.01]
- beta1: 0.9
- beta2: 0.999
- beta3: 1.0
- num_steps: 100
- seed: 42
- normgrad: False
- use_wandb: False

**Results**: See plots in the [plots_momentum_norm](plots_momentum_norm) directory

**Available plots**:

- [eos_ratio_beta1_0.9_beta3_1.0.png](plots_momentum_norm/eos_ratio_beta1_0.9_beta3_1.0.png)
- [eos_eta_0.01_beta1_0.9_beta3_1.0.png](plots_momentum_norm/eos_eta_0.01_beta1_0.9_beta3_1.0.png)
- [eos_all_etas_beta1_0.9_beta3_1.0.png](plots_momentum_norm/eos_all_etas_beta1_0.9_beta3_1.0.png)

---

### Experiment: low_cutoff

**Status**: Success

**Configuration**:
- eta_values: [0.01]
- beta1: 0.9
- beta2: 0.999
- beta3: 1.0
- num_steps: 100
- seed: 42
- lr_cutoff: 5
- use_wandb: False

**Results**: See plots in the [plots_low_cutoff](plots_low_cutoff) directory

**Available plots**:

- [eos_ratio_beta1_0.9_beta3_1.0.png](plots_low_cutoff/eos_ratio_beta1_0.9_beta3_1.0.png)
- [eos_eta_0.01_beta1_0.9_beta3_1.0.png](plots_low_cutoff/eos_eta_0.01_beta1_0.9_beta3_1.0.png)
- [eos_all_etas_beta1_0.9_beta3_1.0.png](plots_low_cutoff/eos_all_etas_beta1_0.9_beta3_1.0.png)

---

### Experiment: higher_power

**Status**: Success

**Configuration**:
- eta_values: [0.01]
- beta1: 0.9
- beta2: 0.999
- beta3: 1.0
- num_steps: 100
- seed: 42
- power: 4
- use_wandb: False

**Results**: See plots in the [plots_higher_power](plots_higher_power) directory

**Available plots**:

- [eos_ratio_beta1_0.9_beta3_1.0.png](plots_higher_power/eos_ratio_beta1_0.9_beta3_1.0.png)
- [eos_eta_0.01_beta1_0.9_beta3_1.0.png](plots_higher_power/eos_eta_0.01_beta1_0.9_beta3_1.0.png)
- [eos_all_etas_beta1_0.9_beta3_1.0.png](plots_higher_power/eos_all_etas_beta1_0.9_beta3_1.0.png)

---

### Experiment: long_run_analysis

**Status**: Success

**Configuration**:
- eta_values: [0.01]
- beta1: 0.9
- beta2: 0.999
- beta3: 1.0
- num_steps: 500
- seed: 42
- use_wandb: False

**Results**: See plots in the [plots_long_run_analysis](plots_long_run_analysis) directory

**Available plots**:

- [eos_ratio_beta1_0.9_beta3_1.0.png](plots_long_run_analysis/eos_ratio_beta1_0.9_beta3_1.0.png)
- [eos_eta_0.01_beta1_0.9_beta3_1.0.png](plots_long_run_analysis/eos_eta_0.01_beta1_0.9_beta3_1.0.png)
- [eos_all_etas_beta1_0.9_beta3_1.0.png](plots_long_run_analysis/eos_all_etas_beta1_0.9_beta3_1.0.png)

---

