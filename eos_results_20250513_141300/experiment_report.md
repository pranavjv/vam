# Edge of Stability Experiments Report

Generated on: 2025-05-13 14:29:20

## Experiment Summary

| Experiment ID | Status | Beta1 | Beta3 | Eta Values | Steps | Plots |
|--------------|--------|-------|-------|------------|-------|-------|
| basic | ✅ Success | 0.9 | 1.0 | [0.001, 0.005, 0.01, 0.05] | 100 | [plots_basic](plots_basic) |
| beta3_0.5 | ✅ Success | 0.9 | 0.5 | [0.01] | 100 | [plots_beta3_0.5](plots_beta3_0.5) |
| beta3_1.0 | ✅ Success | 0.9 | 1.0 | [0.01] | 100 | [plots_beta3_1.0](plots_beta3_1.0) |
| beta3_2.0 | ✅ Success | 0.9 | 2.0 | [0.01] | 100 | [plots_beta3_2.0](plots_beta3_2.0) |
| beta3_5.0 | ✅ Success | 0.9 | 5.0 | [0.01] | 100 | [plots_beta3_5.0](plots_beta3_5.0) |
| beta1_0.7 | ✅ Success | 0.7 | 1.0 | [0.01] | 100 | [plots_beta1_0.7](plots_beta1_0.7) |
| beta1_0.8 | ✅ Success | 0.8 | 1.0 | [0.01] | 100 | [plots_beta1_0.8](plots_beta1_0.8) |
| beta1_0.9 | ✅ Success | 0.9 | 1.0 | [0.01] | 100 | [plots_beta1_0.9](plots_beta1_0.9) |
| beta1_0.95 | ✅ Success | 0.95 | 1.0 | [0.01] | 100 | [plots_beta1_0.95](plots_beta1_0.95) |
| normgrad_true | ❌ Failed | 0.9 | 1.0 | [0.01] | 100 | [plots_normgrad_true](plots_normgrad_true) |
| normgrad_false | ✅ Success | 0.9 | 1.0 | [0.01] | 100 | [plots_normgrad_false](plots_normgrad_false) |
| long_run | ✅ Success | 0.9 | 1.0 | [0.01] | 500 | [plots_long_run](plots_long_run) |

## Experiment Details

### Experiment: basic

**Status**: Success

**Configuration**:
- eta_values: [0.001, 0.005, 0.01, 0.05]
- beta1: 0.9
- beta2: 0.999
- beta3: 1.0
- num_steps: 100
- seed: 42

**Results**: See plots in the [plots_basic](plots_basic) directory

**Available plots**:

- [eos_ratio_beta1_0.9_beta3_1.0.png](plots_basic/eos_ratio_beta1_0.9_beta3_1.0.png)
- [eos_eta_0.001_beta1_0.9_beta3_1.0.png](plots_basic/eos_eta_0.001_beta1_0.9_beta3_1.0.png)
- [eos_eta_0.01_beta1_0.9_beta3_1.0.png](plots_basic/eos_eta_0.01_beta1_0.9_beta3_1.0.png)
- [eos_eta_0.005_beta1_0.9_beta3_1.0.png](plots_basic/eos_eta_0.005_beta1_0.9_beta3_1.0.png)
- [eos_all_etas_beta1_0.9_beta3_1.0.png](plots_basic/eos_all_etas_beta1_0.9_beta3_1.0.png)
- [eos_eta_0.05_beta1_0.9_beta3_1.0.png](plots_basic/eos_eta_0.05_beta1_0.9_beta3_1.0.png)

---

### Experiment: beta3_0.5

**Status**: Success

**Configuration**:
- eta_values: [0.01]
- beta1: 0.9
- beta2: 0.999
- beta3: 0.5
- num_steps: 100
- seed: 42

**Results**: See plots in the [plots_beta3_0.5](plots_beta3_0.5) directory

**Available plots**:

- [eos_eta_0.01_beta1_0.9_beta3_0.5.png](plots_beta3_0.5/eos_eta_0.01_beta1_0.9_beta3_0.5.png)
- [eos_all_etas_beta1_0.9_beta3_0.5.png](plots_beta3_0.5/eos_all_etas_beta1_0.9_beta3_0.5.png)
- [eos_ratio_beta1_0.9_beta3_0.5.png](plots_beta3_0.5/eos_ratio_beta1_0.9_beta3_0.5.png)

---

### Experiment: beta3_1.0

**Status**: Success

**Configuration**:
- eta_values: [0.01]
- beta1: 0.9
- beta2: 0.999
- beta3: 1.0
- num_steps: 100
- seed: 42

**Results**: See plots in the [plots_beta3_1.0](plots_beta3_1.0) directory

**Available plots**:

- [eos_ratio_beta1_0.9_beta3_1.0.png](plots_beta3_1.0/eos_ratio_beta1_0.9_beta3_1.0.png)
- [eos_eta_0.01_beta1_0.9_beta3_1.0.png](plots_beta3_1.0/eos_eta_0.01_beta1_0.9_beta3_1.0.png)
- [eos_all_etas_beta1_0.9_beta3_1.0.png](plots_beta3_1.0/eos_all_etas_beta1_0.9_beta3_1.0.png)

---

### Experiment: beta3_2.0

**Status**: Success

**Configuration**:
- eta_values: [0.01]
- beta1: 0.9
- beta2: 0.999
- beta3: 2.0
- num_steps: 100
- seed: 42

**Results**: See plots in the [plots_beta3_2.0](plots_beta3_2.0) directory

**Available plots**:

- [eos_eta_0.01_beta1_0.9_beta3_2.0.png](plots_beta3_2.0/eos_eta_0.01_beta1_0.9_beta3_2.0.png)
- [eos_ratio_beta1_0.9_beta3_2.0.png](plots_beta3_2.0/eos_ratio_beta1_0.9_beta3_2.0.png)
- [eos_all_etas_beta1_0.9_beta3_2.0.png](plots_beta3_2.0/eos_all_etas_beta1_0.9_beta3_2.0.png)

---

### Experiment: beta3_5.0

**Status**: Success

**Configuration**:
- eta_values: [0.01]
- beta1: 0.9
- beta2: 0.999
- beta3: 5.0
- num_steps: 100
- seed: 42

**Results**: See plots in the [plots_beta3_5.0](plots_beta3_5.0) directory

**Available plots**:

- [eos_all_etas_beta1_0.9_beta3_5.0.png](plots_beta3_5.0/eos_all_etas_beta1_0.9_beta3_5.0.png)
- [eos_ratio_beta1_0.9_beta3_5.0.png](plots_beta3_5.0/eos_ratio_beta1_0.9_beta3_5.0.png)
- [eos_eta_0.01_beta1_0.9_beta3_5.0.png](plots_beta3_5.0/eos_eta_0.01_beta1_0.9_beta3_5.0.png)

---

### Experiment: beta1_0.7

**Status**: Success

**Configuration**:
- eta_values: [0.01]
- beta1: 0.7
- beta2: 0.999
- beta3: 1.0
- num_steps: 100
- seed: 42

**Results**: See plots in the [plots_beta1_0.7](plots_beta1_0.7) directory

**Available plots**:

- [eos_eta_0.01_beta1_0.7_beta3_1.0.png](plots_beta1_0.7/eos_eta_0.01_beta1_0.7_beta3_1.0.png)
- [eos_all_etas_beta1_0.7_beta3_1.0.png](plots_beta1_0.7/eos_all_etas_beta1_0.7_beta3_1.0.png)
- [eos_ratio_beta1_0.7_beta3_1.0.png](plots_beta1_0.7/eos_ratio_beta1_0.7_beta3_1.0.png)

---

### Experiment: beta1_0.8

**Status**: Success

**Configuration**:
- eta_values: [0.01]
- beta1: 0.8
- beta2: 0.999
- beta3: 1.0
- num_steps: 100
- seed: 42

**Results**: See plots in the [plots_beta1_0.8](plots_beta1_0.8) directory

**Available plots**:

- [eos_eta_0.01_beta1_0.8_beta3_1.0.png](plots_beta1_0.8/eos_eta_0.01_beta1_0.8_beta3_1.0.png)
- [eos_all_etas_beta1_0.8_beta3_1.0.png](plots_beta1_0.8/eos_all_etas_beta1_0.8_beta3_1.0.png)
- [eos_ratio_beta1_0.8_beta3_1.0.png](plots_beta1_0.8/eos_ratio_beta1_0.8_beta3_1.0.png)

---

### Experiment: beta1_0.9

**Status**: Success

**Configuration**:
- eta_values: [0.01]
- beta1: 0.9
- beta2: 0.999
- beta3: 1.0
- num_steps: 100
- seed: 42

**Results**: See plots in the [plots_beta1_0.9](plots_beta1_0.9) directory

**Available plots**:

- [eos_ratio_beta1_0.9_beta3_1.0.png](plots_beta1_0.9/eos_ratio_beta1_0.9_beta3_1.0.png)
- [eos_eta_0.01_beta1_0.9_beta3_1.0.png](plots_beta1_0.9/eos_eta_0.01_beta1_0.9_beta3_1.0.png)
- [eos_all_etas_beta1_0.9_beta3_1.0.png](plots_beta1_0.9/eos_all_etas_beta1_0.9_beta3_1.0.png)

---

### Experiment: beta1_0.95

**Status**: Success

**Configuration**:
- eta_values: [0.01]
- beta1: 0.95
- beta2: 0.999
- beta3: 1.0
- num_steps: 100
- seed: 42

**Results**: See plots in the [plots_beta1_0.95](plots_beta1_0.95) directory

**Available plots**:

- [eos_ratio_beta1_0.95_beta3_1.0.png](plots_beta1_0.95/eos_ratio_beta1_0.95_beta3_1.0.png)
- [eos_eta_0.01_beta1_0.95_beta3_1.0.png](plots_beta1_0.95/eos_eta_0.01_beta1_0.95_beta3_1.0.png)
- [eos_all_etas_beta1_0.95_beta3_1.0.png](plots_beta1_0.95/eos_all_etas_beta1_0.95_beta3_1.0.png)

---

### Experiment: normgrad_true

**Status**: Failed

**Configuration**:
- eta_values: [0.01]
- beta1: 0.9
- beta2: 0.999
- beta3: 1.0
- num_steps: 100
- seed: 42
- normgrad: True

**Results**: See plots in the [plots_normgrad_true](plots_normgrad_true) directory


---

### Experiment: normgrad_false

**Status**: Success

**Configuration**:
- eta_values: [0.01]
- beta1: 0.9
- beta2: 0.999
- beta3: 1.0
- num_steps: 100
- seed: 42
- normgrad: False

**Results**: See plots in the [plots_normgrad_false](plots_normgrad_false) directory

**Available plots**:

- [eos_ratio_beta1_0.9_beta3_1.0.png](plots_normgrad_false/eos_ratio_beta1_0.9_beta3_1.0.png)
- [eos_eta_0.01_beta1_0.9_beta3_1.0.png](plots_normgrad_false/eos_eta_0.01_beta1_0.9_beta3_1.0.png)
- [eos_all_etas_beta1_0.9_beta3_1.0.png](plots_normgrad_false/eos_all_etas_beta1_0.9_beta3_1.0.png)

---

### Experiment: long_run

**Status**: Success

**Configuration**:
- eta_values: [0.01]
- beta1: 0.9
- beta2: 0.999
- beta3: 1.0
- num_steps: 500
- seed: 42

**Results**: See plots in the [plots_long_run](plots_long_run) directory

**Available plots**:

- [eos_ratio_beta1_0.9_beta3_1.0.png](plots_long_run/eos_ratio_beta1_0.9_beta3_1.0.png)
- [eos_eta_0.01_beta1_0.9_beta3_1.0.png](plots_long_run/eos_eta_0.01_beta1_0.9_beta3_1.0.png)
- [eos_all_etas_beta1_0.9_beta3_1.0.png](plots_long_run/eos_all_etas_beta1_0.9_beta3_1.0.png)

---

