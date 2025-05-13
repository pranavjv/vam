# Edge of Stability Experiments Report

Generated on: 2025-05-13 13:59:47

## Experiment Summary

| Experiment ID | Status | Beta1 | Beta3 | Eta Values | Steps |
|--------------|--------|-------|-------|------------|-------|
| basic | ❌ Failed | 0.9 | 1.0 | [0.001, 0.005, 0.01, 0.05] | 100 |
| beta3_0.5 | ❌ Failed | 0.9 | 0.5 | [0.01] | 100 |
| beta3_1.0 | ❌ Failed | 0.9 | 1.0 | [0.01] | 100 |
| beta3_2.0 | ❌ Failed | 0.9 | 2.0 | [0.01] | 100 |
| beta3_5.0 | ❌ Failed | 0.9 | 5.0 | [0.01] | 100 |
| beta1_0.7 | ❌ Failed | 0.7 | 1.0 | [0.01] | 100 |
| beta1_0.8 | ❌ Failed | 0.8 | 1.0 | [0.01] | 100 |
| beta1_0.9 | ❌ Failed | 0.9 | 1.0 | [0.01] | 100 |
| beta1_0.95 | ❌ Failed | 0.95 | 1.0 | [0.01] | 100 |
| normgrad_true | ❌ Failed | 0.9 | 1.0 | [0.01] | 100 |
| normgrad_false | ❌ Failed | 0.9 | 1.0 | [0.01] | 100 |
| long_run | ❌ Failed | 0.9 | 1.0 | [0.01] | 500 |

## Experiment Details

### Experiment: basic

**Status**: Failed

**Configuration**:
- eta_values: [0.001, 0.005, 0.01, 0.05]
- beta1: 0.9
- beta2: 0.999
- beta3: 1.0
- num_steps: 100
- seed: 42

**Results**: See plots in the `plots` directory

---

### Experiment: beta3_0.5

**Status**: Failed

**Configuration**:
- eta_values: [0.01]
- beta1: 0.9
- beta2: 0.999
- beta3: 0.5
- num_steps: 100
- seed: 42

**Results**: See plots in the `plots` directory

---

### Experiment: beta3_1.0

**Status**: Failed

**Configuration**:
- eta_values: [0.01]
- beta1: 0.9
- beta2: 0.999
- beta3: 1.0
- num_steps: 100
- seed: 42

**Results**: See plots in the `plots` directory

---

### Experiment: beta3_2.0

**Status**: Failed

**Configuration**:
- eta_values: [0.01]
- beta1: 0.9
- beta2: 0.999
- beta3: 2.0
- num_steps: 100
- seed: 42

**Results**: See plots in the `plots` directory

---

### Experiment: beta3_5.0

**Status**: Failed

**Configuration**:
- eta_values: [0.01]
- beta1: 0.9
- beta2: 0.999
- beta3: 5.0
- num_steps: 100
- seed: 42

**Results**: See plots in the `plots` directory

---

### Experiment: beta1_0.7

**Status**: Failed

**Configuration**:
- eta_values: [0.01]
- beta1: 0.7
- beta2: 0.999
- beta3: 1.0
- num_steps: 100
- seed: 42

**Results**: See plots in the `plots` directory

---

### Experiment: beta1_0.8

**Status**: Failed

**Configuration**:
- eta_values: [0.01]
- beta1: 0.8
- beta2: 0.999
- beta3: 1.0
- num_steps: 100
- seed: 42

**Results**: See plots in the `plots` directory

---

### Experiment: beta1_0.9

**Status**: Failed

**Configuration**:
- eta_values: [0.01]
- beta1: 0.9
- beta2: 0.999
- beta3: 1.0
- num_steps: 100
- seed: 42

**Results**: See plots in the `plots` directory

---

### Experiment: beta1_0.95

**Status**: Failed

**Configuration**:
- eta_values: [0.01]
- beta1: 0.95
- beta2: 0.999
- beta3: 1.0
- num_steps: 100
- seed: 42

**Results**: See plots in the `plots` directory

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

**Results**: See plots in the `plots` directory

---

### Experiment: normgrad_false

**Status**: Failed

**Configuration**:
- eta_values: [0.01]
- beta1: 0.9
- beta2: 0.999
- beta3: 1.0
- num_steps: 100
- seed: 42
- normgrad: False

**Results**: See plots in the `plots` directory

---

### Experiment: long_run

**Status**: Failed

**Configuration**:
- eta_values: [0.01]
- beta1: 0.9
- beta2: 0.999
- beta3: 1.0
- num_steps: 500
- seed: 42

**Results**: See plots in the `plots` directory

---

