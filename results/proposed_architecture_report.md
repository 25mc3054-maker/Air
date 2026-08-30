# B10 — PROPOSED ARCHITECTURE SPECIFICATION & MODEL DESIGN REPORT

**Project:** Deep Learning Air Quality Forecasting (72h-to-72h PM2.5)  
**Proposed Model:** Coupled Multi-Branch Forecast Model (`CoupledMultiBranchForecastModel`)  
**Status:** `B10 STATUS: COMPLETED & VERIFIED`  
**Next Stage:** `READY FOR B11 — PROPOSED MODEL IMPLEMENTATION & SANITY CHECK`

---

## 1. Executive Summary

Based on the empirical findings from the baseline benchmarking suite (B6 GRU, B7 LSTM, B8 TCN, and B9 Leaderboard Analysis), we present the formal architectural specification for the **Coupled Multi-Branch Forecast Model**.

Rather than increasing network depth uniform-style, the proposed architecture specifically targets the physical bottlenecks and failure modes identified during baseline benchmarking:
1. **Domain-Specific Multi-Branch Encoding:** Processes Pollution (9), Meteorology (21), Atmospheric External (9), and Temporal/Availability (10) feature groups through dedicated causal dilated convolutional branches.
2. **Gated Feature Fusion & Causal Self-Attention:** Combines cross-domain representations via Gated Linear Units (GLU) and 4-head causal self-attention, capturing multi-day periodicity across the 72-hour historical window.
3. **Spike-Aware Loss & Dual-Head Mechanism:** Incorporates an auxiliary spike classification head and asymmetric spike-weighted MSE loss to counteract the variance compression ($\sigma_p/\sigma_y \approx 0.72$) and spike underprediction observed in all single-stream baseline models.
4. **Target Parameter Budget:** **`819,874` parameters** (falling strictly within the 0.8M – 2.0M parameter budget limit).

---

## 2. Empirical Justifications from Baseline Benchmarking (B6–B9)

| Empirical Finding from B6–B9 Benchmark | Baseline Limitation / Failure Mode | Proposed Architectural Solution | Justification |
| :--- | :--- | :--- | :--- |
| **TCN Champion Supremacy ($R^2 = 0.2053$ vs GRU $-0.1667$)** | RNN hidden states suffer from vanishing temporal gradients over 72 steps. | Dilated causal convolutions across all domain encoders. | Receptive field extension (127 hours) captures multi-day temporal dependencies without gradient decay. |
| **Feature Group Heterogeneity** | Single-stream models treat all 49 features uniformly despite distinct physical interaction scales. | 4-Branch Domain Feature Encoding (Pollution, Meteorology, Atmospheric, Temporal). | Dedicated feature branches prevent dominant meteorology variables from washing out subtle chemical precursor signals. |
| **Short-Term & Diurnal Autocorrelation** | Persistence wins at +1h ($21.02 \mu g/m^3$) and 24h diurnal cycles ($56.43 \mu g/m^3$). | Causal Multi-Head Self-Attention + Gated Linear Unit (GLU) fusion. | Self-attention directly routes historical diurnal states ($t - 24h, t - 48h$) to future forecast timesteps. |
| **Extreme Pollution Spike Underprediction** | Standard MSE loss causes variance compression ($\sigma_p/\sigma_y = 0.72$) and underpredicts spikes ($\ge 345 \mu g/m^3$ MAE = $223 \mu g/m^3$). | Dual-head output (Forecast Head + Auxiliary Spike Head) & Spike-Weighted MSE. | Asymmetric loss weighting forces higher penalty on extreme spike errors during training while maintaining 100% standard inference interface. |

---

## 3. Comprehensive Model Architecture Specification

The model takes input sequence $X \in \mathbb{R}^{B \times 72 \times 49}$ and outputs primary forecast $\hat{y} \in \mathbb{R}^{B \times 72 \times 1}$.

```
               [ Input Tensor: X in R^{B x 72 x 49} ]
                                 |
         +-----------------------+-----------------------+
         |                       |                       |
 [ Pollution (9) ]     [ Meteorology (21) ]    [ Atmospheric (9) ]    [ Temporal (10) ]
 Causal Conv (64 ch)   Causal Conv (64 ch)     Causal Conv (64 ch)    Causal Conv (32 ch)
         |                       |                       |                       |
         +-----------------------+-----------------------+-----------------------+
                                 |
              [ Concatenation: 224 Channels ]
                                 |
              [ Gated Feature Fusion (GLU): 256 Dim ]
                                 |
              [ Causal Multi-Head Self-Attention (4 Heads) ]
                                 |
              [ LayerNorm & Feed-Forward Network (512 Dim) ]
                                 |
         +-----------------------+-----------------------+
         |                                               |
 [ Primary Forecast Head ]                       [ Auxiliary Spike Head ]
 Conv1D(256 -> 128 -> 1)                         Conv1D(256 -> 64 -> 1) + Sigmoid
 Output: [B, 72, 1] PM2.5                         Output: [B, 72, 1] Pr[Spike >= 345]
```

### Module Breakdown & Parameter Budget

| Component / Module | Architecture Details | Parameters | Parameter Share (%) |
| :--- | :--- | :---: | :---: |
| **Branch 1: Pollution** | 2 Dilated Causal Conv Blocks ($9 \to 64 \to 64$ ch, $k=3, d=1,2$) | `39,296` | `4.79%` |
| **Branch 2: Meteorology** | 2 Dilated Causal Conv Blocks ($21 \to 64 \to 64$ ch, $k=3, d=1,2$) | `42,368` | `5.17%` |
| **Branch 3: Atmospheric** | 2 Dilated Causal Conv Blocks ($9 \to 64 \to 64$ ch, $k=3, d=1,2$) | `39,296` | `4.79%` |
| **Branch 4: Temporal** | 1 Causal Conv Block ($10 \to 32$ ch, $k=3, d=1$) | `4,864` | `0.59%` |
| **Gated Feature Fusion** | GLU Value & Gate Convolutions ($224 \to 256$ ch $\times 2$) | `115,200` | `14.05%` |
| **Causal Self-Attention** | 4-Head Self-Attention ($d_{\text{model}}=256$, LayerNorm) | `263,168` | `32.10%` |
| **Feed-Forward Network** | Linear ($256 \to 512 \to 256$), LayerNorm | `263,680` | `32.16%` |
| **Primary Forecast Head** | Conv1D ($256 \to 128 \to 1$), BatchNorm | `33,153` | `4.04%` |
| **Auxiliary Spike Head** | Conv1D ($256 \to 64 \to 1$), BatchNorm, Sigmoid | `18,849` | `2.30%` |
| **Total Proposed Model** | `CoupledMultiBranchForecastModel` | **`819,874`** | **`100.00%`** |

---

## 4. Parameter Budget Comparison Against Baselines

| Model Architecture | Parameters | Parameter Budget Status | Parameter Ratio vs Proposed |
| :--- | :---: | :---: | :---: |
| **GRU Baseline (B6)** | `167,937` | Baseline | `4.88x` capacity |
| **LSTM Baseline (B7)** | `223,873` | Baseline | `3.66x` capacity |
| **TCN Baseline (B8)** | `570,625` | Baseline Champion | `1.44x` capacity |
| **Proposed Architecture (B10)** | **`819,874`** | **APPROVED (Target: 0.8M – 2.0M)** | **`1.00x` (Target Budget)** |

---

## 5. Spike-Aware Loss Strategy & Auxiliary Head Design

To resolve the extreme pollution spike underprediction ($\ge 345.00 \mu g/m^3$), the proposed model utilizes a **Spike-Weighted Loss Function** during training:

$$L_{\text{total}} = L_{\text{MSE}}(y, \hat{y}) + \alpha \cdot w_{\text{spike}} \odot L_{\text{MSE}}(y, \hat{y}) + \beta \cdot L_{\text{BCE}}(y_{\text{binary\_spike}}, p_{\text{spike}})$$

Where:
* $w_{\text{spike}} = 3.0$ for timesteps where ground truth $y_{\text{true}} \ge 345.00 \mu g/m^3$.
* $\alpha = 0.5$, $\beta = 0.2$.
* $p_{\text{spike}}$ is the auxiliary head probability output.
* **Inference Efficiency:** During final test evaluation, only the primary forecast output $\hat{y}$ is used, incurring zero additional computational overhead!

---

## 6. Controlled Ablation Plan (B11–B14 Roadmap)

| Ablation Stage | Model Configuration | Core Hypothesis | Changed Component |
| :---: | :--- | :--- | :--- |
| **Ablation A** | TCN Champion Baseline (B8) | Receptive field alone provides baseline performance. | Benchmark |
| **Ablation B** | Multi-Branch TCN (No Attention/Spike) | Domain branch separation improves feature representation. | 4-branch encoders + Concatenation |
| **Ablation C** | Multi-Branch + Gated Attention | Causal self-attention captures long-range diurnal patterns. | Add GLU + Multi-Head Self-Attention |
| **Ablation D** | Full Proposed Architecture + Spike Head | Auxiliary spike loss eliminates extreme underprediction bias. | Add Dual-Head + Spike-Weighted Loss |

---

## 7. Architecture Validation Suite Results (12/12 Checks Passed)

The specification was programmatically verified via [`tests/test_proposed_architecture.py`](file:///d:/My%20Projects/SIH2026_PersonB/tests/test_proposed_architecture.py):

- [x] **Check 1 & 2:** Model Import & Instantiation (`PASS`)
- [x] **Check 3, 4, 6:** Input `[B, 72, 49]` $\to$ Output `[B, 72, 1]` Float32 (`PASS`)
- [x] **Check 5:** Batch-Size Flexibility ($B=1, 16, 64$) (`PASS`)
- [x] **Check 7 & 8:** Zero NaNs & Zero Infs (`PASS`)
- [x] **Check 9:** Parameter Count Verification (`819,874` params $\in [800k, 2M]$) (`PASS`)
- [x] **Check 10:** Strict Causal Behavior Test (Max past diff $= 0.0000000000$) (`PASS`)
- [x] **Check 11:** Deterministic Forward Pass under Seed 42 (`PASS`)
- [x] **Check 12:** Compatibility with Verified B5 DataLoaders (`PASS`)

---

## 8. Deliverable Verification Index

* **Machine-Readable Config:** [`configs/proposed_architecture.json`](file:///d:/My%20Projects/SIH2026_PersonB/configs/proposed_architecture.json)
* **Parameter Budget Breakdown:** [`results/proposed_architecture_parameter_budget.json`](file:///d:/My%20Projects/SIH2026_PersonB/results/proposed_architecture_parameter_budget.json)
* **Architecture Diagram:** [`results/proposed_architecture_diagram.png`](file:///d:/My%20Projects/SIH2026_PersonB/results/proposed_architecture_diagram.png)
* **PyTorch Model Code:** [`models/proposed_model.py`](file:///d:/My%20Projects/SIH2026_PersonB/models/proposed_model.py)
* **Validation Test Suite:** [`tests/test_proposed_architecture.py`](file:///d:/My%20Projects/SIH2026_PersonB/tests/test_proposed_architecture.py)

---

```
B10 STATUS: COMPLETED & VERIFIED
```

```
READY FOR B11 — PROPOSED MODEL IMPLEMENTATION / SANITY CHECK
```
