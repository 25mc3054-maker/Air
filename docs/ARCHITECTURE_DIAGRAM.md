# ATMOSAIR — NEURAL ARCHITECTURE SPECIFICATION & DIAGRAM

> **Model Class:** `CoupledMultiBranchForecastModel`  
> **Total Parameters:** `819,874` parameters (100% trainable)  
> **Input Shape:** `[Batch, 72, 49]` (72 historical hours, 49 features)  
> **Output Shape:** `[Batch, 72, 1]` (72 future PM2.5 forecast hours)  

---

## 1. High-Level System Flow Diagram

```mermaid
flowchart TD
    subgraph INPUT ["Input Historical Window [B, 72, 49]"]
        A["49 Input Features (Pollution, Met, Satellite, Cycles, Avail Masks)"]
    end

    subgraph PREPROC ["Preprocessing & Feature Splitting"]
        B1["Pollution Features (9 cols)"]
        B2["Meteorology Features (21 cols)"]
        B3["Atmospheric External Features (9 cols)"]
        B4["Temporal & Availability Features (10 cols)"]
    end

    subgraph BRANCHES ["Domain-Specific Causal Convolutional Streams"]
        C1["Pollution Branch<br/>2 Causal Conv Blocks (64 ch, dilations [1, 2])"]
        C2["Meteorology Branch<br/>2 Causal Conv Blocks (64 ch, dilations [1, 2])"]
        C3["Atmospheric External Branch<br/>2 Causal Conv Blocks (64 ch, dilations [1, 2])"]
        C4["Temporal & Availability Branch<br/>1 Causal Conv Block (32 ch)"]
    end

    subgraph FUSION ["Feature Coupling & Attention"]
        D["Branch Concatenation [B, 72, 224]"]
        E["Gated Linear Unit (GLU, 256 dim)"]
        F["4-Head Causal Multi-Head Self-Attention (d_model=256)"]
    end

    subgraph HEADS ["Dual Parallel Output Projection"]
        G1["Primary Forecast Head<br/>Linear Linear(256 -> 1) [B, 72, 1]"]
        G2["Auxiliary Spike Head<br/>Linear Linear(256 -> 1) + Sigmoid [B, 72, 1]"]
    end

    subgraph OUTPUT ["Production Output Format"]
        H1["Physical PM2.5 Trajectory (µg/m³)"]
        H2["Spike Probability Index (≥ 345 µg/m³)"]
    end

    A --> B1 & B2 & B3 & B4
    B1 --> C1
    B2 --> C2
    B3 --> C3
    B4 --> C4
    C1 & C2 & C3 & C4 --> D
    D --> E
    E --> F
    F --> G1 & G2
    G1 --> H1
    G2 --> H2
```

---

## 2. Text Architectural Summary

```
Input Sequence: [B, 72, 49]
  ├── Pollution Branch    : [B, 72,  9] ──> Conv1D Blocks ──> [B, 72, 64]
  ├── Meteorology Branch  : [B, 72, 21] ──> Conv1D Blocks ──> [B, 72, 64]
  ├── External Sat Branch : [B, 72,  9] ──> Conv1D Blocks ──> [B, 72, 64]
  └── Temporal Branch     : [B, 72, 10] ──> Conv1D Block  ──> [B, 72, 32]
                                                                  │
                                 Concatenation ───────────────────┘ [B, 72, 224]
                                      │
                         Gated Linear Unit (GLU) ─────────────────> [B, 72, 256]
                                      │
                     4-Head Causal Self-Attention ────────────────> [B, 72, 256]
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
       Primary Forecast Head                     Auxiliary Spike Head
        Linear Projection                         Linear + Sigmoid
           [B, 72, 1]                                [B, 72, 1]
```
