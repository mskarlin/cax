# Arithmetic Intensity Analysis: NCA vs Transformer Inference

This document summarizes the arithmetic intensity (AI) analysis for Neural Cellular Automata (NCA) architectures, comparing them to transformer-based models. The goal is to inform implementation strategies for distilling transformer models to NCAs with superior inference efficiency.

## Executive Summary

| Workload | AI (FLOPs/byte) | Bound | Notes |
|----------|-----------------|-------|-------|
| Transformer decode (autoregressive) | ~1 | Memory | KV cache bottleneck |
| NCA naive (per-step HBM) | ~17 | Memory | Repeated state R/W |
| A100 HBM threshold | ~156 | — | 312 TFLOPs / 2.0 TB/s |
| A100 L2 threshold | ~62 | — | 312 TFLOPs / 5.0 TB/s |
| **NCA tiled (vs HBM)** | **~306** | **Compute** | State I/O to HBM |
| **NCA tiled (vs L2)** | **~253** | **Compute** | Weight streaming from L2 |
| Flash Attention prefill (S=4K) | ~2,048 | Compute | O(S²) scaling |

**Key insight**: Autoregressive transformer decoding (~1 FLOPs/byte) is significantly more memory-bound than even naive NCA (~17 FLOPs/byte). With multi-step fusion, NCAs achieve ~306 FLOPs/byte (vs HBM) and ~253 FLOPs/byte (vs L2), exceeding *both* thresholds and making them solidly compute-bound.

**Memory hierarchy note**: The fusion strategy uses a hybrid approach—tile state in SRAM, weights streaming from L2 cache. Because L2 has higher bandwidth (~5 TB/s vs 2 TB/s), its compute-bound threshold is *lower* (~62 vs ~156), making L2 weight streaming easier to hide, not harder.

---

## 1. NCA Architecture Overview

Reference architecture from `examples/45_self_autoencoding_mnist.ipynb`:

### State Representation
- **Shape**: `(H, W, D, C)` = `(28, 28, 42, 16)`
- **Spatial dims**: H×W for image, D for depth/information flow
- **Channels**: 16 per cell

### Per-Step Operations

```
State (28, 28, 42, 16)
        ↓
    3×3×3 depthwise conv (perception)
        ↓
Perception (28, 28, 42, 64)    ← 4 kernels × 16 channels
        ↓
    1×1×1 conv + ReLU (MLP hidden)
        ↓
Hidden (28, 28, 42, 256)
        ↓
    1×1×1 conv (MLP output)
        ↓
Update (28, 28, 42, 16)
        ↓
    residual: state += update
```

### Perception Kernels
The 4 perception kernels per channel are typically:
- 1 identity kernel (cell's own value)
- 3 Sobel-style gradient kernels (∂/∂x, ∂/∂y, ∂/∂z)

These detect local change/edges, giving cells information about their neighborhood.

### Key Design Properties
- **3×3×3 conv**: Only place where neighboring cells communicate
- **1×1×1 convs (MLP)**: Same weights applied independently at every cell
- **Depthwise perception**: Each channel processed separately
- **All cells update simultaneously** each step

---

## 2. Arithmetic Intensity Analysis

### Definition
```
Arithmetic Intensity (AI) = FLOPs / Bytes transferred to/from memory
```

Higher AI = more compute-bound (good for GPU utilization)
Lower AI = more memory-bound (bottlenecked by bandwidth)

### Per-Cell FLOPs (Single Step)

| Operation | Calculation | FLOPs |
|-----------|-------------|-------|
| Perception (depthwise 3³) | 64 outputs × 27 MACs × 2 | 3,456 |
| MLP Layer 1 (64→256) | 64 × 256 × 2 | 32,768 |
| MLP Layer 2 (256→16) | 256 × 16 × 2 | 8,192 |
| **Total per cell** | | **~44 KFLOPs** |

### Model Weights

| Layer | Shape | Size |
|-------|-------|------|
| Perception kernel | 3×3×3×16×4 | 6.9 KB |
| MLP Layer 1 | 64×256 | 65.5 KB |
| MLP Layer 2 | 256×16 | 16.4 KB |
| **Total** | | **~89 KB** |

### Naive Per-Step AI (State: 28×28×42×16)

| Operation | FLOPs | Memory | AI |
|-----------|-------|--------|-----|
| Perception | 114M | 10.5 MB | ~11 |
| MLP Layer 1 | 1.08G | 42 MB | ~26 |
| MLP Layer 2 | 270M | 36 MB | ~7.5 |
| **Full step** | **~1.5G** | **~90 MB** | **~17** |

**Problem**: Each step reads/writes full state to HBM. With 96 steps, this is ~8.6 GB of memory traffic.

---

## 3. Parametric AI Analysis

This section derives arithmetic intensity formulas in terms of architectural parameters, showing how AI scales with design choices.

### Parameter Definitions

| Symbol | Description | Example Value |
|--------|-------------|---------------|
| `V` | Total cells (spatial volume) | H × W × D = 28 × 28 × 42 = 32,928 |
| `C` | Channels per cell | 16 |
| `K` | Kernel size (assume cubic K³) | 3 |
| `R` | Num perception kernels per channel | 4 (identity + 3 gradients) |
| `P` | Perception size = R × C | 64 |
| `M` | MLP hidden size | 256 |
| `N` | Fused steps | 4 |
| `T` | Tile interior size (assume cubic T³) | 8 |
| `B` | Bytes per element | 4 (FP32) or 2 (FP16/BF16) |

### Per-Cell FLOPs (Single Step)

| Operation | Formula | Simplifies To |
|-----------|---------|---------------|
| Perception (depthwise K³) | P × K³ × 2 | 2 × R × C × K³ |
| MLP Layer 1 (P → M) | P × M × 2 | 2 × R × C × M |
| MLP Layer 2 (M → C) | M × C × 2 | 2 × M × C |
| **Total per cell** | | **2C(RK³ + RM + M)** |

Factored form:
```
FLOPs_cell = 2C(R(K³ + M) + M)
           = 2C·M(R + 1) + 2C·R·K³
```

For typical NCAs where M >> K³, the MLP dominates:
```
FLOPs_cell ≈ 2C·M·(R + 1)
```

### Weight Memory

| Layer | Formula | Size |
|-------|---------|------|
| Perception | K³ × C × R × B | K³CRB |
| MLP Layer 1 | P × M × B = RCM × B | RCMB |
| MLP Layer 2 | M × C × B | MCB |
| **Total** | | **CB(K³R + RM + M)** |

Factored:
```
Weights = CB(R(K³ + M) + M)
        = CB·M·(R + 1) + CB·R·K³
```

Note: `Weights = FLOPs_cell × B / 2` — they have the same structure!

### Naive Per-Step AI (Full Grid)

**Compute:**
```
FLOPs_step = V × FLOPs_cell = 2VC(R(K³ + M) + M)
```

**Memory (assuming each operation reads/writes full tensors):**

| Operation | Read | Write |
|-----------|------|-------|
| Perception | VC·B (state) | VP·B (perception) |
| MLP Layer 1 | VP·B | VM·B (hidden) |
| MLP Layer 2 | VM·B | VC·B (update) |

```
Memory_step = B(VC + VP + VP + VM + VM + VC)
            = B(2VC + 2VP + 2VM)
            = 2VB(C + P + M)
            = 2VB(C + RC + M)
            = 2VB(C(1 + R) + M)
```

**Naive AI:**
```
AI_naive = FLOPs_step / Memory_step
         = 2VC(R(K³ + M) + M) / (2VB(C(1 + R) + M))
         = C(R(K³ + M) + M) / (B(C(1 + R) + M))
```

When M >> K³ and M >> C:
```
AI_naive ≈ C(RM + M) / (B·M)
         = C(R + 1) / B
```

**Key insight**: Naive AI scales with **C/B** and **(R+1)**, independent of spatial size V!

| Precision | C=16, R=4 | C=32, R=4 | C=64, R=4 |
|-----------|-----------|-----------|-----------|
| FP32 (B=4) | 20 | 40 | 80 |
| FP16 (B=2) | 40 | 80 | 160 |

### Tiled Multi-Step Fusion AI

With spatial tiling (interior T³) and N fused steps:

**Halo size**: Each step requires 1-cell halo for K=3 conv. N steps require N-cell halo.

**Tile dimensions:**
```
Interior: T³
With halo: (T + 2N)³
```

**Memory per tile:**
```
Read tile+halo:  (T + 2N)³ × C × B
Write interior:  T³ × C × B
Weights:         CB(R(K³ + M) + M)  [reused across tiles, amortized]
```

If weights cached (amortized across many tiles):
```
Memory_tile ≈ CB((T + 2N)³ + T³)
```

**Compute per tile:**
```
FLOPs_tile = N × T³ × FLOPs_cell
           = 2N·T³·C(R(K³ + M) + M)
```

**Tiled Fusion AI:**
```
AI_tiled = FLOPs_tile / Memory_tile
         = 2N·T³·C(R(K³ + M) + M) / (CB((T + 2N)³ + T³))
         = 2N·T³(R(K³ + M) + M) / (B((T + 2N)³ + T³))
```

When M >> K³:
```
AI_tiled ≈ 2N·T³·M(R + 1) / (B((T + 2N)³ + T³))
```

**Scaling insights:**
- AI scales **linearly with N** (fused steps) — main lever!
- AI scales with **T³ / ((T + 2N)³ + T³)** — tile efficiency
- AI scales with **M(R+1) / B** — model size and precision

### Tile Efficiency Factor

Define tile efficiency η:
```
η = T³ / ((T + 2N)³ + T³)
```

| T | N | (T+2N)³ | T³ | η |
|---|---|---------|-----|---|
| 8 | 1 | 1,000 | 512 | 0.34 |
| 8 | 2 | 1,728 | 512 | 0.23 |
| 8 | 4 | 4,096 | 512 | 0.11 |
| 16 | 4 | 13,824 | 4,096 | 0.23 |
| 32 | 4 | 64,000 | 32,768 | 0.34 |

Larger tiles improve efficiency but require more SRAM.

### AI Scaling Summary

```
AI_tiled ≈ 2N × η × M(R + 1) / B
```

| Factor | Effect | How to Improve |
|--------|--------|----------------|
| N (fused steps) | Linear ↑ | Fuse more steps (limited by SRAM) |
| η (tile efficiency) | Sub-linear ↑ | Larger tiles (limited by SRAM) |
| M (hidden size) | Linear ↑ | Wider MLP (more params) |
| R (perception kernels) | Linear ↑ | More kernels (more params) |
| B (bytes) | Linear ↓ | Use FP16/BF16 |

### Example Calculations

**Config 1: Reference (C=16, M=256, R=4, T=8, N=4, FP32)**
```
η = 512 / (4096 + 512) = 0.11
AI ≈ 2 × 4 × 0.11 × 256 × 5 / 4 = 282 FLOPs/byte ✓
```

**Config 2: Wider MLP (C=16, M=512, R=4, T=8, N=4, FP32)**
```
AI ≈ 2 × 4 × 0.11 × 512 × 5 / 4 = 563 FLOPs/byte ✓✓
```

**Config 3: FP16 (C=16, M=256, R=4, T=8, N=4, FP16)**
```
AI ≈ 2 × 4 × 0.11 × 256 × 5 / 2 = 563 FLOPs/byte ✓✓
```

**Config 4: More channels (C=32, M=256, R=4, T=8, N=4, FP32)**
```
Note: AI formula doesn't depend on C directly (cancels out)!
But larger C means larger state → may need smaller T → lower η
```

**Config 5: Larger tiles (C=16, M=256, R=4, T=16, N=4, FP32)**
```
η = 4096 / (13824 + 4096) = 0.23
AI ≈ 2 × 4 × 0.23 × 256 × 5 / 4 = 589 FLOPs/byte ✓✓
Requires: (T+2N)³ × C × B = 24³ × 16 × 4 = 884 KB SRAM ❌
```

### Comparison: Transformer Decode AI

For autoregressive decoding with sequence length S, head dim d:
```
FLOPs = 4 × S × d
Memory = 2 × S × d × B    (KV cache read)
AI_decode = 4Sd / (2SdB) = 2/B
```

| Precision | Transformer Decode AI |
|-----------|----------------------|
| FP32 | 0.5 FLOPs/byte |
| FP16 | 1 FLOPs/byte |

**NCA advantage ratio:**
```
AI_tiled / AI_decode = N × η × M(R + 1) × B / B
                     = N × η × M(R + 1)
```

For Config 1: `4 × 0.11 × 256 × 5 = 563×` better than transformer decode!

---

## 4. Multi-Step Fusion Strategy

### Core Idea
Keep state in on-chip memory (SRAM/registers) across multiple NCA steps:

```
Naive:     HBM → Step 1 → HBM → Step 2 → HBM → ... → HBM
Fused:     HBM → Step 1 → Step 2 → Step 3 → Step 4 → HBM
```

### Challenge: State Too Large for SRAM

| Memory Level | Capacity (A100) | Full State (28×28×42×16) |
|--------------|-----------------|--------------------------|
| Registers | ~256 KB/SM | 2.1 MB ❌ |
| Shared Memory | ~192 KB/SM | 2.1 MB ❌ |
| L2 Cache | 40 MB | 2.1 MB ✓ (contention) |

### Solution: Spatial Tiling with Halos

Tile the spatial domain so each tile fits in shared memory.

**Halo requirement**: 3×3×3 conv needs 1-cell halo per step. For N fused steps, need N-cell halo.

| Fused Steps | Halo | Tile+Halo (8³ interior) | Memory |
|-------------|------|-------------------------|--------|
| 1 | 1 | 10×10×10×16 | 64 KB |
| 4 | 4 | 16×16×16×16 | 262 KB |
| 8 | 8 | 24×24×24×16 | 884 KB ❌ |

**Sweet spot**: ~4 fused steps with 8³ interior tiles.

### Tiled Fusion AI Calculation

For 8³ interior tile, 4 fused steps:

**Memory:**
- Tile + halo read: 16×16×16×16 × 4 bytes = 262 KB
- Weights: 89 KB (reused across tiles, amortized)
- Output write: 8×8×8×16 × 4 bytes = 32 KB
- **Per-tile total: ~294-383 KB**

**Compute:**
- 4 steps × 44 KFLOPs × 8³ cells = 90 MFLOPs

**AI: 90M / 294K ≈ 235-306 FLOPs/byte**

This exceeds the A100 compute-bound threshold (~156 FLOPs/byte).

### SRAM Capacity Constraints and Realistic Memory Hierarchy

**Critical caveat**: The above AI calculation assumes weights are "amortized" across tiles. In reality, we must consider where data physically resides during fusion.

#### What Needs to Fit Together?

| Component | Size (FP32) | Size (BF16) |
|-----------|-------------|-------------|
| Tile + halo (16³×16) | 262 KB | 131 KB |
| Weights (perception + MLP) | 89 KB | 44.5 KB |
| Intermediate activations | ~64 KB | ~32 KB |
| **Total if all in SRAM** | **~415 KB** | **~208 KB** |
| A100 Shared Memory/SM | 192 KB | 192 KB |

**Problem**: Even with BF16, tile state + weights + activations exceed shared memory capacity.

#### Realistic Memory Hierarchy Strategy

The fusion actually works with a **hybrid L2/SRAM approach**:

```
┌─────────────────────────────────────────────────────────┐
│ HBM (2 TB/s)                                            │
│   └── Full state grid, initial read / final write       │
├─────────────────────────────────────────────────────────┤
│ L2 Cache (40 MB, ~5 TB/s effective)                     │
│   └── Weights (89 KB) - loaded once, reused all tiles   │
├─────────────────────────────────────────────────────────┤
│ Shared Memory (192 KB/SM)                               │
│   └── Tile + halo state during N fused steps            │
│   └── Streaming window for activations                  │
└─────────────────────────────────────────────────────────┘
```

**Key insight**: Weights stream from L2 cache (not SRAM) during computation. This works because:
1. Weights (89 KB) easily fit in L2 cache (40 MB)
2. L2→SM bandwidth (~5 TB/s) is ~2.5× faster than HBM→SM (2 TB/s)
3. Weights are accessed predictably, enabling effective prefetching

#### Revised AI Calculation (Checking Both Memory Levels)

For 8³ interior tile, 4 fused steps, with L2 weight streaming:

**Memory traffic:**
```
HBM traffic (state I/O):
  Read tile+halo:  262 KB
  Write interior:   32 KB
  Subtotal:        294 KB

L2 traffic (weights, per tile):
  4 steps × 89 KB = 356 KB
```

**Compute-bound analysis requires checking each memory level separately:**

Since L2 has higher bandwidth than HBM (~5 TB/s vs 2 TB/s on A100), its compute-bound threshold is *lower*:
```
AI_threshold_HBM = 312 TFLOPs / 2.0 TB/s = 156 FLOPs/byte
AI_threshold_L2  = 312 TFLOPs / 5.0 TB/s =  62 FLOPs/byte
```

**Per-memory-level AI:**
```
AI_HBM = 90 MFLOPs / 294 KB = 306 FLOPs/byte  > 156 ✓ compute-bound
AI_L2  = 90 MFLOPs / 356 KB = 253 FLOPs/byte  >  62 ✓ compute-bound
```

**Result**: We exceed the compute-bound threshold for *both* memory levels. The L2 weight streaming is not a bottleneck—L2's higher bandwidth makes it easier to saturate compute, not harder.

This is actually better than the idealized analysis suggested: we're solidly compute-bound even when properly accounting for the hybrid memory hierarchy.

#### When Does This Scheme Break Down?

| Scenario | Problem | Mitigation |
|----------|---------|------------|
| Larger MLP (M > 512) | Weights exceed L2 working set | Reduce tile size, accept lower η |
| Many channels (C > 32) | Tile state exceeds SRAM | Smaller tiles, fewer fused steps |
| More fused steps (N > 6) | Halo grows, tile efficiency drops | Cap at N=4-6 |
| L2 cache pressure | Weight eviction, HBM fallback | Dedicated L2 partitioning (if available) |

#### BF16 Enables Larger Tiles

With BF16 precision:
```
Tile + halo (16³×16, BF16):  131 KB  ← fits in 192 KB SRAM!
Weights in L2:                44.5 KB
Activations (streaming):     ~32 KB
```

BF16 makes the scheme more robust by:
1. Halving state memory → larger tiles possible
2. Halving weight traffic → less L2 pressure
3. Enabling potential 20³ tiles with 4-step fusion

---

## 5. Comparison with Transformer Attention

### Flash Attention - Prefill (Full Sequence)

Processing a prompt of length S with head dimension d:

```
FLOPs:  ~4 × S² × d     (QK^T and scores×V)
Memory: ~4 × S × d      (Q, K, V, O only - no materialized attention matrix)
AI:     S / bytes       (scales with sequence length!)
```

| Seq Length | AI (FP16) |
|------------|-----------|
| 1,024 | 512 |
| 4,096 | 2,048 |
| 8,192 | 4,096 |

Flash Attention prefill is extremely compute-bound due to O(S²) compute.

### Flash Attention - Autoregressive Decoding (Per Token)

```
FLOPs:  ~4 × S × d      (linear in sequence length)
Memory: ~2 × S × d      (must read entire KV cache)
AI:     2 / bytes ≈ 1 FLOPs/byte (FP16)
```

**Decoding is catastrophically memory-bound** - every token requires reading the full KV cache with only O(S) compute.

### Why NCA Has Structural Advantages for Inference

| Property | Transformer Decode | NCA (Fused) |
|----------|-------------------|-------------|
| Memory access pattern | Read full KV cache per token | Tiled, local |
| Compute scaling | O(S) per token | O(cells × steps) |
| State reuse | None (KV cache grows) | Multi-step fusion |
| AI | ~1 FLOPs/byte | ~235+ FLOPs/byte |

---

## 6. Implementation Strategy

### Phase 1: Quick Wins
1. **Mixed precision (BF16)**: Halves memory traffic → 2× effective AI
2. **JAX `lax.scan`**: Enable some XLA fusion optimizations

```python
@jax.jit
def multi_step(state, num_steps):
    def step_fn(state, _):
        perception = perceive(state)
        return update(state, perception), None
    return lax.scan(step_fn, state, None, length=num_steps)[0]

state = state.astype(jnp.bfloat16)
```

### Phase 2: Tiled Multi-Step Fusion (Pallas)

```python
import jax.experimental.pallas as pl

def nca_fused_kernel(state_ref, output_ref, weights_ref):
    # Load tile + halo into shared memory
    tile_idx = pl.program_id(0)
    local_state = pl.load(state_ref, tile_slice_with_halo)

    # Run N steps entirely in shared memory
    for step in range(N_FUSED_STEPS):
        perception = depthwise_conv_3x3x3(local_state, weights_ref)
        local_state = mlp_update(local_state, perception, weights_ref)

    # Write only interior back to HBM
    pl.store(output_ref, interior_slice, local_state[halo:-halo, ...])
```

### Phase 3: Advanced Optimizations
- **Overlapping compute/transfer**: Pipeline tile loading with computation
- **Adaptive tile sizes**: Optimize for specific hardware
- **Sparse updates**: Skip cells with minimal change

---

## 7. Implications for Transformer→NCA Distillation

### Why Distill to NCA?

1. **Inference efficiency**: ~235× better AI than transformer decoding
2. **Constant memory**: No growing KV cache
3. **Parallel generation**: All positions update simultaneously
4. **Hardware utilization**: Compute-bound on modern GPUs

### Distillation Considerations

1. **Sequence→Spatial mapping**: Map token positions to NCA spatial dimensions
2. **Step budget**: Trade NCA steps for accuracy (more steps = more compute, same memory)
3. **Channel capacity**: Sufficient channels to represent token embeddings
4. **Perception receptive field**: May need larger kernels or more steps for long-range dependencies

### Expected Trade-offs

| Aspect | Transformer | NCA |
|--------|-------------|-----|
| Long-range dependencies | Excellent (attention) | Limited (local conv) |
| Inference latency | High (sequential, memory-bound) | Low (parallel, compute-bound) |
| Memory scaling | O(S) KV cache | O(1) state |
| Training complexity | Standard | Distillation required |

---

## 8. Hardware Considerations

### Compute-Bound Thresholds by Memory Level

The compute-bound threshold depends on which memory level is the bottleneck:

```
AI_threshold = Peak Compute (FLOPs/s) / Memory Bandwidth (bytes/s)
```

| GPU | Peak TFLOPs (FP16) | HBM BW | HBM Threshold | L2 BW | L2 Threshold |
|-----|-------------------|--------|---------------|-------|--------------|
| A100 | 312 | 2.0 TB/s | ~156 | ~5 TB/s | ~62 |
| H100 | 990 | 3.35 TB/s | ~296 | ~12 TB/s | ~82 |
| V100 | 125 | 900 GB/s | ~139 | ~3 TB/s | ~42 |

**Key insight**: L2 cache has much higher bandwidth than HBM, so its compute-bound threshold is *lower* (easier to achieve).

### Are We Compute-Bound? (Checking Both Memory Levels)

For our fused NCA kernel, we must check compute-boundedness against each memory level separately:

**Per-tile metrics (8³ interior, 4 fused steps):**
```
Compute:     90 MFLOPs
HBM traffic: 294 KB (state read/write)
L2 traffic:  356 KB (weights × 4 steps)
```

**Arithmetic Intensity by memory level:**
```
AI_HBM = 90 MFLOPs / 294 KB = 306 FLOPs/byte
AI_L2  = 90 MFLOPs / 356 KB = 253 FLOPs/byte
```

**Compute-bound check:**

| GPU | AI_HBM vs Threshold | AI_L2 vs Threshold | Status |
|-----|--------------------|--------------------|--------|
| A100 | 306 > 156 ✓ | 253 > 62 ✓ | **Compute-bound** |
| H100 | 306 > 296 ✓ | 253 > 82 ✓ | **Compute-bound** (barely for HBM) |
| V100 | 306 > 139 ✓ | 253 > 42 ✓ | **Compute-bound** |

**Result**: We're compute-bound with respect to *both* memory levels on all GPUs. The L2 weight streaming is not a bottleneck because L2's higher bandwidth lowers its threshold significantly.

**H100 note**: On H100, we're only ~3% above the HBM threshold (306 vs 296). Larger tiles or more fused steps would provide more headroom.

### Memory Hierarchy Utilization

```
Registers (fast, tiny)     → Per-cell intermediate values
Shared Memory (fast, ~192KB) → Tile state during fusion
L2 Cache (medium, ~40MB)   → Weight reuse across tiles
HBM (slow, large)          → Full state, initial/final I/O
```

---

## 9. Summary

The arithmetic intensity analysis reveals that:

1. **Naive NCA** (~17 FLOPs/byte) is memory-bound but already better than transformer decoding
2. **Tiled multi-step fusion** achieves ~306 FLOPs/byte (vs HBM) and ~253 FLOPs/byte (vs L2), exceeding compute-bound thresholds for both memory levels
3. **Transformer decoding** (~1 FLOPs/byte) is severely memory-bound
4. **The ~300× AI improvement** of fused NCA over transformer decoding represents significant inference speedup potential

**Memory hierarchy reality**: The fusion strategy requires careful placement of data:
- **Tile state** must fit in SRAM (shared memory) for multi-step fusion to work
- **Weights** stream from L2 cache—and because L2 bandwidth is higher (~5 TB/s vs 2 TB/s), the L2 threshold is *lower* (~62 vs ~156), making weight streaming easier to hide
- **Full state grid** lives in HBM, accessed only at tile boundaries

**Compute-bound verification**: We must check AI against *each* memory level's threshold separately. Because L2 has higher bandwidth, its threshold is lower, not higher. Our workload exceeds both thresholds, confirming we're compute-bound.

**Scaling constraints**: The scheme works well for small-to-medium NCAs (M ≤ 512, C ≤ 32) but may degrade with larger models due to L2 cache pressure and reduced tile efficiency.

This makes NCA an attractive target architecture for distillation when inference efficiency is critical, particularly for:
- Real-time applications
- Edge deployment
- High-throughput serving
- Latency-sensitive workloads
