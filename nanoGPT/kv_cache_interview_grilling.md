# Systems & ML Infra Interview Guide: Grilling on KV Caching (nanoGPT to Production)

This document contains a comprehensive collection of **hard interview grilling questions, edge-case probes, mathematical derivations, and production-grade answers** on KV Caching for Machine Learning Infrastructure, Systems ML, and LLM Engineering interviews.

---

## Table of Contents
1. [Category 1: Code-Level Vulnerabilities & Edge Cases in nanoGPT](#category-1-code-level-vulnerabilities--edge-cases-in-nanogpt)
2. [Category 2: Memory & Compute Complexity Grilling](#category-2-memory--compute-complexity-grilling)
3. [Category 3: PyTorch & Hardware Level Execution](#category-3-pytorch--hardware-level-execution)
4. [Category 4: Production Memory Optimizations (PagedAttention, GQA, Quantization)](#category-4-production-memory-optimizations-pagedattention-gqa-quantization)
5. [Category 5: Distributed Inference & Serving Systems](#category-5-distributed-inference--serving-systems)
6. [Category 6: Hands-On Coding & Design Grilling](#category-6-hands-on-coding--design-grilling)

---

## Category 1: Code-Level Vulnerabilities & Edge Cases in nanoGPT

### Q1.1: "Look at your `model.py` code. What happens if a user inputs a prompt whose length exceeds `config.block_size` when `use_cache=True`?"
**Interviewer Goal**: Test if you understand prompt cropping vs cache offset synchronization.

**Answer**:
* In `GPT.generate()`, when $i=0$ (prefill step), if `idx.size(1) > block_size`, the prompt is cropped to the last `block_size` tokens: `idx_cond = idx[:, -self.config.block_size:]`.
* `curr_pos` is incremented by `idx_cond.size(1)` (which equals `block_size`).
* **The Bug/Vulnerability**: In subsequent token steps ($i > 0$), `curr_pos` becomes `block_size + 1`. In `GPT.forward()`, `pos = torch.arange(pos_offset, pos_offset + 1)` generates position index `[block_size]`.
* When calling position embedding lookup `self.transformer.wpe(pos)`, PyTorch raises an out-of-bounds `IndexError` because `wpe` weight matrix is shape `(block_size, n_embd)`.
* **Fix**: Enforce `assert pos_offset + t <= self.config.block_size` in `GPT.forward()` and crop historical cache if sliding context window is desired.

---

### Q1.2: "In `CausalSelfAttention.forward()`, you set `is_causal = (T == T_ctx)`. Why does this work for $T=1$ token decoding, but fail catastrophically if someone passes a chunked prompt of length $1 < T < T_{ctx}$?"
**Interviewer Goal**: Test deep understanding of FlashAttention / PyTorch SDPA masking semantics.

**Answer**:
* **Prefill Phase ($T = T_{ctx}$)**: `T == T_ctx` is `True`. PyTorch SDPA applies a lower-triangular causal mask of size $(T, T)$.
* **Single-Token Decoding ($T = 1 < T_{ctx}$)**: `T == T_ctx` is `False`. SDPA treats attention as unmasked across sequence length $T_{ctx}$. Because $T=1$, the single query token is at position $T_{ctx}-1$ (the very end of the cached context), so attending to all past $0..T_{ctx}-1$ tokens is naturally causal without masking!
* **Failure Case (Chunked Prefill $1 < T < T_{ctx}$)**: If a user feeds a chunk of length $T=4$ into a cache already containing 10 tokens ($T_{ctx}=14$), `is_causal` evaluates to `False`. SDPA executes **unmasked attention**, allowing token 2 inside the chunk to illegally attend to future token 4 inside the same chunk!
* **Fix**: For chunked prefill, construct an explicit 2D/4D block-diagonal attention mask matrix of shape `(T, T_ctx)` matching absolute token positions `[pos_offset .. pos_offset + T - 1]`.

---

### Q1.3: "If `self.flash = False` (manual attention), why does `att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))` crash during chunked prefill?"
**Interviewer Goal**: Test multi-dimensional tensor broadcasting awareness.

**Answer**:
* `att` tensor shape is `(B, nh, T, T_ctx)`.
* `self.bias[:, :, :T, :T]` shape is `(1, 1, T, T)`.
* When $T < T_{ctx}$, PyTorch attempts to broadcast `(1, 1, T, T)` with `(B, nh, T, T_ctx)`. PyTorch raises `RuntimeError: The size of tensor a (T_ctx) must match the size of tensor b (T) at non-singleton dimension 3`.
* For single-step decoding ($T=1$), `self.bias[:,:,:1,:1]` shape `(1,1,1,1)` broadcasts to `(B, nh, 1, T_ctx)`, and since `bias[0,0,0,0] == 1`, `masked_fill` fills nothing (correct for $T=1$). But for $T > 1$, sliced indexing matching `pos_offset` (`self.bias[:, :, pos_offset:pos_offset+T, :T_ctx]`) is required.

---

## Category 2: Memory & Compute Complexity Grilling

### Q2.1: "Calculate the exact KV cache memory consumption in GB for Llama-3-70B running batch size 32, context length 8,192 tokens in FP16 precision."
**Interviewer Goal**: Test if you can perform real-world hardware sizing calculations under pressure.

**Model Parameters**:
* Layers ($L$) = 80
* KV Heads ($H_{kv}$) = 8 (Grouped-Query Attention)
* Head Dimension ($d_h$) = 128
* Batch Size ($B$) = 32
* Sequence Length ($S$) = 8,192
* Bytes per element ($P$) = 2 (FP16)

**Formula**:
$$\text{KV Cache Bytes} = 2 \ (\text{for } K \text{ and } V) \times L \times H_{kv} \times d_h \times S \times P \times B$$

**Calculation**:
$$\begin{aligned}
\text{Per-token KV size per layer} &= 2 \times 8 \text{ heads} \times 128 \text{ dim} \times 2 \text{ bytes} = 4,096 \text{ bytes (4 KB)} \\
\text{Per-token KV size across 80 layers} &= 80 \times 4,096 = 327,680 \text{ bytes (320 KB)} \\
\text{Single sequence (8,192 tokens)} &= 8,192 \times 320 \text{ KB} = 2,621.44 \text{ MB} \approx 2.62 \text{ GB} \\
\text{Batch of 32 sequences} &= 32 \times 2.62144 \text{ GB} = \mathbf{83.88\text{ GB}}
\end{aligned}$$

* **Takeaway**: The KV cache alone for batch size 32 takes **83.88 GB**, exceeding a full 80GB A100 GPU's memory without even counting model weights!

---

### Q2.2: "Why is the Prefill Phase compute-bound, while the Decoding Phase is memory-bandwidth-bound? Show the Arithmetic Intensity."
**Interviewer Goal**: Test understanding of Roofline Model analysis in Systems ML.

**Answer**:
* **Arithmetic Intensity (Operational Intensity)** = $\frac{\text{Total FLOPs}}{\text{Total Bytes Transferred from Memory (HBM to SRAM)}}$.
* **Prefill Phase ($P$ tokens)**:
  - Matrix multiplication $Q \cdot K^T$ operates on matrices of shape $(P, d) \times (d, P)$.
  - Computes $O(P^2 \cdot d)$ FLOPs while loading $O(P \cdot d)$ weight bytes.
  - Arithmetic Intensity $\propto P$. For large $P$, intensity exceeds the GPU Roofline turning point (~150-300 FLOPs/byte on A100). Thus, **Prefill is Compute-Bound**.
* **Decoding Phase ($T=1$ token)**:
  - Performs Matrix-Vector multiplications (GEMV) of shape $(1, d) \times (d, d)$.
  - Computes $O(d^2)$ FLOPs, but MUST load ALL model weight parameters $O(d^2)$ and ALL KV cache historical bytes $O(T \cdot d)$ from HBM for every single token!
  - Arithmetic Intensity $\approx 1-2 \text{ FLOPs/byte}$.
  - The GPU Tensor Cores sit idle ~95% of the time waiting for memory transfers over the HBM bus. Thus, **Decoding is Memory-Bandwidth-Bound**.

---

## Category 3: PyTorch & Hardware Level Execution

### Q3.1: "Why is `torch.cat((self.cache_k, k), dim=2)` unacceptable in a production inference engine?"
**Interviewer Goal**: Test memory management awareness (allocation churn, fragmentation, latency spikes).

**Answer**:
1. **Dynamic Memory Reallocation Churn**: Calling `torch.cat` on every step allocates a completely new memory chunk of size $(B, nh, T_{past}+1, hs)$ and copies all previous $T_{past}$ data over CUDA streams.
2. **GPU Memory Fragmentation**: Repeated allocations and frees of continuously growing tensors pollute PyTorch's `CUDA Caching Allocator`, causing virtual memory fragmentation and early Out-Of-Memory (OOM) errors even when total free memory appears sufficient.
3. **CUDA Driver Overhead**: Synchronous memory allocations trigger kernel launch latency overheads.

---

### Q3.2: "How would you implement a production-grade Static KV Cache in PyTorch without `torch.cat`?"
**Interviewer Goal**: Test hands-on PyTorch engineering skill.

**Answer**:
Pre-allocate a fixed max-capacity tensor buffer during model initialization, and mutate slices in-place using narrow view indexing:

```python
class StaticKVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_len, n_kv_heads, head_dim, device, dtype):
        super().__init__()
        # Pre-allocate contiguous static memory buffer ONCE
        self.k_cache = torch.zeros((max_batch_size, n_kv_heads, max_seq_len, head_dim), device=device, dtype=dtype)
        self.v_cache = torch.zeros((max_batch_size, n_kv_heads, max_seq_len, head_dim), device=device, dtype=dtype)

    def update(self, k_val, v_val, start_pos):
        # k_val shape: (B, n_kv_heads, seq_len, head_dim)
        seq_len = k_val.size(2)
        end_pos = start_pos + seq_len
        
        # In-place slice mutation (NO memory allocation)
        self.k_cache[:, :, start_pos:end_pos, :] = k_val
        self.v_cache[:, :, start_pos:end_pos, :] = v_val
        
        # Return valid cached context slice
        return self.k_cache[:, :, :end_pos, :], self.v_cache[:, :, :end_pos, :]
```

---

## Category 4: Production Memory Optimizations (PagedAttention, GQA, Quantization)

### Q4.1: "Explain PagedAttention (vLLM). How does it solve internal and external memory fragmentation?"
**Interviewer Goal**: Test knowledge of modern LLM serving architecture (vLLM paper).

**Answer**:
* **Traditional KV Cache Problem**: Standard inference engines allocate contiguous memory for max context length (e.g. 2048 tokens) per request. This causes:
  - **Internal Fragmentation**: Reserved un-generated token slots go wasted.
  - **External Fragmentation**: Virtual memory allocators cannot assemble non-contiguous free memory chunks.
  - Up to **60%–80% of KV cache VRAM is wasted**.
* **PagedAttention Solution**: Inspired by OS Virtual Memory:
  - Divides KV cache into fixed-size physical memory **Blocks** (e.g. 16 tokens per block).
  - Maintains a **Block Table** mapping logical token sequence ranges to non-contiguous physical GPU RAM blocks.
  - Allocates blocks dynamically on-demand as new tokens are generated.
  - **Result**: Reduces wasted KV cache memory to $< 4\%$, enabling up to **2x–4x higher batch size and serving throughput**.

---

### Q4.2: "Compare Multi-Head Attention (MHA), Multi-Query Attention (MQA), and Grouped-Query Attention (GQA)."
**Interviewer Goal**: Test understanding of architectural trade-offs in modern models (Llama-3, Mistral).

```
   MHA (N Query, N KV)        GQA (N Query, G KV)         MQA (N Query, 1 KV)
  Q Q Q Q   K K K K          Q Q Q Q     K   K           Q Q Q Q       K
  │ │ │ │   │ │ │ │          │ │ │ │     │   │           │ │ │ │       │
  ▼ ▼ ▼ ▼   ▼ ▼ ▼ ▼          ▼ ▼ ▼ ▼     ▼   ▼           ▼ ▼ ▼ ▼       ▼
  (Ratio 1:1)               (Ratio 4:2 = 2:1)           (Ratio 4:1)
```

| Attention Variant | Key/Value Heads ($H_{kv}$) | KV Cache Memory Savings | Model Quality / Capacity | Example Models |
| :--- | :--- | :--- | :--- | :--- |
| **MHA** | $H_{kv} = H_q$ | 0% (Baseline 1.0x) | Highest | GPT-2, Original Transformer |
| **MQA** | $H_{kv} = 1$ | **$(H_q - 1)/H_q \approx 90-98\%$ reduction** | Slight quality degrade / harder training | Falcon-7B, PaLM |
| **GQA** | $1 < H_{kv} < H_q$ | **$\sim 87.5\%$ reduction** (for 8:1 ratio) | Matches MHA accuracy | Llama-2-70B, Llama-3, Mistral |

---

### Q4.3: "How does FP8 / INT4 KV Cache Quantization work? What are the key precision issues with Key vs Value tensors?"
**Interviewer Goal**: Test deep understanding of quantization dynamics.

**Answer**:
* **Quantization Scheme**: Scale factors mapping FP16 values to FP8 (`e4m3` or `e5m2`) or INT8:
  $$X_{quant} = \text{clamp}\left(\text{round}\left(\frac{X}{\text{scale}}\right), -128, 127\right)$$
* **Asymmetry between Key and Value Tensors**:
  - **Key ($K$) Tensors**: Contain magnitude outliers along specific channel dimensions (especially after RoPE positional encodings). Channel-wise or per-head scaling factors are necessary to prevent attention matrix distortion.
  - **Value ($V$) Tensors**: More uniformly distributed. Per-token or per-tensor quantization works well.
* **FP8 (`e4m3`) vs (`e5m2`)**: `e4m3` is preferred for KV cache because it offers higher precision (4 mantissa bits) over range (3 exponent bits), which is critical for attention accuracy.

---

## Category 5: Distributed Inference & Serving Systems

### Q5.1: "How is the KV Cache distributed across GPUs under Tensor Parallelism (Megatron-LM column/row parallel)?"
**Interviewer Goal**: Test multi-GPU distributed system architecture.

**Answer**:
* In Tensor Parallelism with size $N$:
  - Linear projection $W_{qkv}$ is **Column-Parallel split** across $N$ GPUs.
  - GPU $k$ holds query heads $[k \cdot \frac{H_q}{N} .. (k+1) \cdot \frac{H_q}{N}]$ and key/value heads $[k \cdot \frac{H_{kv}}{N} .. (k+1) \cdot \frac{H_{kv}}{N}]$.
  - **Key Result**: Each GPU stores **only its local fraction ($\frac{1}{N}$) of the total KV cache**!
  - Local attention output $Y_k = \text{SDPA}(Q_k, K_k, V_k)$ is computed independently on GPU $k$.
  - An `All-Reduce (Sum)` operation across $N$ GPUs is performed AFTER output projection $W_o$ (Row-Parallel).

---

### Q5.2: "What is Chunked Prefill (e.g. Sarathi-Lean / vLLM Chunked Prefill) and why is it crucial for latency SLOs?"
**Interviewer Goal**: Test knowledge of state-of-the-art LLM serving schedulers.

**Answer**:
* **Problem (Interference & Head-of-Line Blocking)**:
  - Long prefill requests (e.g., 8,192 prompt tokens) take hundreds of milliseconds of GPU compute.
  - If a long prefill request enters the queue, ongoing decoding requests (which require fast ~10-20ms Time-To-First-Token / Inter-Token Latency) get blocked waiting for the prefill to finish!
* **Chunked Prefill Solution**:
  - Breaks long prompt prefills into smaller fixed-size chunks (e.g., 512 tokens).
  - Co-locates prompt chunks with active single-token decoding requests in the same batch iteration.
  - Maintains consistent GPU compute utilization and eliminates decoding latency spikes.

---

## Category 6: Hands-On Coding & Design Grilling

### Q6.1: "Design a Prefix Caching System (Prompt Cache) for a Multi-Tenant API."
**Interviewer Goal**: System design test combining data structures, hashing, and cache eviction.

**Architecture Design**:
1. **Radix Tree / Trie Data Structure**:
   - Store prompt token sequences in a **Radix Tree** where each node represents a block of tokens (e.g., 16 tokens).
   - Each node points to physical KV cache memory block IDs in GPU RAM.
2. **Deterministic Token Hashing**:
   - Compute cryptographic hash (e.g., `SHA-256` or `xxHash`) of `(Parent Node Hash + Block Tokens)` to identify reusable prompt prefixes across tenant requests.
3. **Reference Counting & LRU Eviction**:
   - Assign reference count $R$ to cached nodes.
   - When GPU memory is constrained, evict nodes with $R=0$ using a Least Recently Used (LRU) policy.
4. **Impact**: Reduces Time-To-First-Token (TTFT) for common system prompts by **up to 90%** by eliminating prefill compute.

---

## Summary Cheat Sheet for Interviews

| Topic | Quick Key Fact |
| :--- | :--- |
| **Decoding Bottleneck** | Memory Bandwidth Bound (Arithmetic Intensity $\approx 1-2$ FLOPs/byte) |
| **Prefill Bottleneck** | Compute Bound (GEMM Matrix Multiplication) |
| **KV Cache Size Formula** | $2 \times B \times L \times H_{kv} \times d_h \times S \times \text{Precision}$ |
| **PyTorch `torch.cat` Flaw** | Memory churn, allocation latency, CUDA allocator fragmentation |
| **SDPA `is_causal` logic** | `True` for prefill ($T=T_{ctx}$); `False` for 1-token decoding ($T=1 < T_{ctx}$) |
| **PagedAttention** | Eliminates internal/external memory fragmentation using OS-style page blocks |
| **GQA vs MHA** | GQA reduces KV cache size by $8\times$ while keeping MHA model accuracy |
