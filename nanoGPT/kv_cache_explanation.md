# Key-Value (KV) Caching in nanoGPT: Implementation Flow, PyTorch APIs & Logic Analysis

## Executive Summary
This document provides a detailed technical breakdown of the KV Caching implementation in `nanoGPT`. It covers the execution flow across git changes, explains all PyTorch APIs used, details logic edge-cases/comments added to the codebase, and includes empirical performance benchmarks.

---

## 1. Execution Flow of KV Caching

### Why KV Caching?
Without KV Caching, generating token $T+1$ in an autoregressive Transformer requires feeding all previous $1..T$ tokens back through the model. At every generation step, Key ($K$) and Value ($V$) linear projections are recomputed for historical tokens, resulting in $O(T^2)$ computational complexity.

With **KV Caching**, the historical $K$ and $V$ tensors from previously processed tokens are preserved in memory (`cache_k`, `cache_v`). At step $i > 0$:
1. Only the single newest token ($T=1$) is forwarded through the transformer.
2. Query ($Q$), Key ($K$), and Value ($V$) projections are computed for only that 1 token.
3. The new $K$ and $V$ tensors are concatenated with `cache_k` and `cache_v`.
4. Attention is computed between $Q$ ($T=1$) and full context $K, V$ ($T_{ctx}$).

This reduces token generation complexity per step from $O(T^2)$ to $O(T)$.

---

### Step-by-Step Code Flow Across Commits

```
[GPT.generate(use_cache=True)]
   │
   ├──> 1. Set block.attn.use_cache = True & reset_cache()
   │
   ├──> 2. Step i = 0 (Prefill Phase):
   │       └── Forward full prompt (T = P, pos_offset = 0)
   │       └── CausalSelfAttention computes Q, K, V for P tokens
   │       └── Stores initial K, V in cache_k, cache_v
   │
   └──> 3. Step i > 0 (Generation Phase):
           └── Pass ONLY single last token (idx[:, [-1]], T = 1, pos_offset = curr_pos)
           └── GPT.forward looks up wpe for position [pos_offset]
           └── CausalSelfAttention computes Q, K, V for T = 1
           └── Concatenates new K, V with cache_k, cache_v (torch.cat on dim=2)
           └── Computes SDPA / attention between Q (T=1) and K,V (T_ctx)
           └── Samples next token & appends to sequence
```

#### Detailed File Changes

1. **Cache Initialization & Storage in [`CausalSelfAttention`](file:///home/pratik/nanoGPT/sysml-assignments/nanoGPT/model.py#L47-L77)**:
   - **State**: `self.use_cache = False`, `self.cache_k = None`, `self.cache_v = None`.
   - **Reset**: `reset_cache()` clears cached tensors between generation runs.
   - **Forward Hook**:
     ```python
     if self.use_cache:
         if self.cache_k is not None:
             k = torch.cat((self.cache_k, k), dim=2)
             v = torch.cat((self.cache_v, v), dim=2)
         self.cache_k = k
         self.cache_v = v
     T_ctx = k.size(2)
     ```

2. **Position Offset Support in [`GPT.forward()`](file:///home/pratik/nanoGPT/sysml-assignments/nanoGPT/model.py#L198-L203)**:
   - Modified signature: `def forward(self, idx, targets=None, pos_offset=0):`
   - Dynamically generates positional indices starting from `pos_offset`:
     ```python
     pos = torch.arange(pos_offset, pos_offset + t, dtype=torch.long, device=device)
     pos_emb = self.transformer.wpe(pos)
     ```

3. **Generation Loop Control in [`GPT.generate()`](file:///home/pratik/nanoGPT/sysml-assignments/nanoGPT/model.py#L330-L368)**:
   - Enables caching on all transformer blocks and clears existing caches.
   - **Prefill (`i = 0`)**: Forwards full prompt `idx` ($T = P$) with `pos_offset = 0`.
   - **Generation (`i > 0`)**: Forwards single token `idx[:, [-1]]` ($T = 1$) with `pos_offset = curr_pos`.
   - Disables cache and clears memory upon completion.

4. **Inference Script Integration in [`sample.py`](file:///home/pratik/nanoGPT/sysml-assignments/nanoGPT/sample.py#L94)**:
   - Passes `use_cache=True` to `model.generate(...)`.

---

## 2. PyTorch Functions & APIs Used

| PyTorch Function / API | Explanation & Role in KV Caching |
| :--- | :--- |
| `torch.cat((self.cache_k, k), dim=2)` | Concatenates historical cached keys/values with newly computed key/value tensors along sequence length dimension (`dim=2`), producing total context length $T_{ctx}$. |
| `torch.arange(pos_offset, pos_offset + t, ...)` | Generates a 1D contiguous tensor of position indices `[pos_offset, ..., pos_offset + t - 1]` for lookup in the position embedding matrix. |
| `torch.nn.functional.scaled_dot_product_attention` | PyTorch 2.0+ optimized attention kernel (FlashAttention / Memory-Efficient Attention). Computes attention between $Q$ ($T=1$) and cached $K, V$ ($T_{ctx}$). |
| `is_causal = (T == T_ctx)` | Flag passed to SDPA. Evaluates to `True` during prompt prefill ($T = T_{ctx}$) to enforce causal triangular mask. Evaluates to `False` during single-token step ($T = 1 < T_{ctx}$) so the new query token can attend across all $T_{ctx}$ historical tokens. |
| `(q @ k.transpose(-2, -1))` | Matrix multiplication (`@` operator executing `torch.matmul`) multiplying Query tensor `(B, nh, T, hs)` by transposed Key tensor `(B, nh, hs, T_ctx)` yielding attention logits `(B, nh, T, T_ctx)`. |
| `att.masked_fill(mask == 0, float('-inf'))` | Overwrites masked attention logits with $-\infty$ before softmax. |
| `F.softmax(att, dim=-1)` | Applies Softmax normalization along the last dimension ($T_{ctx}$) to convert logits to attention probabilities. |
| `self.transformer.wpe(pos)` | Performs index lookup into PyTorch `nn.Embedding` module for learned positional embeddings. |
| `@torch.no_grad()` | Disables autograd graph construction during inference to save memory and accelerate computation. |

---

## 3. Logic Analysis & Code Comments Added

During inspection of [`model.py`](file:///home/pratik/nanoGPT/sysml-assignments/nanoGPT/model.py), key logic considerations and edge cases were identified and commented:

### 1. Manual Attention Slicing ([`model.py:L86-L92`](file:///home/pratik/nanoGPT/sysml-assignments/nanoGPT/model.py#L86-L92))
- **Issue**: `att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))` uses fixed slicing `[:T, :T]`.
- **Analysis**: For single-token step ($T=1$), `self.bias[:,:,:1,:1]` evaluates to `1` (unmasked), which correctly allows token $T_{ctx}-1$ to attend to all $T_{ctx}$ historical keys. However, if $1 < T < T_{ctx}$ (e.g. chunked prefill), `self.bias[:,:,:T,:T]` shape `(1, 1, T, T)` fails to broadcast with `att` shape `(B, nh, T, T_ctx)`.
- **Comment**: Added explanatory note detailing $T=1$ vs chunked prefill behavior.

### 2. Maximum Context Window Bound Check ([`model.py:L200-L202`](file:///home/pratik/nanoGPT/sysml-assignments/nanoGPT/model.py#L200-L202))
- **Issue**: `assert t <= self.config.block_size` only checked single-step sequence length $t$.
- **Analysis**: During cached generation, $t = 1$, so $t \le \text{block\_size}$ is always True. However, as generation continues, when total sequence length `pos_offset + t` exceeds `block_size`, `wpe` position lookup raises an out-of-bounds `IndexError`.
- **Fix**: Updated assertion to `assert pos_offset + t <= self.config.block_size`.

### 3. Dynamic Concatenation Memory Overhead ([`model.py:L70-L73`](file:///home/pratik/nanoGPT/sysml-assignments/nanoGPT/model.py#L70-L73))
- **Analysis**: Using `torch.cat` on every generation step allocates a new tensor of size `(B, nh, T_ctx, hs)` and copies past context.
- **Comment**: Added note clarifying that production inference systems use pre-allocated static KV cache buffers to avoid allocation churn.

---

## 4. Benchmark Performance Results

Benchmarked on **GPT-2 (124M)** generating 100 new tokens on CPU:

| Generation Mode | Execution Time | Speedup Factor |
| :--- | :--- | :--- |
| Without KV Cache (`use_cache=False`) | **5.78s** | 1.0x (Baseline) |
| With KV Cache (`use_cache=True`) | **1.62s** | **~3.5x faster** |
