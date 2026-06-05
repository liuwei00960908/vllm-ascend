# DSA Latent KV Offload (GLM5.1 / GlmMoeDsa) — Design

Status: **v1 design locked, implementation in progress**

## Goal

For the GLM5.1 model (`GlmMoeDsaForCausalLM`, which reuses `deepseek_v2.py` with
DeepSeek Sparse Attention / DSA) on Ascend NPU:

- **Prefill**: after prefill, offload every token's **MLA latent KV** to LMCache,
  and do not keep that latent resident in NPU memory.
- **Decode**: the DSA lightning indexer selects `topk` historical tokens per query;
  load **only those selected tokens'** latent back from LMCache, run sparse
  attention over them. Newly generated decode tokens' latent stays resident on
  the NPU (v1 limitation, see below).
- Store/load with LMCache is **layerwise**.

## Why DSA makes this natural (and what is new)

DSA has two distinct caches per layer:

| Cache | Tensor | Size / token | Lifecycle in this design |
|---|---|---|---|
| **Indexer key** | `kv_cache[2]` (+`[3]` scale) | tiny (~132 B fp8) | **Stays fully resident** — the indexer must score *all* historical tokens to pick top-k, so it cannot be sparsely offloaded. Normal paged cache. |
| **MLA latent** | `kv_cache[0]`=k_nope, `kv_cache[1]`=k_pe | large (`kv_lora_rank + qk_rope_head_dim`) | **Offloaded to LMCache, not paged.** Only top-k selected tokens are gathered back per decode step. |

The existing Ascend SFA kernel (`vllm_ascend/attention/sfa_v1.py`) already does
**compute sparsity**: `npu_sparse_flash_attention(key=kv_cache[0], key_rope=kv_cache[1],
sparse_indices=topk_indices, block_table=...)` gathers the selected tokens from the
**fully-resident paged latent cache**. This design adds **memory sparsity** on top:
the latent is no longer fully resident; we materialize only the selected tokens into
a small scratch buffer and point the kernel at it.

## Locked decisions

1. **Option A** — latent never enters the seq-length-growing paged cache. Prefill
   latent is computed → streamed → pushed to LMCache → buffer reused. No paged
   block freeing is needed because no full latent paged blocks are ever allocated.
   The block manager only ever grows the small indexer-key cache.
2. **A1 scratch** — a single contiguous scratch pool (not a mini pager). Each decode
   step gathers the batch's selected latents compactly into it and rebuilds a compact
   `sparse_indices` (`[0..k-1]`) + `block_table` pointing at the scratch.
3. **Indexer key resident** — accepted; small.
4. **v1 perf** — per decode step, per layer, a synchronous LMCache gather on the
   critical path is acceptable (no layer-overlap/prefetch in v1). This is *forced*:
   layer N's selection isn't known until layer N's indexer runs, so prefetch is
   impossible anyway. This is also why the scratch only needs **one layer's** capacity
   (reused across layers within a forward).
5. **Decode tokens not offloaded in v1** — `store` happens exactly once at prefill end.
   Newly generated decode tokens' latent stays on the NPU.

## Consequence of (5): the decode gather is mixed-source

The indexer keys cover the *whole* sequence (prefill + decode), so the top-k it
returns at decode step *t* can include both:

- **prefill tokens** → latent in **LMCache** → gather via `LatentOffloadBackend.load_layer`.
- **previously-generated decode tokens** → latent **not** in LMCache → must be read
  from a resident **decode-latent store** on the NPU.

Both are gathered into the same scratch before the kernel runs.

## Buffers (all reserved from the profiled KV-cache budget — no rogue allocation)

To keep the scheduler OOM-safe, both buffers are **fixed-size** and **pre-subtracted**
from `available` before the block-budget split (`available -= scratch + decode_store`),
then allocated by the KV-cache machinery. The scheduler's `num_blocks` shrinks
accordingly; it never hands out memory we use.

| Buffer | Persistence | Layers | Approx size | Notes |
|---|---|---|---|---|
| **scratch pool** (A1) | per-layer, reused within a forward | 1 | `max_num_seqs × topk × latent` (~0.6 GB) | read buffer for the kernel |
| **decode-latent store** | persistent across steps | all | `max_num_seqs × D × num_layers × latent` | `D = max_resident_decode_tokens` (config). v1 cap. |

**v1 limit**: a single request generating more than `D` tokens exceeds the decode
store. v2 lifts this by also offloading decode tokens to LMCache.

## Data flow

```
KV-cache init (NPUModelRunner):
  available -= (scratch_bytes + decode_store_bytes)        # scheduler-safe reservation
  indexer-key layer  -> normal paged spec (full seq, resident)
  MLA latent layers  -> wired to scratch / decode-store (no full paged alloc)

Prefill (AscendSFAImpl, prefill branch):
  indexer key -> paged kv_cache[2]            (resident, unchanged)
  latent      -> streaming buffer -> backend.store_layer(req, layer, all_pos, latent)   # once

Decode (AscendSFAImpl, between indexer and sparse-attn):
  topk_indices = indexer_select_post_process(...)          # unchanged
  prefill_idx, decode_idx = split(topk_indices, prompt_len)
  scratch[0..a]   <- backend.load_layer(req, layer, prefill_idx)   # from LMCache
  scratch[a..a+b] <- decode_store[req, layer, decode_idx]          # from NPU
  compact_indices, scratch_block_table = remap(...)        # A1
  npu_sparse_flash_attention(key=scratch_knope, key_rope=scratch_kpe,
                             sparse_indices=compact_indices,
                             block_table=scratch_block_table, ...)
  decode_store[req, layer, t] <- current_token_latent       # keep new token on NPU
```

## LMCache abstraction boundary

All dependence on the (still-evolving) LMCache layerwise API is isolated to
`offload_backend.py::LatentOffloadBackend`. Agreed so far:

- `store_layer(req_id, layer_id, token_positions, latent)` — called once at prefill end.
- `load_layer(req_id, layer_id, selected_tokens)` — gather-by-index at decode; the
  `selected_tokens` argument is confirmed. Async/event-returning, batched-multi-request,
  and exact tensor layouts are **left open** and will be filled when the colleague's
  interface lands. An in-memory reference backend is provided so the full pipeline
  runs end-to-end before then.

## Out of scope for v1
- Layer-overlap / prefetch of latent loads.
- GPU "hot cache" of recently selected tokens (fetch-miss-only).
- Offloading decode-generated tokens to LMCache.
- Models other than `GlmMoeDsaForCausalLM` / DSA-style sparse attention.
