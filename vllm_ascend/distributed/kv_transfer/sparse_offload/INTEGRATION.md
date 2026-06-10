# On-NPU integration checklist (#5 — AscendSFAImpl wiring)

## Round 1 FIRST: introspection (no LMCache, no offload logic)

Workflow: code is written off-NPU; the NPU box only pulls + runs + returns data. So
before finalizing the kernel wiring, capture the ground-truth facts in ONE run:

```bash
VLLM_ASCEND_DSA_OFFLOAD_INTROSPECT=1 \
VLLM_ASCEND_DSA_INTROSPECT_FILE=/tmp/dsa_introspect.log \
  <your normal GLM5.1 launch>   # send ONE short request (prefill + a few decode steps)
# then bring back /tmp/dsa_introspect.log
```

The probes are read-only and gated; they dump: SFA `attn_metadata` fields, the
`kv_cache` tuple layout, `topk_indices` shape + padding sentinel, the MLA layer count
vs `num_hidden_layers`, and `input_batch` fields (req-id / prompt-length source). That
file lets the off-NPU code finalize C/D/E correctly without guessing.



Everything else (backend abstraction, manager, A1 planning, buffer reservation,
manager construction) is done and CPU-unit-tested. This file lists the remaining
wiring that must be done/validated on NPU, because it depends on metadata field
names and the CP code paths that can't be confirmed off-hardware. Anchors are
`file:line` at the time of writing.

The tested entry points to call are in `sfa_hooks.py`:
`store_prefill(...)` and `gather_decode(...)`.

## A. Expose the manager to the attention impl  — DRAFTED (verify on NPU)

DONE: `set_ascend_forward_context` takes a `dsa_offload_manager` kwarg and stores it
on the forward context; the runner's execute path passes
`getattr(self, "dsa_offload_manager", None)`. In the impl, read it via
`get_forward_context().dsa_offload_manager` (None → unchanged native path).

Original notes:

`self.dsa_offload_manager` is built in
`vllm_ascend/worker/model_runner_v1.py::_maybe_init_dsa_latent_offload` (already
added). Make it reachable from `AscendSFAImpl.forward`:
- stash it on the forward context each step (where the runner already sets up
  `ascend_forward_context`), e.g. `ctx.dsa_offload_manager = self.dsa_offload_manager`,
- read it in the impl via the existing `_EXTRA_CTX` / `get_forward_context()`.
A `None` manager means the feature is off → take the unchanged native path.

## B. Carry req_ids + prompt_lens in the SFA metadata  — DRAFTED (verify on NPU)

DONE: `AscendSFAMetadata` now has `req_ids` / `prompt_lens` fields (default None) and
the builder sets them best-effort via `getattr(common_attn_metadata, ...)`.
HW-VERIFY: these fields are NOT on CommonAttentionMetadata today — they live on the
runner's `input_batch` (`input_batch.req_ids[:num_reqs]` and the per-request prompt
token count). Thread them into `common_attn_metadata` (or set on the metadata in the
runner after build) so the getattr resolves; otherwise they stay None and the gather
path must not be enabled.

Original notes:

`AscendSFAMetadata` (sfa_v1.py:115) currently has `seq_lens`, `num_decodes`,
`num_prefills`, `num_decode_tokens` but **not** `req_ids` or `prompt_lens`. Add:
- `req_ids: list[str]` — request id per row, in batch order;
- `prompt_lens: torch.Tensor` — prompt length per request (prefill/decode boundary);
- `query_start_loc: torch.Tensor` — already available as `common_attn_metadata`;
- `context_lens: torch.Tensor` — computed tokens before this step (for chunked prefill).
Populate them where `AscendSFAMetadata` is built (sfa_v1.py:~236-325) from the
common attn metadata / input batch.

## C. Prefill store anchor

In `AscendSFAImpl.forward`, the prompt latent (`k_nope`, `k_pe`) is written to the
paged cache via `reshape_and_cache` (one site at sfa_v1.py:1184, plus the non-CP
site). Right after the latent for this layer is final, on the **prefill** rows:

```python
mgr = get_forward_context().dsa_offload_manager
if mgr is not None and attn_metadata.num_prefills:                 # HW-VERIFY gating
    from vllm_ascend.distributed.kv_transfer.sparse_offload.sfa_hooks import store_prefill
    store_prefill(
        mgr, self.layer_name, attn_metadata.req_ids,               # HW-VERIFY: prefill rows only
        attn_metadata.query_start_loc, attn_metadata.context_lens,
        k_nope.view(-1, self.kv_lora_rank), k_pe.view(-1, self.qk_rope_head_dim),
    )
```
Stage 1: keep the existing `reshape_and_cache` (latent stays paged → lets you assert
the offload path matches native). Stage 2: drop it and shrink the latent KV spec.

## D. Decode gather anchor

Between the indexer (sfa_v1.py:1211, produces `topk_indices`) and the sparse-attn
call (sfa_v1.py:1222), on the **decode** rows:

```python
mgr = get_forward_context().dsa_offload_manager
if mgr is not None and attn_metadata.num_decodes:                  # HW-VERIFY gating
    from vllm_ascend.distributed.kv_transfer.sparse_offload.sfa_hooks import gather_decode
    s_knope, s_kpe, sparse_indices, block_table = gather_decode(
        mgr, self.layer_name, attn_metadata.req_ids,               # HW-VERIFY: decode rows only
        topk_indices, attn_metadata.prompt_lens, block_size,
        cur_k_nope, cur_k_pe,                                       # this step's new-token latent
    )
    # feed these to npu_sparse_flash_attention instead of kv_cache[0]/[1]+block_table
    attn_output = self._execute_sparse_flash_attention_process(
        ql_nope, q_pe, (s_knope, s_kpe), sparse_indices, attn_metadata,
        actual_seq_lengths_query, actual_seq_lengths_key, block_table_override=block_table)
else:
    attn_output = self._execute_sparse_flash_attention_process(...)  # unchanged
```
`_execute_sparse_flash_attention_process` (sfa_v1.py:1009) reads `kv=kv_cache[0]`,
`key_rope=kv_cache[1]`, `block_table=attn_metadata.block_table` — parameterize it to
accept the scratch tensors + override block_table.

## E. Advance the decode step once per step

After the **final** layer of each decode step, call
`mgr.advance_decode_step(req_id)` for every running decode request (every layer wrote
the new token to the same row; this bumps the row). Natural place: end of
`NPUModelRunner.execute_model` for the decode requests. Also call
`mgr.free_request(req_id)` when a request finishes (hook into the runner's finished-
request handling).

## F. The single kernel-semantics HW check

`offload_manager.resolve_scratch_gather` encodes the assumed
`npu_sparse_flash_attention` semantics (`block_table[i//block_size]*block_size +
i%block_size`, `PA_BSND`, `sparse_block_size=1`). The CPU equivalence test
(`test_offload_path_attends_to_exactly_the_selected_tokens`) is green under that
assumption. On NPU, assert the scratch-path output equals the native sparse output
for the same inputs (Stage 1) — if it diverges, the only thing to fix is how
`sparse_indices`/`block_table` are built for the scratch.

## On-NPU runbook & data to collect

### Step 0 — baseline (feature OFF)
- Run GLM5.1 with `VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD=0` (default). Confirm normal
  output. COLLECT (baseline to compare against later):
  - `num_gpu_blocks` (logged at startup), peak NPU memory, max usable context length;
  - throughput (tok/s), TTFT, inter-token latency (ITL) at a fixed batch/seqlen;
  - a fixed prompt set's **greedy output token ids** (the parity reference).

### Step 1 — finish wiring (no kernel redirect yet)
- B: thread `input_batch.req_ids[:num_reqs]` and per-request prompt length into the
  SFA metadata so `attn_metadata.req_ids` / `prompt_lens` are non-None.
- C: add the gated `store_prefill(...)` call after the prefill latent is cached.
- E: call `manager.advance_decode_step(req_id)` once per step after the last layer,
  and `manager.free_request(req_id)` on request finish.
- Turn the flag ON but DO NOT yet redirect the kernel (D). Latent still read from the
  paged cache. COLLECT / CHECK:
  - startup log "DSA latent offload enabled for N MLA layers" → **N == 78**? (vs MTP);
  - "Reserved X GiB for DSA latent offload buffers" → sanity vs your config;
  - `num_gpu_blocks` dropped by ~the reserved amount, **no OOM**, server boots;
  - output token ids **unchanged** vs Step 0 (store path must be side-effect free).

### Step 2 — Stage 1 parity (kernel redirect ON, latent still paged)
- D: build scratch via `gather_decode(...)` and feed it to
  `npu_sparse_flash_attention` instead of `kv_cache[0]/[1]`+`block_table`.
- Keep the paged latent write so you can compare both paths in one run.
- KEY CHECK (F — the one kernel-semantics assumption): for the same decode step,
  assert the scratch-path attention output ≈ the native-sparse output.
  - add a temp debug: run `_execute_sparse_flash_attention_process` both ways and
    `torch.allclose` (or max-abs-diff) the outputs; COLLECT the max-abs / rel diff per
    layer. Expect ~0 (bf16 rounding only). If it diverges → the `sparse_indices` /
    `block_table` construction for the scratch is wrong (see `resolve_scratch_gather`).
  - end-to-end: greedy output token ids **identical** to Step 0 on the fixed prompts;
  - COLLECT: per-step / per-layer `gather_decode` time, LMCache `wait_for_layer_load`
    latency, the D2H sync cost of building the flat `selected_tokens` list.

### Step 3 — Stage 2 memory (actually free the latent)
- Shrink the SFA latent KV spec so `kv_cache[0]/[1]` are no longer sized for the full
  context (only indexer keys `kv_cache[2]` stay full + resident). Remove the paged
  latent write; latent lives only in LMCache (prefill) + decode-store (new tokens).
- COLLECT / CHECK:
  - peak NPU memory and `num_gpu_blocks` vs Step 0 → quantify the saving;
  - **max usable context length** now achievable (the headline win);
  - output token ids still match Step 0 (or document expected sparse-attn deviation);
  - behavior at the decode-store cap: a request generating > `D`
    (`VLLM_ASCEND_DSA_MAX_RESIDENT_DECODE_TOKENS`, default 1024) tokens must raise the
    guarded error, not corrupt — confirm and size `D` for your workload.

### Numbers worth logging throughout
- correctness: max-abs / max-rel logit diff per layer (Stage 1), greedy-token match %;
- memory: reserved GiB, num_gpu_blocks, peak NPU mem, max context len;
- latency: TTFT, ITL, per-layer gather time, LMCache load time, tok/s at fixed load;
- scale: largest batch × context that boots without OOM, before vs after.

### If parity fails (data to capture for debugging)
- dump one decode step: `topk_indices`, `prompt_lens`, the `GatherPlan`
  (`sparse_indices`, `dest_slot`, `scratch_block_table`, `seq_lens_kv`), and the
  scratch contents; compare `resolve_scratch_gather(...)` against directly indexing
  the native paged latent at the topk positions — that isolates remap vs kernel.

## M-B remaining: free prefill latent (the memory win) —施工图

State: `self.dsa_free_paged` flag + SFA spec shrink (head_size -> index_head_dim)
are committed (gated by `VLLM_ASCEND_DSA_OFFLOAD_FREE_PAGED`, default off). The
remaining pieces below are NPU-coupled and startup-crash-prone; do them together,
validate on NPU (boots? `GPU KV cache size` grows? output matches baseline?).

1. **Allocation** (`model_runner_v1.py` `_allocate_kv_cache_tensors`, the
   `if self.use_sparse:` block ~2823): when `self.dsa_free_paged`, do NOT use
   `sparse_kv_cache_ratio` (it assumes 3 entries). Instead size:
   - `dsa_k_tensor_size = kv_cache_tensor.size` (the whole page = indexer);
   - `k_tensor_size = block_size * kv_lora_rank * dtype_bytes` (1-block dummy);
   - `v_tensor_size = block_size * qk_rope_head_dim * dtype_bytes` (1-block dummy);
   - no dsa_k_scale (c8 forced off).
2. **Reshape** (`_reshape_kv_cache_tensors` ~3000): when `dsa_free_paged`, reshape the
   latent k/v caches to a **1-block** shape `(1, block_size, 1, k_dim/v_dim)` (dummies),
   and the indexer to `(num_blocks, block_size, 1, index_head_dim)` at the grown
   num_blocks. Bind tuple = (dummy_knope, dummy_kpe, indexer[, scale]).
3. **exec_kv -> pool** (`sfa_v1.py` forward, `dsa_free_paged` path): reserve pool blocks
   for this step's positions, compute the pool slot_mapping, and call `exec_kv` with the
   pool's (knope, kpe) layer tensors + pool slot_mapping so the op writes latent into
   the pool (NOT the dummy kv_cache[0]/[1]). Then the existing prefill-attn-from-pool and
   decode-gather-from-pool paths work unchanged. (No need for populate_pool_layer /
   store_decode_token in this mode — the op writes the pool directly.)
4. **Free after prefill**: after a request's prefill completes, free its pool latent
   blocks (`PagedLatentPool.free_request` for the prefill range) — content is in LMCache;
   decode reads prefill-selected from LMCache. Hook at the prefill->decode transition or
   via the connector lifecycle.
5. **Pool sizing**: `pool_num_blocks` should be derived from the freed budget; for
   bring-up set `VLLM_ASCEND_DSA_LATENT_POOL_BLOCKS` large enough for concurrent prefill.
   NOTE the known gap (pragmatic, not scheduler-coordinated): if the pool can't fit the
   concurrent prefill load it will raise "PagedLatentPool exhausted" (no backpressure);
   that's the trade-off vs the proper KV-group route.

Validate: baseline `GPU KV cache size: 1,643,264 tokens` should grow ~5.5x with
FREE_PAGED=1; greedy output should match baseline.

## G. Interface check with the LMCache author

`offload_backend.py` assumes `wait_for_layer_load` writes loaded latent into the
registered buffer **tightly in `selected_tokens` order**. Confirm; if it instead
writes at `token_start_index`-based offsets, only `InMemoryLatentOffloadBackend` and
the staging-copy offsets in `gather_decode_layer` change.

## P2 — real LMCache offload (FINAL design, agreed)

Connector: `LMCacheAscendConnectorV1Dynamic` (env: LMCACHE_ENABLE_SPARSE_ATTENTION=true,
LMCACHE_USE_LAYERWISE=true, LMCACHE_CHUNK_SIZE=256; `--kv-transfer-config`). Requires
P1 (`VLLM_ASCEND_DSA_UNBUNDLE=1`) so indexer is its own resident KV group.

NPU memory layout (decode steady state):
  - indexer: vLLM resident group (full context)               [P1, keep]
  - latent prefill buffer: full context, for prefill FA; freed after prefill (may free
    per chunked-prefill chunk). [ours]
  - latent decode scratch: ONE fixed topk-compact paged buffer (k=index_topk), reused
    per layer/step, structure [retrieve | decode]. [ours, registered with connector]

Per-step decode flow (Model C, compact):
  1. indexer -> topk (selected absolute positions).
  2. start_load_kv builds retrieve_layer_head_token_wise(slot_mapping = OUR compact
     slots into the scratch).
  3. per layer: wait_for_layer_load(layer, selected_tokens=topk_prefill_positions,
     token_start_index) -> LMCache scatters the selected PREFILL latent into
     scratch[0 .. n_retrieve).
  4. we append the selected DECODE-token latent (kept on NPU, id>=prompt_len) into
     scratch[n_retrieve .. k).
  5. kernel reads scratch with COMPACT sparse_indices [0..k).

Division of labor:
  - LMCache (colleague): save_kv_layer (offload latent to CPU), retrieve_layer_head_token_wise
    (load selected into the scratch via slot_mapping), Ascend sparse transfer kernel.
  - us (vllm-ascend): allocate+register the topk scratch as the latent layer's kv_cache;
    construct the compact slot_mapping by scratch layout; feed selected_tokens=topk; fill
    the decode segment; compact indices for the kernel; prefill full-buffer + free.

Reuse from the pool work (already parity-validated): scratch buffer (allocate_buffers),
build_gather_plan/resolve_scratch_gather (A1 compact indices), decode-latent store. The
only change from the pool: the RETRIEVE segment is filled by the real LMCache connector
(one fused transfer) instead of my eager per-request gather — which also removes the
eager-mode perf problem.

Blocked on: colleague rebuild of LMCache-Ascend (undefined symbol
kvcache_ops::single_layer_kv_transfer_kernel_v2_mla_dsa_sparse in c_ops.so) before any
smoke test.

## Step A — two-group mode: VERIFIED ON HW (2026-06-10)

vllm fork branch `dsa-two-groups` @ 0d385ed49 + vllm-ascend sparse @ 3d1b1b25,
flags UNBUNDLE=1 + TWO_GROUPS=1 (+ --no-enable-prefix-caching, no LMCache):
  - "Per-group KV block pools: 2 pools x 12976 blocks each"
  - GPU KV cache 1,660,928 tokens, concurrency 50.69x (== P1/base)
  - TPOT 22.36 (base 22.5), seed-0 output token-identical to P1.

Step B (next, the actual memory saving):
  B1 vllm: at prefill end, shrink the request's LATENT blocks — keep the first
     ceil(k/128)=16 blocks as the [retrieve|decode] scratch, swap the rest to
     null_block and free them (same pattern as SlidingWindowManager.
     remove_skipped_blocks; avoids double-free on request finish).
  B2 ascend: decode FA reads the compact scratch — remap topk to compact
     indices [0..k) (reuse pool A1 logic), scratch block_table (first 16
     blocks), copy selected DECODE-token latent into scratch[n_retrieve..k),
     LMCache wait_for_layer_load(selected) fills scratch[0..n_retrieve).
  B3 integration point with colleague (LMCache): on sparse decode the
     ReqMeta slot_mapping must be the SCRATCH slots (first 16 blocks => k
     slots), not the full prefill expansion — else the length-mismatch
     ValueError returns.
  B4 sizing knob: shrink the latent pool (N_l < N_i) => the actual memory
     number. Needs per-group num_blocks in KVCacheConfig.
