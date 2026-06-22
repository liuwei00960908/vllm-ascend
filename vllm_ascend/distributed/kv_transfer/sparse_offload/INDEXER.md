# DSA Indexer — handoff for adapter integration

Goal of this doc: give the adapter owner (1) the **entry points** of the indexer, (2)
the **indexer logic**, and (3) **how it currently sits in the forward**, so the
indexer (its key cache + selection) can be pulled into the adapter the same way the
latent already was.

All code is in `vllm_ascend/attention/sfa_v1.py`, class `AscendSFAImpl`. Line numbers
are approximate (they drift); search by method name.

---

## 0. What the indexer is (1 paragraph)

DSA attention is two-stage. A **lightweight "lightning indexer"** first scores *every*
historical token cheaply and picks the **top-k = 2048** most relevant tokens per query;
then the heavy sparse-attention kernel only attends those 2048. The indexer has its own
tiny per-token **key** (`index_head_dim ≈ 128`, GLM5.1: `n_head=64, head_dim=128`),
**separate from the MLA latent**. That indexer-key cache must stay **fully resident** —
the indexer scores *all* history every step, so it can't be sparsely offloaded (unlike
the latent). The indexer's output (`topk_indices`) is exactly the input your adapter's
`retrieve()` already consumes.

---

## 1. Entry points (already cleanly separated)

The indexer is **two methods** on `AscendSFAImpl`, plus one cache write between them:

| Phase | Method | What it returns |
|---|---|---|
| build the **key** for this step's tokens | `indexer_select_pre_process(x, cos, sin)` (~L959) | `(k_li, k_li_scale)` |
| *(write key to resident cache)* | `npu_scatter_nd_update_` in `forward` (~L1354) | — (writes `kv_cache[2]`) |
| build the **query** + run selection | `indexer_select_post_process(x, q_c, kv_cache, attn_metadata, cos, sin, actual_seq_lengths_query, actual_seq_lengths_key)` (~L999) | `topk_indices` |

The actual selection kernel is `npu_lightning_indexer` (3 variants, §4), called *inside*
`post_process`.

These two methods are self-contained pure-ish functions (state = the layer's indexer
weights + `kv_cache[2]`), so they're the natural seam to move into the adapter.

---

## 2. Phase 1 — `indexer_select_pre_process(x, cos, sin)` (build the key)

Input: `x` = hidden_states `[num_tokens, hidden]`; `cos/sin` = RoPE tables.

```
k_li = wk(x)                      # [num_tokens, head_dim=128]   (self.wk)
k_li = k_norm(k_li)               # RMSNorm                       (self.k_norm)
k_li = RoPE(k_li)                 # rotate the first qk_rope_head_dim dims
# c8 mode only: k_li = Hadamard(k_li); k_li, k_li_scale = dynamic_quant(int8)
return k_li, k_li_scale           # k_li_scale is None unless c8
```

Then `forward` writes the key into the **resident indexer-key cache** (`kv_cache[2]`):

```python
torch_npu.npu_scatter_nd_update_(
    kv_cache[2].view(-1, k_li.shape[-1]), idx_slot_mapping.view(-1, 1), k_li.view(...))
# c8: same for kv_cache[3] (the int8 key scale)
```

- `kv_cache[2]` = the indexer-key cache, **PA_BSND** paged layout, full-context resident.
- `idx_slot_mapping` = where this step's tokens go (two-group: indexer group's own slots;
  single-group: shared `slot_mapping`). See §5.

## 3. Phase 2 — `indexer_select_post_process(...)` (build query + select)

```
weights = weights_proj(x)                         # per-token scoring weights
q_li = wq_b(q_c)                                   # [num_tokens, n_head=64, 128]  (self.wq_b)
q_li = RoPE(q_li)
# c8 mode only: q_li = Hadamard(q_li); q_li, q_li_scale = dynamic_quant(int8)
topk_indices = npu_lightning_indexer(
    query   = q_li,
    key     = kv_cache[2],                          # the resident indexer-key cache
    weights = weights,
    actual_seq_lengths_query = actual_seq_lengths_query,   # cum_query_lens (per request)
    actual_seq_lengths_key   = actual_seq_lengths_key,     # seq_lens (per request)
    block_table = indexer_block_table,              # §5
    layout_query = "TND", layout_key = "PA_BSND",
    sparse_count = 2048, sparse_mode = 3,
)
return topk_indices
```

- `q_c` = the down-projected query latent (`q_a_layernorm` output), the same `q_c`
  used by the MLA q-up-proj.
- **Output `topk_indices`**: `[num_query_tokens, 1, sparse_count=2048]`, int. Each row =
  one query token's selected **absolute sequence positions**, front-packed, `-1` padded.
  This is exactly what the sparse FA (and your `retrieve()`) consume. NOTE the singleton
  middle dim — the decode path collapses `[:, 0, :]`.

## 4. The 3 kernel variants (which one runs)

Selected in `post_process` by flags set in `__init__`:

1. `use_sparse_c8_indexer` (`ascend_config.enable_sparse_c8`) → `torch.ops._C_ascend.npu_lightning_indexer_quant`, and `kv_cache` is a **4-tuple** (`[2]` key int8, `[3]` key scale).
2. `use_torch_npu_lightning_indexer` (set for `model_type == "glm_moe_dsa"`) → `torch_npu.npu_lightning_indexer`. **← GLM5.1 path.**
3. else → `torch.ops._C_ascend.npu_lightning_indexer`.

For your GLM5.1 bring-up it's **variant 2** (and `is_rope_neox_style = False` for GLM).

## 5. Inputs / dependencies the indexer needs

| Thing | Source | Notes |
|---|---|---|
| `x` (hidden) | forward arg | for `wk` (key) and `weights_proj` |
| `q_c` | `q_a_layernorm(q_a_proj(x))` upstream | for `wq_b` (query) |
| `cos`, `sin` | `attn_metadata.cos/sin` | RoPE |
| `kv_cache[2]` (+`[3]` c8) | resident indexer-key cache | written by phase 1, read by phase 2 |
| `actual_seq_lengths_query` | `attn_metadata.cum_query_lens` | per **request** |
| `actual_seq_lengths_key` | `attn_metadata.seq_lens` | per **request** |
| `block_table` | `attn_metadata.indexer_block_table` or `.block_table` | §5 below |
| weights | `self.indexer`: `wq_b`, `wk`, `weights_proj`, `k_norm` | `n_head=64`, `head_dim=128` |
| `idx_slot_mapping` | `attn_metadata.indexer_slot_mapping` or `.slot_mapping` | where to write the key |

**Single-group vs two-group (`VLLM_ASCEND_DSA_UNBUNDLE`)**:
- single-group: indexer shares the latent's `block_table` / `slot_mapping`.
- two-group (unbundle): the indexer key is its **own KV group**
  (`DeepseekV32IndexerCache`, sibling `...self_attn.indexer.k_cache`), with its own
  `indexer_block_table` / `indexer_slot_mapping`. `forward` re-assembles a 3-tuple
  `kv_cache = (k_nope, k_pe, indexer_key)` at entry (~L1156-1171) so phases 1/2 work
  unchanged.

## 6. Where it sits in `forward` (current order, decode)

```
1. exec_kv(...)                      -> k_nope, k_pe        (MLA latent)
2. k_li, k_li_scale = indexer_select_pre_process(x, cos, sin)
3. npu_scatter_nd_update_(kv_cache[2], idx_slot_mapping, k_li)   # cache the key (resident)
4. topk_indices = indexer_select_post_process(x, q_c, kv_cache, attn_metadata, cos, sin,
                                              actual_seq_lengths_query, actual_seq_lengths_key)
5. sparse FA over the selected tokens   # adapter: retrieve(topk_indices) -> block_table -> FA
```

Profiled cost (per layer-call, GLM5.1 / 10-layer test): indexer ≈ **0.55–0.6 ms**, i.e.
comparable to one adapter retrieve sub-step. The indexer key cache is resident
(~hundreds of MB for the full context × concurrency × layers; not offloaded).

## 7. What moving the indexer into the adapter implies

The indexer is **independent of the latent offload** — it only needs (a) its key cache
resident and (b) the lightning kernel. To pull it into the adapter:

1. **Own `kv_cache[2]`** (the indexer-key cache) in the adapter, the way you own the
   latent pools. It's the *one* cache that stays fully resident; it is NOT sparse-able.
2. Call `indexer_select_pre_process` → write the key into your indexer cache via your
   own `idx_slot_mapping` (paged, PA_BSND).
3. Call `indexer_select_post_process` (or inline the kernel) with `key = your indexer
   cache` and your `block_table`, the per-request `actual_seq_lengths_*` → `topk_indices`.
4. Feed `topk_indices` to the existing decode path (it already flows into `retrieve()`).

Caveats to carry over (we hit these on the latent side):
- **ACL-graph batch padding**: `topk_indices` rows = padded query length; the
  sparse-attn kernel wants `sparse_indices` at the padded length but
  `block_table`/`actual_seq_lengths_kv` at the **real** request count. The lightning
  indexer itself takes `actual_seq_lengths_*` per request (real) + `block_table` per
  request. Match native shapes, don't assume one batch size.
- **No host syncs on the hot path** (decode runs inside the graph / async scheduler).
- **MTP / draft model** reuses these layers — don't cache per-step state keyed by layer
  identity or `attn_metadata` identity.

---

### Quick reference — indexer symbols in sfa_v1.py

- `indexer_select_pre_process` — build key (~L959)
- `indexer_select_post_process` — build query + select (~L999)
- key cache write `npu_scatter_nd_update_` — (~L1354)
- forward call sites — pre (~L1232), post (~L1376)
- indexer weights bound in `__init__` from `self.indexer` (~L481-486)
- `kv_cache[2]` = indexer key, `kv_cache[3]` = key scale (c8); two-group reassembly (~L1156)
