# DSA shared pool raw-slab reshape (DSA replay Step 5 / 5b-1).
# Copied wholesale from the internal fork vllm-ascend-sparse@c7c4a4ac
# vllm_ascend/worker/dsa_shared_pool.py (self-contained; no edits).
# Official base: v0.23.0 final (5cb98caaa) has no counterpart.
import math

import torch


def reshape_dsa_shared_pool_raw(
    raw: torch.Tensor,
    dtype: torch.dtype,
    block_size: int,
    num_kv_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    index_head_dim: int,
    *,
    is_indexer: bool,
) -> tuple[torch.Tensor, ...]:
    """Create latent or indexer PA_BSND views from one DSA shared raw slab."""

    elt = torch.empty((), dtype=dtype).element_size()
    latent_page = block_size * num_kv_heads * (kv_lora_rank + qk_rope_head_dim) * elt
    indexer_page = block_size * num_kv_heads * index_head_dim * elt
    bundle_page = math.lcm(latent_page, indexer_page)
    assert bundle_page == 9 * indexer_page, (
        "DSA shared pool expects one bundle to be nine indexer pages; "
        f"latent_page={latent_page}, indexer_page={indexer_page}."
    )
    assert raw.numel() % bundle_page == 0
    slot_count = raw.numel() // bundle_page
    latent_blocks = slot_count * (bundle_page // latent_page)
    indexer_blocks = slot_count * (bundle_page // indexer_page)
    nope_pages = latent_blocks * kv_lora_rank // index_head_dim
    pe_pages = latent_blocks * qk_rope_head_dim // index_head_dim
    assert nope_pages + pe_pages == indexer_blocks

    if is_indexer:
        return (
            raw.view(dtype).view(
                indexer_blocks,
                block_size,
                num_kv_heads,
                index_head_dim,
            ),
        )

    nope_bytes = nope_pages * indexer_page
    pe_bytes = pe_pages * indexer_page
    k_nope = raw[:nope_bytes].view(dtype).view(
        latent_blocks,
        block_size,
        num_kv_heads,
        kv_lora_rank,
    )
    k_pe = raw[nope_bytes : nope_bytes + pe_bytes].view(dtype).view(
        latent_blocks,
        block_size,
        num_kv_heads,
        qk_rope_head_dim,
    )
    return k_nope, k_pe
