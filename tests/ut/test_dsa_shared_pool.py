import torch

from vllm_ascend.worker.dsa_shared_pool import reshape_dsa_shared_pool_raw


def test_dsa_shared_pool_raw_views_match_bundle_layout():
    block_size = 128
    num_heads = 1
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    index_head_dim = 128
    dtype = torch.float16
    element_size = torch.tensor([], dtype=dtype).element_size()

    capacity_bundles = 4
    slot_count = capacity_bundles + 1
    base_page_bytes = block_size * num_heads * index_head_dim * element_size
    bundle_bytes = 9 * base_page_bytes
    raw = torch.empty(slot_count * bundle_bytes, dtype=torch.int8)

    k_nope, k_pe = reshape_dsa_shared_pool_raw(
        raw,
        dtype,
        block_size,
        num_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        index_head_dim,
        is_indexer=False,
    )
    (indexer,) = reshape_dsa_shared_pool_raw(
        raw,
        dtype,
        block_size,
        num_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        index_head_dim,
        is_indexer=True,
    )

    assert k_nope.shape == (slot_count * 2, block_size, num_heads, kv_lora_rank)
    assert k_pe.shape == (slot_count * 2, block_size, num_heads, qk_rope_head_dim)
    assert indexer.shape == (slot_count * 9, block_size, num_heads, index_head_dim)

    raw_ptr = raw.data_ptr()
    assert k_nope.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    assert k_pe.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    assert indexer.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()

    # Bundle 1 maps to latent blocks [2, 3] and indexer rows
    # [8..15, 41] when slot_count == 5.  These offsets must alias.
    assert k_nope[2].data_ptr() - raw_ptr == indexer[8].data_ptr() - raw_ptr
    assert k_pe[2].data_ptr() - raw_ptr == indexer[41].data_ptr() - raw_ptr


def test_dsa_shared_pool_raw_rejects_wrong_bundle_shape():
    raw = torch.empty(128, dtype=torch.int8)

    try:
        reshape_dsa_shared_pool_raw(
            raw,
            torch.float16,
            block_size=128,
            num_kv_heads=1,
            kv_lora_rank=256,
            qk_rope_head_dim=64,
            index_head_dim=128,
            is_indexer=False,
        )
    except AssertionError as exc:
        assert "nine indexer pages" in str(exc)
    else:
        raise AssertionError("expected invalid DSA shared shape to fail")
