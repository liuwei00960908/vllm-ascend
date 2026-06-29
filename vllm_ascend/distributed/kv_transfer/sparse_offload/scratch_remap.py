"""Device-only top-k remap for the DSA latent scratch (Step B2).

Decode reads the latent through two disjoint index spaces resolved by the SAME
per-request block table:

  * positions already saved in LMCache (< lmcache_len) -> compact scratch rows
    [0..n_ret) (the request's first ceil(k/block_size) latent blocks, filled by
    LMCache);
  * decode positions not yet saved in LMCache -> kept in the live decode window
    and read in place. When a decode window is configured, absolute decode
    positions are mapped back to prompt_len + ((pos - prompt_len) % window).

Everything is fixed-shape tensor math: no D2H sync, graph-mode friendly.
"""

import torch


def scratch_remap(
    topk_indices: torch.Tensor,
    prompt_lens: torch.Tensor,
    need_packed: bool = True,
    scratch_base: torch.Tensor | None = None,
    lmcache_lens: torch.Tensor | None = None,
    decode_window_tokens: int = 0,
):
    """Remap absolute top-k indices for the compact-scratch decode path.

    Args:
        topk_indices: [bs, 1, k] (or [bs, k]) absolute token positions selected
            by the indexer; negative entries are padding.
        prompt_lens: [bs] prompt length per decode request. Callers must ensure
            prompt_len >= k for every row (else scratch rows would alias live
            decode positions).
        need_packed: whether to build the LMCache selected-token payload.
        scratch_base: optional [bs] compact scratch base per row. This lets MTP
            rows for the same request use disjoint compact scratch ranges.
        lmcache_lens: optional [bs] absolute token boundary already saved in
            LMCache. Defaults to prompt_lens, preserving the original
            prompt-only sparse decode behavior.
        decode_window_tokens: size of the live decode ring window in tokens.
            0 disables window remapping and preserves original absolute decode
            positions.

    Returns:
        new_indices: same shape as topk_indices. LMCache-selected entries are
            replaced by their compact scratch row (scratch_base + rank in top-k
            order); live decode entries are mapped to the configured resident
            decode window; padding entries stay unchanged.
        selected_packed: [bs, k] int32. LMCache-selected ABSOLUTE positions
            front-packed in top-k order (the LMCache `selected_tokens` rows; row
            i goes to scratch slot i), tail padded with 0.
    """
    orig_shape = topk_indices.shape
    sel = topk_indices.reshape(orig_shape[0], -1)
    k = sel.shape[1]
    plen = prompt_lens.reshape(-1, 1).to(device=sel.device, dtype=sel.dtype)
    if lmcache_lens is None:
        lmc_len = plen
    else:
        lmc_len = lmcache_lens.reshape(-1, 1).to(
            device=sel.device, dtype=sel.dtype
        )
    if scratch_base is None:
        base = torch.zeros((sel.shape[0], 1), dtype=sel.dtype, device=sel.device)
    else:
        base = scratch_base.reshape(-1, 1).to(device=sel.device, dtype=sel.dtype)

    is_lmcache = (sel >= 0) & (sel < lmc_len)
    # Compact rank among LMCache-selected entries, in top-k order. NOTE:
    # torch.cumsum promotes integer dtypes to int64 by default; the sparse FA
    # kernel requires int32 indices, so pin the dtype explicitly.
    rank = torch.cumsum(is_lmcache, dim=1, dtype=sel.dtype) - 1
    new_indices = torch.where(is_lmcache, base + rank, sel)
    if decode_window_tokens > 0:
        window = torch.tensor(
            decode_window_tokens, dtype=sel.dtype, device=sel.device
        )
        is_live_decode = (sel >= lmc_len) & (sel >= plen)
        window_pos = plen + torch.remainder(sel - plen, window)
        new_indices = torch.where(is_live_decode, window_pos, new_indices)

    if not need_packed:
        # `packed` (the front-packed absolute positions) is only consumed by
        # LMCache's selected_tokens; skip its scatter-based build (the heaviest
        # op here) when no connector will read it.
        return new_indices.reshape(orig_shape), None

    # Front-pack the LMCache-selected absolute positions into [bs, k] (+1 trash
    # column so non-LMCache entries scatter harmlessly off the end).
    packed = sel.new_zeros(sel.shape[0], k + 1)
    dst = torch.where(is_lmcache, rank, torch.full_like(rank, k))
    packed.scatter_(1, dst.long(), sel)

    return new_indices.reshape(orig_shape), packed[:, :k].to(torch.int32)
