"""Device-only top-k remap for the DSA latent scratch (Step B2).

Decode reads the latent through two disjoint index spaces resolved by the SAME
per-request block table:

  * prefill-selected positions (< prompt_len) -> compact scratch rows [0..n_ret)
    (the request's first ceil(k/block_size) latent blocks, filled by LMCache);
  * decode-selected positions (>= prompt_len >= k) -> kept ABSOLUTE, read in
    place from their tail blocks. No copy, no [retrieve|decode] assembly.

Everything is fixed-shape tensor math: no D2H sync, graph-mode friendly.
"""

import torch


def scratch_remap(topk_indices: torch.Tensor, prompt_lens: torch.Tensor):
    """Remap absolute top-k indices for the compact-scratch decode path.

    Args:
        topk_indices: [bs, 1, k] (or [bs, k]) absolute token positions selected
            by the indexer; negative entries are padding.
        prompt_lens: [bs] prompt length per decode request. Callers must ensure
            prompt_len >= k for every row (else scratch rows would alias live
            decode positions).

    Returns:
        new_indices: same shape as topk_indices — prefill-selected entries
            replaced by their compact scratch row (rank in top-k order),
            decode-selected / padding entries unchanged.
        selected_packed: [bs, k] int32 — prefill-selected ABSOLUTE positions
            front-packed in top-k order (the LMCache `selected_tokens` rows;
            row i goes to scratch slot i), tail padded with 0.
    """
    orig_shape = topk_indices.shape
    sel = topk_indices.reshape(orig_shape[0], -1)
    k = sel.shape[1]
    plen = prompt_lens.reshape(-1, 1).to(sel.dtype)

    is_pref = (sel >= 0) & (sel < plen)
    # Compact rank among prefill-selected entries, in top-k order. NOTE:
    # torch.cumsum promotes integer dtypes to int64 by default; the sparse FA
    # kernel requires int32 indices, so pin the dtype explicitly.
    rank = torch.cumsum(is_pref, dim=1, dtype=sel.dtype) - 1
    new_indices = torch.where(is_pref, rank, sel)

    # Front-pack the prefill-selected absolute positions into [bs, k] (+1
    # trash column so non-prefill entries scatter harmlessly off the end).
    packed = sel.new_zeros(sel.shape[0], k + 1)
    dst = torch.where(is_pref, rank, torch.full_like(rank, k))
    packed.scatter_(1, dst.long(), sel)

    return new_indices.reshape(orig_shape), packed[:, :k].to(torch.int32)
