# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only introspection probes for DSA latent offload bring-up (Round 1).

Goal: in ONE cheap NPU run, capture the ground-truth facts the off-NPU code can't
confirm (metadata field names, kv_cache tuple layout, ``topk_indices`` shape +
padding sentinel, MLA layer count, ``input_batch`` fields), so the integration
(C/D/E in INTEGRATION.md) can be finalized correctly without burning round-trips.

Everything here is gated by ``VLLM_ASCEND_DSA_OFFLOAD_INTROSPECT``, dumps once per
tag to the log + a file, and is wrapped so a probe can never crash the run.
"""

import torch
from vllm.logger import logger

from vllm_ascend import envs

_dumped: set[str] = set()


def enabled() -> bool:
    return bool(envs.VLLM_ASCEND_DSA_OFFLOAD_INTROSPECT)


def _emit(tag: str, lines: list[str]) -> None:
    if tag in _dumped:
        return
    _dumped.add(tag)
    text = f"\n===== [DSA-INTROSPECT] {tag} =====\n" + "\n".join(lines) + "\n"
    logger.info(text)
    try:
        with open(envs.VLLM_ASCEND_DSA_INTROSPECT_FILE, "a") as f:
            f.write(text)
    except Exception as e:  # never let logging break the run
        logger.warning("DSA introspect file write failed: %s", e)


def _public_attrs(obj) -> list[str]:
    return sorted(a for a in dir(obj) if not a.startswith("_"))


def _t(name: str, x) -> str:
    if isinstance(x, torch.Tensor):
        return f"  {name}: Tensor shape={tuple(x.shape)} dtype={x.dtype} device={x.device}"
    return f"  {name}: {type(x).__name__} = {x!r}"


def probe_metadata_and_kv_cache(attn_metadata, kv_cache) -> None:
    """Dump the SFA attn_metadata fields and the kv_cache tuple layout."""
    if not enabled():
        return
    try:
        lines = [f"attn_metadata type: {type(attn_metadata).__name__}"]
        lines.append("attn_metadata public attrs: " + ", ".join(_public_attrs(attn_metadata)))
        for f in ("num_decodes", "num_prefills", "num_decode_tokens", "num_actual_tokens"):
            lines.append(_t(f, getattr(attn_metadata, f, "<absent>")))
        for f in ("seq_lens", "block_table", "cum_query_lens", "slot_mapping"):
            lines.append(_t(f, getattr(attn_metadata, f, "<absent>")))
        # are our offload fields populated?
        lines.append(_t("req_ids", getattr(attn_metadata, "req_ids", "<absent>")))
        lines.append(_t("prompt_lens", getattr(attn_metadata, "prompt_lens", "<absent>")))
        # kv_cache tuple layout (expect [0]=k_nope,[1]=k_pe,[2]=indexer key,[3]=scale?)
        if isinstance(kv_cache, (list, tuple)):
            lines.append(f"kv_cache: tuple len={len(kv_cache)}")
            for i, t in enumerate(kv_cache):
                lines.append(_t(f"kv_cache[{i}]", t))
        else:
            lines.append(_t("kv_cache", kv_cache))
        _emit("sfa_forward.metadata", lines)
    except Exception as e:
        logger.warning("DSA introspect probe_metadata failed: %s", e)


def probe_topk(topk_indices) -> None:
    """Dump topk_indices shape/dtype and infer the padding sentinel."""
    if not enabled():
        return
    try:
        x = topk_indices
        lines = [_t("topk_indices", x)]
        if isinstance(x, torch.Tensor):
            xl = x.flatten().to(torch.long)
            lines.append(f"  min={int(xl.min())} max={int(xl.max())}")
            neg = int((xl < 0).sum())
            lines.append(f"  count<0 (likely sentinel)={neg}; sample row0={x[0][:16].tolist()}")
            # most frequent value often = the pad sentinel
            vals, cnts = torch.unique(xl, return_counts=True)
            top = vals[int(cnts.argmax())]
            lines.append(f"  most-frequent value={int(top)} (count={int(cnts.max())})")
        _emit("sfa_forward.topk", lines)
    except Exception as e:
        logger.warning("DSA introspect probe_topk failed: %s", e)


def probe_decode_latent(name_nope: str, k_nope, name_pe: str, k_pe) -> None:
    """Dump the current-step latent tensors (so we know what to feed gather_decode)."""
    if not enabled():
        return
    try:
        _emit("sfa_forward.cur_latent", [_t(name_nope, k_nope), _t(name_pe, k_pe)])
    except Exception as e:
        logger.warning("DSA introspect probe_decode_latent failed: %s", e)


def probe_runner(input_batch, mla_layer_names, num_hidden_layers) -> None:
    """Dump MLA layer count vs config and the input_batch fields (req ids / prompt
    lengths source for metadata wiring)."""
    if not enabled():
        return
    try:
        lines = [
            f"num MLA layers (offload targets) = {len(mla_layer_names)}; "
            f"config num_hidden_layers = {num_hidden_layers}",
            f"first/last MLA layer names: {mla_layer_names[:2]} ... {mla_layer_names[-2:]}",
        ]
        if input_batch is not None:
            lines.append("input_batch public attrs: " + ", ".join(_public_attrs(input_batch)))
            for f in ("req_ids", "num_prompt_tokens", "num_computed_tokens_cpu",
                      "num_tokens", "num_reqs"):
                lines.append(_t(f, getattr(input_batch, f, "<absent>")))
        _emit("runner.layers_and_input_batch", lines)
    except Exception as e:
        logger.warning("DSA introspect probe_runner failed: %s", e)
