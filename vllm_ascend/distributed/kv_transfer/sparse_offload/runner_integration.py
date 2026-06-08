# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Glue between the model runner / worker and the DSA latent offload manager.

Kept separate from ``model_runner_v1.py`` so the runner edit stays a few gated
calls and the sizing logic is unit-testable without NPU.

Memory model (see DESIGN.md): both NPU buffers are fixed-size and **subtracted from
the KV-cache budget before the block split**, so the scheduler can never hand out
the memory we use. :func:`compute_reserved_bytes` is what the worker subtracts in
``determine_available_memory``; :func:`allocate_buffers` then allocates exactly that
much, and :func:`build_manager` hands the buffers to the manager (which never
allocates device memory itself).
"""

import torch
from vllm.config import VllmConfig
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size

from vllm_ascend import envs
from vllm_ascend.distributed.kv_transfer.sparse_offload.offload_backend import (
    InMemoryLatentOffloadBackend,
    LatentOffloadBackend,
)
from vllm_ascend.distributed.kv_transfer.sparse_offload.offload_manager import (
    SparseLatentOffloadManager,
    SparseOffloadConfig,
)


def is_dsa_latent_offload_enabled(vllm_config: VllmConfig) -> bool:
    """True iff the env flag is on AND the model is a DSA / v3.2-style sparse model.

    The DSA marker is ``index_topk`` on the HF config (same check the model uses to
    set ``is_v32``). Confirmed present on the GLM5.1 (``glm_moe_dsa``) config
    (``index_topk=2048``).
    """
    if not envs.VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD:
        return False
    hf_config = vllm_config.model_config.hf_text_config
    return hasattr(hf_config, "index_topk")


def config_from_vllm(
    vllm_config: VllmConfig,
    device: torch.device | str = "npu",
) -> SparseOffloadConfig | None:
    """Build a :class:`SparseOffloadConfig` from vLLM config, or None if disabled.

    Field names confirmed against the GLM5.1 (``glm_moe_dsa``) config:
    ``kv_lora_rank=512``, ``qk_rope_head_dim=64``, ``index_topk=2048``,
    ``num_hidden_layers=78``. ``num_hidden_layers`` is used only to *size* the
    decode-store budget; the authoritative per-layer mapping (and whether the MTP
    layer, ``num_nextn_predict_layers``, participates) is established at runtime
    when the manager is built against the actual SFA attention layers.
    """
    if not is_dsa_latent_offload_enabled(vllm_config):
        return None
    hf_config = vllm_config.model_config.hf_text_config
    cache_config = vllm_config.cache_config

    cache_dtype = cache_config.cache_dtype
    if cache_dtype == "auto":
        dtype = vllm_config.model_config.dtype
    else:
        dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]

    return SparseOffloadConfig(
        num_layers=hf_config.num_hidden_layers,
        kv_lora_rank=hf_config.kv_lora_rank,
        qk_rope_head_dim=hf_config.qk_rope_head_dim,
        block_size=cache_config.block_size,
        max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
        topk_tokens=hf_config.index_topk,
        dtype=dtype,
        device=torch.device(device),
    )


def maybe_reserved_bytes(vllm_config: VllmConfig) -> int:
    """KV-budget bytes to reserve for DSA offload buffers (0 if disabled).

    Called from ``determine_available_memory`` so the reservation is subtracted
    before the scheduler computes block counts. Device-independent (size only).
    """
    config = config_from_vllm(vllm_config, device="cpu")
    return compute_reserved_bytes(config) if config is not None else 0


def _load_buffer_rows(config: SparseOffloadConfig) -> int:
    """Worst-case flat prefill tokens loaded per layer (whole batch all-prefill)."""
    return config.max_num_seqs * config.topk_tokens


def compute_reserved_bytes(config: SparseOffloadConfig) -> int:
    """Total NPU bytes to reserve for the scratch pool + load buffer.

    scratch (one layer, reused):   scratch_num_blocks * block_size * latent_dim
    load buffer (one layer):       max_num_seqs * topk_tokens * latent_dim

    Decode-generated latent is NOT reserved here — it stays in the paged latent cache
    (vLLM-managed), so there is no fixed decode-store and no length cap.
    """
    elt = get_dtype_size(config.dtype)
    scratch_slots = config.scratch_num_blocks * config.block_size
    scratch_bytes = scratch_slots * config.latent_dim * elt
    load_buffer_bytes = _load_buffer_rows(config) * config.latent_dim * elt
    return scratch_bytes + load_buffer_bytes


def allocate_buffers(
    config: SparseOffloadConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate scratch (k_nope, k_pe) and the LMCache load buffer.

    Shapes match what :class:`SparseLatentOffloadManager` expects:
        scratch_knope: [scratch_num_blocks, block_size, 1, kv_lora_rank]
        scratch_kpe:   [scratch_num_blocks, block_size, 1, qk_rope_head_dim]
        load_buffer:   [max_num_seqs * topk_tokens, latent_dim]
    """
    scratch_knope = torch.zeros(
        (config.scratch_num_blocks, config.block_size, 1, config.kv_lora_rank),
        dtype=config.dtype,
        device=config.device,
    )
    scratch_kpe = torch.zeros(
        (config.scratch_num_blocks, config.block_size, 1, config.qk_rope_head_dim),
        dtype=config.dtype,
        device=config.device,
    )
    load_buffer = torch.zeros(
        (_load_buffer_rows(config), config.latent_dim),
        dtype=config.dtype,
        device=config.device,
    )
    return scratch_knope, scratch_kpe, load_buffer


def build_manager(
    config: SparseOffloadConfig,
    layer_names: list[str],
    backend: LatentOffloadBackend | None = None,
) -> SparseLatentOffloadManager:
    """Allocate buffers and construct the manager.

    ``backend`` defaults to the in-memory reference backend so the pipeline runs
    end-to-end before the real LMCache adapter is wired in.
    """
    if backend is None:
        # Reference backend (LMCache stand-in). Device from env: "npu" (default) keeps
        # latent in device memory (correctness-only); "cpu" stages it in host RAM to
        # simulate off-NPU LMCache. Swap for the real LMCache adapter when available.
        store_dev = envs.VLLM_ASCEND_DSA_OFFLOAD_BACKEND_DEVICE
        backend = InMemoryLatentOffloadBackend(
            device=config.device if store_dev == "npu" else store_dev
        )
    scratch_knope, scratch_kpe, load_buffer = allocate_buffers(config)
    return SparseLatentOffloadManager(
        config=config,
        backend=backend,
        layer_names=layer_names,
        scratch_knope=scratch_knope,
        scratch_kpe=scratch_kpe,
        load_buffer=load_buffer,
    )
