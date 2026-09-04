import os
from dataclasses import dataclass, field
from functools import lru_cache
from threading import Lock
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

import numpy as np
import scipy  # type: ignore
import torch
import torch_npu
import vllm.envs as envs_vllm
from torch import nn
from vllm.config import CUDAGraphMode, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadataBuilder
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backend import (
    AttentionBackend,  # type: ignore
    AttentionCGSupport,
    MLAAttentionImpl,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.utils import select_common_block_size

from vllm_ascend import envs
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.common_cp import AscendPCPMetadata
from vllm_ascend.attention.mla_v1 import MAX_O_PROJ_PREFETCH_SIZE, MLAPO_MAX_SUPPORTED_TOKENS
from vllm_ascend.attention.utils import (
    SFA_QSFA_TILE_SIZE,
    AscendCommonAttentionMetadata,
    ascend_chunked_prefill_workspace_size,
    enable_cp,
    get_lmcache_sparse_cached_tokens,
    get_sfa_qsfa_packed_head_dim,
    maybe_save_kv_layer_to_connector,
    notify_kv_cache_written,
    staged_sfa_connector_supports_sparse_load,
    trans_rope_weight,
    transdata,
    wait_for_kv_layer_from_connector,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.mxfp_compat import FLOAT8_E8M0FNU_DTYPE
from vllm_ascend.distributed.utils import all_gather_async
from vllm_ascend.memcache_comm_fence import (
    record_attention_compute_start,
)
from vllm_ascend.ops.layer_shard_linear import (
    is_hidden_layer,
    post_process_after_loading_for_shard_weight_series,
    reach_layer_for_shard_weight_series,
    register_all_layers_to_shard_weight_series,
)
from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla
from vllm_ascend.ops.triton.rope import rope_forward_triton_siso
from vllm_ascend.quantization.methods import (
    AscendW8A8DynamicLinearMethod,
    AscendW8A8LinearMethod,
    AscendW8A8MXFP8DynamicLinearMethod,
)
from vllm_ascend.utils import (
    ACL_FORMAT_FRACTAL_ND,
    ACL_FORMAT_FRACTAL_NZ,
    AscendDeviceType,
    _round_up,
    dispose_layer,
    enable_dsa_cp,
    enable_dsa_cp_with_layer_shard,
    enable_dsa_cp_with_o_proj_tp,
    enable_sfa_dcp_replicated_indexer,
    enable_sp,
    get_ascend_device_type,
    get_weight_prefetch_method,
    maybe_trans_nz,
)
from vllm_ascend.worker.npu_input_batch import NPUInputBatch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

# token count limits within bmm_transpose operator
BMM_TRANS_MAX_SUPPORTED_TOKENS = 1024

_lmcache_sparse_wait_sync_once_done = False
_lmcache_sparse_wait_sync_once_lock = Lock()


def _lmcache_sparse_wait_sync_once_enabled() -> bool:
    """One-shot LMCache sparse wait sync gate (B2b env flag)."""
    return bool(envs.VLLM_ASCEND_LMCACHE_SPARSE_WAIT_SYNC_ONCE)


class _ByteGatherPart(NamedTuple):
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    num_bytes_per_row: int


O_PROJ_ACLNN_INPUT_PARAMS = (
    "aclnn_input_scale",
    "aclnn_input_scale_reciprocal",
    "aclnn_input_offset",
)


class DCPGatherContext(NamedTuple):
    """State needed to finish an async fused DCP all-gather."""

    # The gathered fused tensor.
    gathered: torch.Tensor
    # Async all-gather work handle. None means the gather completed synchronously.
    handle: torch.distributed.Work | None
    # Permutation that restores the original dimension order after dim>0 gather.
    restore_perm: tuple[int, ...] | None
    # Last-dimension sizes used to split the fused tensor after gather.
    split_sizes: tuple[int, ...]


def _get_indexer_types(configs: tuple[Any, ...]) -> Any | None:
    for config in configs:
        if config is None:
            continue
        indexer_types = getattr(config, "indexer_types", None)
        if indexer_types is not None:
            return indexer_types
    return None


def _has_shared_indexer_layers(configs: tuple[Any, ...]) -> bool:
    indexer_types = _get_indexer_types(configs)
    if indexer_types is None:
        return False
    return any(isinstance(indexer_type, str) and indexer_type.lower() == "shared" for indexer_type in indexer_types)


def _get_config_bool(configs: tuple[Any, ...], attr: str) -> bool:
    for config in configs:
        if config is not None and hasattr(config, attr):
            return bool(getattr(config, attr))
    return False


class AscendSFABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        # HACK(Ronald1995): vllm `initialize_kv_cache` method in model runner v2 make
        # attention name assertion, we just set name to FLASH_ATTN to avoid assertion error.
        # rectify this when vllm disable the assertion.
        return "ASCEND_SFA" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"

    @staticmethod
    def get_builder_cls():
        if enable_sfa_dcp_replicated_indexer():
            from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPMetadataBuilder

            return AscendSFADCPMetadataBuilder
        if enable_cp():
            from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFACPMetadataBuilder

            return AscendSFACPMetadataBuilder
        return AscendSFAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_type: str = "",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_impl_cls() -> type["AscendSFAImpl"]:
        if enable_sfa_dcp_replicated_indexer():
            from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPImpl

            return AscendSFADCPImpl
        if enable_cp():
            from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFACPImpl

            return AscendSFACPImpl
        return AscendSFAImpl

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128]


@dataclass
class DCPContext:
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    kv_gather_block_ids: torch.Tensor | None = None
    kv_gather_block_table: torch.Tensor | None = None
    gather_context: DCPGatherContext | None = None


@dataclass
class DSACPContext:
    num_tokens: int
    num_tokens_pad: int
    local_start: int
    local_end: int
    local_end_with_pad: int
    slot_mapping_cp: torch.Tensor
    actual_seq_lengths_query: torch.Tensor
    actual_seq_lengths_key: torch.Tensor


@dataclass
class AscendSFAMetadata:
    """Metadata for MLACommon.

    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|
    num_actual_tokens: int  # Number of tokens excluding padding.
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    cum_query_lens: torch.Tensor
    block_table: torch.Tensor
    sin: torch.Tensor
    cos: torch.Tensor

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.
    # The dimension of the attention heads
    head_dim: int | None = None
    attn_mask: torch.Tensor = None
    # chunked prefill by default if no attn_states passed
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill
    dcp_context: DCPContext | None = None
    dsa_cp_context: DSACPContext | None = None
    reshape_cache_event: torch.npu.Event = None
    sfa_cp_metadata: AscendPCPMetadata | None = None
    num_decodes: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    block_size: int = 0
    group_len: torch.Tensor | None = None
    group_key_idx: torch.Tensor | None = None
    group_key_cache_idx: torch.Tensor | None = None
    # DSA two-groups replay (Step 4): the indexer group's own table/slots.
    # None in unbundle-only mode (shared block id space); consumers fall back
    # to block_table / slot_mapping when None.
    indexer_block_table: torch.Tensor | None = None
    indexer_slot_mapping: torch.Tensor | None = None

    # DSA shrink replay (B2c): per-request compact-scratch data plane.
    # decode_req_indices_cpu/tensor map each decode row to its request index
    # (-1 for padding rows); decode_split_boundary_cpu(+tensor) is the
    # builder-owned boundary storage the B2c updater overwrites in place;
    # split_boundary is its device view. decode_target_slot_mapping /
    # decode_selected_tokens / decode_selected_counts carry the remap kernel's
    # outputs for the selective retrieve (populated by B2d). All None for
    # non-shrink modes (official paths untouched).
    # Provenance: fork sfa_v1.py:885-912 (subset; decode_scratch_base fields,
    # union_mapping and the staged-graph buffers are P9/P11+ and not ported).
    req_ids: list[str] | None = None
    prompt_lens: list[int] | None = None
    decode_req_indices: torch.Tensor | None = None
    decode_req_indices_cpu: Any = None
    decode_valid_row_indices: Any = None
    decode_row_offsets: Any = None
    decode_split_boundary_cpu: Any = None
    decode_split_boundary_cpu_tensor: Any = None
    split_boundary: torch.Tensor | None = None
    decode_split_boundary: torch.Tensor | None = None
    decode_target_slot_mapping: torch.Tensor | None = None
    decode_selected_tokens: torch.Tensor | None = None
    decode_selected_counts: torch.Tensor | None = None
    need_sparse_lmcache_payload: bool = False

    # P9 staged graph (Batch 3): fixed-layout data plane. The union
    # workspace is the caller-owned remap kernel workspace (Batch 1
    # dispatch contract); decode_remap_boundary is the stable-address
    # graph-A boundary input (mechanism 2 — host writes, graph reads);
    # prompt_lens_cpu_rows / decode_request_ids_compact feed the staged
    # metadata channels; staged_sfa_payload_validated is the validate-once
    # gate for the P10 payload fast path.
    # Provenance: fork sfa_v1.py:885-912 (staged subset; the
    # decode_scratch_base/compact fields are dead pipes, not ported).
    decode_union_mapping_workspace: torch.Tensor | None = None
    prompt_lens_cpu_rows: Any = None
    decode_remap_boundary: torch.Tensor | None = None
    decode_remap_boundary_ready: bool = False
    decode_request_ids_compact: list[str] | None = None
    staged_sfa_payload_validated: bool = False
    # Scratch capacity for the request-union layout (decode_threshold ×
    # index_topk); _validate_dsa_scratch_capacity reads it defensively via
    # getattr. The row-specific decode_scratch_base fields remain dead
    # pipes (P11+) and are not ported.
    # Provenance: fork sfa_v1.py:903.
    decode_scratch_capacity: int | None = None


M = TypeVar("M", bound=AscendSFAMetadata)


def _staged_prompt_lens_rows(plens: Any, query_width: int) -> np.ndarray:
    """Expand per-request prompt lengths to the fixed-layout row array.

    The staged boundary validator requires one prompt length per token row
    (num_reqs x query_width); slicing the per-request array silently
    truncates for MTP widths greater than one. Provenance: fork
    sfa_v1.py:1276-1280 (fixed-layout reshape).
    """
    plens_np = np.asarray(plens, dtype=np.int32).reshape(-1)
    return np.repeat(plens_np, int(query_width))


def _staged_fixed_layout(
    num_input_tokens: int,
    num_reqs_padded: int,
    num_decode_rows: int,
    active_requests: int,
    query_width: int,
    prompts_all_computed: bool,
) -> bool:
    """Whether the batch carries the padded request-major staged layout.

    Fork semantics (:1256-1266), three conditions: the padded token view
    matches the padded request count (``num_input_tokens == num_reqs *
    width``), EVERY active request contributes exactly ``width`` decode
    rows (``num_decode_rows == active * width`` with ``active`` counting
    all active requests, not just decode ones), and every prompt is fully
    computed (decode phase). Exact-capacity capture batches satisfy all
    three with ``active == num_reqs``; a live batch padded to a captured
    capacity (the common case, e.g. 3 requests on capacity 4) satisfies
    them with fewer active requests — which the historical
    ``num_decode_rows == num_reqs * width`` check wrongly rejected, leaving
    the staged buffers unattached and crashing graph pre.

    The second condition must count ALL active requests: a prefix-hit
    prefill tail (log32: 2 remaining prompt tokens on one request) or a
    mixed batch can coincidentally satisfy the token-view equation, and
    counting only decode requests would let ``0 == 0 * width`` open the
    staged branch for a batch with zero decode rows — the prompt-row
    expansion then crashes on the shape mismatch.
    """
    return (
        num_input_tokens == num_reqs_padded * query_width
        and num_decode_rows == active_requests * query_width
        and prompts_all_computed
    )


class AscendSFAMetadataBuilder(MLACommonMetadataBuilder[AscendSFAMetadata]):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendSFAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            metadata_cls if metadata_cls is not None else AscendSFAMetadata,
            supports_dcp_with_varlen,
        )

        self.block_size = vllm_config.cache_config.block_size
        # Match the logical block size selected for BlockTable.
        self.kernel_block_size = select_common_block_size(kv_cache_spec.block_size, [AscendSFABackend])
        self.max_blocks = (vllm_config.model_config.max_model_len + self.block_size - 1) // self.block_size

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.actual_seq_lengths_query = torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=device)
        self.actual_seq_lengths_key = torch.empty_like(self.actual_seq_lengths_query)
        self.spec_actual_seq_lengths_query: list[torch.Tensor] | None = None
        self.spec_actual_seq_lengths_key: list[torch.Tensor] | None = None
        # Persistent int32 buffers for store_kv_block_metadata inputs, sized to
        # max_num_batched_tokens (matches model_runner_v1._make_buffer sizing).
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.group_len = torch.zeros(max_num_batched_tokens, dtype=torch.int32, device=device)
        self.group_key_idx = torch.zeros(max_num_batched_tokens, dtype=torch.int32, device=device)
        self.group_key_cache_idx = torch.zeros(max_num_batched_tokens, dtype=torch.int32, device=device)
        self.spec_group_len: list[torch.Tensor] | None = None
        self.spec_group_key_idx: list[torch.Tensor] | None = None
        self.spec_group_key_cache_idx: list[torch.Tensor] | None = None
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, (
                f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"
            )
            self.spec_actual_seq_lengths_query = [
                torch.zeros(max_num_reqs * (spec_token_num + 1) + 1, dtype=torch.int32, device=device)
                for _ in range(spec_token_num)
            ]
            self.spec_actual_seq_lengths_key = [
                torch.zeros(max_num_reqs * (spec_token_num + 1) + 1, dtype=torch.int32, device=device)
                for _ in range(spec_token_num)
            ]
            self.spec_group_len = [
                torch.zeros(max_num_batched_tokens, dtype=torch.int32, device=device) for _ in range(spec_token_num)
            ]
            self.spec_group_key_idx = [
                torch.zeros(max_num_batched_tokens, dtype=torch.int32, device=device) for _ in range(spec_token_num)
            ]
            self.spec_group_key_cache_idx = [
                torch.zeros(max_num_batched_tokens, dtype=torch.int32, device=device) for _ in range(spec_token_num)
            ]

        self.reorder_batch_threshold = self.decode_threshold
        self.attn_mask_builder = AttentionMaskBuilder(self.device)
        self.rope_dim = self.model_config.hf_text_config.qk_rope_head_dim
        self.enable_dsa_cp = enable_dsa_cp()

        # DSA shrink replay (B2c): builder-side gate and the scratch
        # structural check. The gate reads UNBUNDLE (fork :959-964); the
        # runner only injects the data plane under two-groups, so an
        # unbundle-without-two-groups configuration stays inert here.
        # decode_threshold > 2 with shrink is rejected (MTP<=2 constraint).
        # Provenance: fork sfa_v1.py:959-982.
        self.dsa_shrink_latent = (
            int(envs.VLLM_ASCEND_DSA_SHRINK_LATENT)
            if bool(envs.VLLM_ASCEND_DSA_UNBUNDLE)
            else 0
        )
        if self.dsa_shrink_latent and self.decode_threshold > 2:
            raise ValueError(
                "DSA shrink-latent compact-scratch decode supports at most "
                f"MTP2 (decode_threshold={self.decode_threshold})."
            )
        self.dsa_index_topk = int(
            getattr(self.model_config.hf_text_config, "index_topk", 0) or 0
        )
        if self.dsa_shrink_latent:
            if self.dsa_index_topk <= 0:
                raise ValueError(
                    "DSA shrink-latent requires the model to define "
                    "index_topk."
                )
            if self.dsa_index_topk % self.kernel_block_size:
                raise ValueError(
                    "DSA index_topk must be an integer multiple of "
                    f"block_size: index_topk={self.dsa_index_topk}, "
                    f"block_size={self.kernel_block_size}."
                )
            # Builder-owned fixed-address per-row storage (fork :984-1059):
            # CPU numpy + device tensors so the boundary updater can rewrite
            # rows without allocating (graph-replay address stability; eager
            # fills the same structures).
            max_rows = max_num_batched_tokens
            self._dsa_max_num_rows = max_rows
            self._dsa_split_boundary_np = np.zeros(max_rows, dtype=np.int32)
            self._dsa_req_indices_np = np.full(
                max_rows, -1, dtype=np.int32
            )
            self._dsa_row_offsets_np = np.zeros(max_rows, dtype=np.int32)
            self._dsa_split_boundary_cpu_tensor = torch.from_numpy(
                self._dsa_split_boundary_np
            )
            self._dsa_req_indices_cpu_tensor = torch.from_numpy(
                self._dsa_req_indices_np
            )
            self._dsa_row_offsets_cpu_tensor = torch.from_numpy(
                self._dsa_row_offsets_np
            )
            self._dsa_split_boundary_tensor = torch.empty(
                max_rows, dtype=torch.int32, device=self.device
            )
            self._dsa_req_indices_tensor = torch.empty(
                max_rows, dtype=torch.int32, device=self.device
            )
            self._dsa_row_offsets_tensor = torch.empty(
                max_rows, dtype=torch.int32, device=self.device
            )

            # P9 staged graph (Batch 3): one builder-owned fixed-address
            # workspace for the staged remap kernel's local_to_union
            # contract (Batch 1 dispatch), and one stable boundary device
            # buffer whose address survives metadata rebuilds so graph A
            # replay always reads from the same location.
            # Provenance: fork sfa_v1.py:1031-1034/:1092-1101 (subset).
            self.scratch_capacity = self.decode_threshold * self.dsa_index_topk
            self._dsa_union_mapping = torch.empty(
                (max_num_reqs, self.scratch_capacity),
                dtype=torch.int32,
                device=self.device,
            )
            self.decode_remap_boundary = torch.empty(
                max_rows, dtype=torch.int32, device=self.device
            )
            # P9 batch 5 fix: the request-union fixed buffers the staged
            # pre-compute requires (selected tokens / counts / target
            # slots). Without them every captured graph A reads None
            # channels. Provenance: fork sfa_v1.py:1033-1041.
            self._dsa_selected_tokens = torch.empty(
                (max_num_reqs, self.scratch_capacity),
                dtype=torch.int32,
                device=self.device,
            )
            self._dsa_selected_counts = torch.empty(
                (max_num_reqs, 16),
                dtype=torch.int32,
                device=self.device,
            )
            self._dsa_target_slots = torch.empty(
                (max_num_reqs, self.scratch_capacity),
                dtype=torch.long,
                device=self.device,
            )

    @staticmethod
    def determine_chunked_prefill_workspace_size(vllm_config: VllmConfig) -> int:
        return ascend_chunked_prefill_workspace_size(vllm_config)

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendSFAMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        return AttentionCGSupport.UNIFORM_BATCH

    def reorder_batch(self, input_batch: "NPUInputBatch", scheduler_output: "SchedulerOutput") -> bool:
        # No need to reorder for Ascend SFA
        return False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendSFAMetadata:
        # common_prefix_len / fast_build are unused; kept for API compatibility.
        return self._build(common_attn_metadata, draft_index=None)

    def build_for_drafting(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        draft_index: int,
        **kwargs,
    ) -> AscendSFAMetadata:
        return self._build(common_attn_metadata, draft_index=draft_index)

    def _build(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        draft_index: int | None = None,
    ) -> AscendSFAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        num_input_tokens = common_attn_metadata.num_input_tokens

        block_table = common_attn_metadata.block_table_tensor[:num_reqs]
        slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
        input_positions = common_attn_metadata.positions[:num_input_tokens].long()

        cum_query_lens = common_attn_metadata.query_start_loc[1 : num_reqs + 1]
        seq_lens = common_attn_metadata.seq_lens[:num_reqs]

        # Prefer _seq_lens_cpu (always available, updated during draft
        # iterations) over seq_lens_cpu (None in async spec decode mode).
        if common_attn_metadata._seq_lens_cpu is not None:
            seq_lens_cpu = common_attn_metadata._seq_lens_cpu[:num_reqs]
        elif common_attn_metadata.seq_lens_cpu is not None:
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu[:num_reqs]
        else:
            seq_lens_cpu = common_attn_metadata.seq_lens[:num_reqs].to("cpu")

        # DSA two-groups mirror: slice the indexer group's table/slots with
        # the same bounds as the latent's (fork F-ascend :1150-1156).
        indexer_block_table = None
        indexer_slot_mapping = None
        if common_attn_metadata.indexer_block_table_tensor is not None:
            indexer_block_table = common_attn_metadata.indexer_block_table_tensor[
                :num_reqs
            ]
            assert common_attn_metadata.indexer_slot_mapping is not None
            indexer_slot_mapping = common_attn_metadata.indexer_slot_mapping[
                :num_input_tokens
            ]

        # DSA shrink replay (B2c): expand the per-row data plane when the
        # runner injected the channels (plens_cpu present => shrink active
        # for this batch). Decode rows get their request index and an
        # initial boundary of the request's prompt length (overwritten
        # in-place each step by _update_dsa_split_boundary_in_place);
        # prefill rows keep -1 / 0. Provenance: fork sfa_v1.py:1185-1444
        # (the fork's general mixed-batch path; the staged-only fixed-layout
        # signature cache and cold-compact resume validation are not ported).
        shrink_kwargs: dict = {}
        plens_cpu = common_attn_metadata.prompt_lens_cpu
        if self.dsa_shrink_latent and plens_cpu is not None:
            if num_input_tokens > self._dsa_max_num_rows:
                raise RuntimeError(
                    "DSA sparse row metadata capacity exceeded: "
                    f"num_input_tokens={num_input_tokens}, "
                    f"max_num_batched_tokens={self._dsa_max_num_rows}."
                )
            req_ids = common_attn_metadata.request_ids
            plens = np.asarray(plens_cpu[:num_reqs], dtype=np.int32)
            computed_cpu = common_attn_metadata.num_computed_tokens_cpu
            if computed_cpu is None:
                # Async speculative decode keeps the authoritative CPU mirror
                # on the underscored field (the public tensor is intentionally
                # None to avoid a blocking copy in the runner).
                computed_cpu = common_attn_metadata._num_computed_tokens_cpu
            if computed_cpu is None:
                raise RuntimeError(
                    "DSA shrink requires num_computed_tokens_cpu (or its "
                    "async-spec CPU mirror) for "
                    "per-row decode layout expansion."
                )
            computed = (
                computed_cpu[:num_reqs].detach().numpy()
                if isinstance(computed_cpu, torch.Tensor)
                else np.asarray(computed_cpu[:num_reqs])
            )
            query_start_cpu = common_attn_metadata.query_start_loc_cpu
            if query_start_cpu is None:
                raise RuntimeError(
                    "DSA shrink requires query_start_loc_cpu for per-row "
                    "request ownership."
                )
            qsl = (
                query_start_cpu[: num_reqs + 1].detach().numpy()
                if isinstance(query_start_cpu, torch.Tensor)
                else np.asarray(query_start_cpu[: num_reqs + 1])
            )

            (
                built_boundaries,
                built_req_indices,
                built_row_offsets,
                num_decode_rows,
                num_decode_reqs,
            ) = _dsa_build_decode_row_metadata(
                qsl,
                plens,
                computed,
                num_input_tokens,
            )
            boundary_rows = self._dsa_split_boundary_np[:num_input_tokens]
            row_req = self._dsa_req_indices_np[:num_input_tokens]
            row_offsets = self._dsa_row_offsets_np[:num_input_tokens]
            boundary_rows[:] = built_boundaries
            row_req[:] = built_req_indices
            row_offsets[:] = built_row_offsets

            self._dsa_split_boundary_tensor[:num_input_tokens].copy_(
                self._dsa_split_boundary_cpu_tensor[:num_input_tokens]
            )
            self._dsa_req_indices_tensor[:num_input_tokens].copy_(
                self._dsa_req_indices_cpu_tensor[:num_input_tokens]
            )
            self._dsa_row_offsets_tensor[:num_input_tokens].copy_(
                self._dsa_row_offsets_cpu_tensor[:num_input_tokens]
            )
            shrink_kwargs = dict(
                req_ids=list(req_ids) if req_ids is not None else None,
                prompt_lens=list(plens),
                decode_req_indices=self._dsa_req_indices_tensor[:num_input_tokens],
                decode_req_indices_cpu=row_req,
                decode_row_offsets=self._dsa_row_offsets_tensor[:num_input_tokens],
                decode_split_boundary_cpu=boundary_rows,
                decode_split_boundary_cpu_tensor=(
                    self._dsa_split_boundary_cpu_tensor[:num_input_tokens]
                ),
                split_boundary=self._dsa_split_boundary_tensor[:num_input_tokens],
                num_decodes=num_decode_reqs,
                num_decode_tokens=num_decode_rows,
                need_sparse_lmcache_payload=(
                    self.dsa_shrink_latent != 3
                    and staged_sfa_connector_supports_sparse_load()
                ),
            )

            # P9 staged graph (Batch 3): wire the staged metadata channels
            # when the batch carries the request-major fixed layout. All
            # three fork conditions must hold: padded view matches padded
            # requests, EVERY active request contributes width decode rows,
            # and every prompt is fully computed (decode phase). The
            # every-active-request form rejects prefix-hit prefill tails
            # and mixed batches whose token counts coincidentally match the
            # fixed-layout equation (log32: a 2-token cached-prompt tail
            # crashed the prompt-row expansion before this hardening).
            # Provenance: fork sfa_v1.py:1256-1266/:1415-1576 (staged
            # subset).
            if _staged_fixed_layout(
                num_input_tokens,
                num_reqs,
                num_decode_rows,
                len(plens),
                self.decode_threshold,
                # computed includes padded request rows while plens only
                # describes active requests. Padding rows are zero and must
                # not make an otherwise valid decode layout look like an
                # unfinished prompt (log53 DP-idle dummy regression).
                bool(np.all(computed[: len(plens)] >= plens)),
            ):
                # Dedicated full-padded prompt-row array (fresh storage, not
                # the split-boundary view also exported below): pad rows
                # stay 0, active rows carry their request's prompt length.
                staged_prompt_rows = np.zeros(
                    num_input_tokens, dtype=np.int32
                )
                staged_prompt_rows[:num_decode_rows] = (
                    _staged_prompt_lens_rows(plens, self.decode_threshold)
                )
                shrink_kwargs.update(
                    decode_union_mapping_workspace=(
                        self._dsa_union_mapping[:num_reqs]
                    ),
                    decode_request_ids_compact=(
                        list(req_ids[:num_reqs])
                        if req_ids is not None
                        else None
                    ),
                    prompt_lens_cpu_rows=staged_prompt_rows,
                    decode_remap_boundary=(
                        self.decode_remap_boundary[:num_input_tokens]
                    ),
                    decode_remap_boundary_ready=False,
                    decode_scratch_capacity=self.scratch_capacity,
                    decode_target_slot_mapping=(
                        self._dsa_target_slots[:num_reqs]
                    ),
                    decode_selected_tokens=(
                        self._dsa_selected_tokens[:num_reqs]
                    ),
                    decode_selected_counts=(
                        self._dsa_selected_counts[:num_reqs]
                    ),
                )

        block_size = self.kernel_block_size

        cos, sin = get_cos_and_sin_mla(input_positions, use_cache=(draft_index is None))

        dsa_cp_context = None
        if self.enable_dsa_cp:
            global_tp_size = get_tp_group().world_size
            num_tokens = num_input_tokens
            num_tokens_pad = _round_up(num_tokens, global_tp_size)
            num_tokens_per_device = num_tokens_pad // global_tp_size
            local_start = get_tp_group().rank_in_group * num_tokens_per_device
            local_end_with_pad = local_start + num_tokens_per_device
            local_end = min(local_end_with_pad, num_actual_tokens)

            pad_size = num_tokens_pad - cos.shape[0]
            assert cos.shape == sin.shape, f"cos.shape must be equal to sin.shape, got {cos.shape} and {sin.shape}"

            if pad_size > 0:
                cos = nn.functional.pad(cos, (0, 0, 0, 0, 0, 0, 0, pad_size))
                sin = nn.functional.pad(sin, (0, 0, 0, 0, 0, 0, 0, pad_size))

            pad_size_slot = num_tokens_pad - slot_mapping.shape[0]
            if pad_size_slot > 0:
                slot_mapping = nn.functional.pad(slot_mapping, (0, pad_size_slot), value=-1)
            else:
                slot_mapping = slot_mapping[:num_tokens_pad]
            slot_mapping_cp = slot_mapping[local_start:local_end_with_pad]

            cos = cos[local_start:local_end_with_pad]
            sin = sin[local_start:local_end_with_pad]

            assert cos.shape[0] == num_tokens_per_device, (
                f"cos.shape[0] must be equal to num_tokens_per_device, \
                    got {cos.shape[0]} and {num_tokens_per_device}"
            )
            assert slot_mapping_cp.shape[0] == num_tokens_per_device, (
                f"slot_mapping_cp.shape[0] must be equal to num_tokens_per_device, \
                    got {slot_mapping_cp.shape[0]} and {num_tokens_per_device}"
            )
            assert slot_mapping.shape[0] == num_tokens_pad, (
                f"slot_mapping.shape[0] must be equal to num_tokens_pad, \
                    got {slot_mapping.shape[0]} and {num_tokens_pad}"
            )

            if draft_index is not None:
                assert self.spec_actual_seq_lengths_query is not None
                assert self.spec_actual_seq_lengths_key is not None
                # Per-draft-step buffers: independent, graph-stable storage so
                # later draft steps don't clobber earlier ones' metadata.
                actual_seq_lengths_query = self.spec_actual_seq_lengths_query[draft_index - 1]
                actual_seq_lengths_key = self.spec_actual_seq_lengths_key[draft_index - 1]
            else:
                actual_seq_lengths_query = self.actual_seq_lengths_query
                actual_seq_lengths_key = self.actual_seq_lengths_key

            num_segs = cum_query_lens.shape[0]

            # Vectorized per-request local query/key lengths for this rank's
            # [local_start, local_end_with_pad) slice. Replaces a Python loop
            # that did 2 .item() NPU->CPU syncs per request (2 * num_reqs
            # syncs/step); now fully on-device with zero syncs.
            # global_start[i] = 0 for i==0, else cum_query_lens[i-1]
            global_start = common_attn_metadata.query_start_loc[:num_segs]
            global_end = cum_query_lens

            # Clip each request's [global_start, global_end) to the local range.
            # num_local_tokens may be < 0 when the request falls entirely
            # outside [local_start, local_end_with_pad); clamp before cumsum.
            req_local_start = global_start.clamp(min=local_start)
            req_local_end = global_end.clamp(max=local_end_with_pad)
            num_local_tokens = req_local_end - req_local_start

            local_query_lens = torch.cumsum(num_local_tokens.clamp(min=0), dim=0)
            offset = global_end - req_local_end  # request tokens on later ranks
            valid_local_req = (num_local_tokens > 0) & (seq_lens > 0)
            local_key_lens = torch.where(
                valid_local_req,
                torch.clamp_min(seq_lens - offset, 0),
                torch.zeros_like(seq_lens),
            )

            actual_seq_lengths_query[:num_segs] = local_query_lens
            actual_seq_lengths_key[:num_segs] = local_key_lens
            actual_seq_lengths_query = actual_seq_lengths_query[:num_reqs]
            actual_seq_lengths_key = actual_seq_lengths_key[:num_reqs]

            dsa_cp_context = DSACPContext(
                num_tokens=num_tokens,
                num_tokens_pad=num_tokens_pad,
                local_start=local_start,
                local_end=local_end,
                local_end_with_pad=local_end_with_pad,
                slot_mapping_cp=slot_mapping_cp,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
            )

        if get_ascend_config().c8_enable_reshape_optim:
            if draft_index is not None:
                assert self.spec_group_len is not None
                assert self.spec_group_key_idx is not None
                assert self.spec_group_key_cache_idx is not None
                group_len = self.spec_group_len[draft_index - 1]
                group_key_idx = self.spec_group_key_idx[draft_index - 1]
                group_key_cache_idx = self.spec_group_key_cache_idx[draft_index - 1]
            else:
                group_len = self.group_len
                group_key_idx = self.group_key_idx
                group_key_cache_idx = self.group_key_cache_idx
            actual_group_len = group_len[:num_input_tokens]
            actual_group_key_idx = group_key_idx[:num_input_tokens]
            actual_group_key_cache_idx = group_key_cache_idx[:num_input_tokens]
            torch.ops._C_ascend.store_kv_block_metadata(
                slot_mapping,
                actual_group_len,
                actual_group_key_idx,
                actual_group_key_cache_idx,
                block_size,
            )
        else:
            actual_group_len = None
            actual_group_key_idx = None
            actual_group_key_cache_idx = None

        return self.metadata_cls(  # type: ignore
            num_input_tokens=common_attn_metadata.num_input_tokens,
            num_actual_tokens=num_actual_tokens,
            cum_query_lens=cum_query_lens,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            slot_mapping=slot_mapping,
            head_dim=self.model_config.get_head_size(),
            attn_mask=self.attn_mask_builder.get_attention_mask(common_attn_metadata.causal, self.model_config),
            attn_state=common_attn_metadata.attn_state,
            block_table=block_table,
            sin=sin[:num_input_tokens],
            cos=cos[:num_input_tokens],
            dsa_cp_context=dsa_cp_context,
            block_size=block_size,
            group_len=actual_group_len,
            group_key_idx=actual_group_key_idx,
            group_key_cache_idx=actual_group_key_cache_idx,
            indexer_block_table=indexer_block_table,
            indexer_slot_mapping=indexer_slot_mapping,
            **shrink_kwargs,
        )

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
    ):
        if attn_state in {AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding}:
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        else:
            raise NotImplementedError("Currently we only support building dummy metadata for DecodeOnly state")

        attn_metadata.attn_state = attn_state
        return attn_metadata


def _dsa_indexer_layer_name(layer_name: str) -> str:
    """Map an inner MLAAttention layer name to its sibling indexer cache.

    ``layer_name`` is the inner name (``...self_attn.attn``); the unbundle
    replay slice stores the indexer key in the sibling
    ``...self_attn.indexer.k_cache`` layer's own KV cache.
    """
    return layer_name.rsplit(".", 1)[0] + ".indexer.k_cache"


def _dsa_index_lmcache_enabled() -> bool:
    """Whether the connector supports the indexer KV namespace specifically.

    Generic sparse-latent support does not imply the sibling indexer
    namespace is loadable; waits/saves for ``...indexer.k_cache`` must be
    gated on this separate capability. Provenance: fork
    sfa_v1.py:781-787 (the diagnostic kill-switch env is not ported).
    """
    from vllm.distributed.kv_transfer import (
        get_kv_transfer_group,
        has_kv_transfer_group,
        is_v1_kv_transfer_group,
    )

    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return False
    connector = get_kv_transfer_group()
    return bool(getattr(connector, "supports_dsa_index_lmcache", False))


# ---------------------------------------------------------------------------
# P9 staged graph: fixed-address bridge buffers and capture state.
# The six bridge slots carry graph A's outputs to the eager retrieve window
# and graph B. Slots 0/1/2 (ql_nope / q_pe / topk_indices) stay inside the
# graphs (mechanism 1); slots 3/4/5 (selected_packed / selected_counts /
# target_slots) must reach the host window (mechanism 3). All buffers are
# allocated once during eager warmup at the maximum capture size; graph
# capture/replay mode raises if they are missing (allocation is a Python
# action and can never be recorded). The capture state seals the addresses
# so replay never writes to replaced tensors (03-4 §7 constraints 1-3).
# Provenance: vllm-ascend-sparse@c7c4a4ac sfa_v1.py:128-218, :2845-2934.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _TensorBinding:
    """Immutable address + layout fingerprint for one captured tensor."""

    address: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device

    @property
    def layout(self) -> tuple[Any, ...]:
        return self.shape, self.stride, self.dtype, self.device

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "_TensorBinding":
        return cls(
            tensor.data_ptr(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device,
        )


@dataclass(frozen=True, slots=True)
class _StagedSFALayerBinding:
    """One graph key's full address contract for a single SFA layer."""

    bridge: tuple[_TensorBinding, ...]
    kv_cache: tuple[_TensorBinding, ...]
    remap_boundary: _TensorBinding
    producer_event_id: int


@dataclass(slots=True)
class _StagedSFACaptureState:
    """Own the immutable capture contract for one local SFA layer."""

    producer_event: Any | None = None
    remap_boundary: torch.Tensor | None = None
    runtime: tuple[Any, ...] | None = None
    initialized_cache_capacity: int = 0
    bindings: dict[Any, _StagedSFALayerBinding] = field(
        default_factory=dict,
    )

    def register(
        self,
        key: Any,
        bridge: tuple[torch.Tensor, ...],
        kv_cache: tuple[torch.Tensor, ...],
    ) -> None:
        if key in self.bindings:
            raise RuntimeError(f"staged SFA graph key was captured twice: {key}")
        if self.producer_event is None or self.remap_boundary is None:
            raise RuntimeError("staged SFA capture storage is incomplete")

        binding = _StagedSFALayerBinding(
            tuple(_TensorBinding.from_tensor(tensor) for tensor in bridge),
            tuple(_TensorBinding.from_tensor(tensor) for tensor in kv_cache),
            _TensorBinding.from_tensor(self.remap_boundary),
            id(self.producer_event),
        )
        if self.bindings:
            existing = next(iter(self.bindings.values()))
            if (
                binding.kv_cache != existing.kv_cache
                or binding.producer_event_id != existing.producer_event_id
                or binding.remap_boundary.address
                != existing.remap_boundary.address
                or binding.remap_boundary.layout[1:]
                != existing.remap_boundary.layout[1:]
                or binding.bridge != existing.bridge
            ):
                raise RuntimeError(
                    "staged SFA capture bindings changed between graph keys"
                )
        self.bindings[key] = binding

    def seal(self, expected_keys: tuple[Any, ...]) -> None:
        expected = frozenset(expected_keys)
        missing = expected.difference(self.bindings)
        unexpected = self.bindings.keys() - expected
        if (
            self.producer_event is None
            or self.remap_boundary is None
            or self.runtime is None
            or missing
            or unexpected
        ):
            raise RuntimeError(
                "staged SFA capture state is incomplete: "
                f"missing_keys={tuple(getattr(key, 'request_capacity', key) for key in missing)}, "
                f"unexpected_keys={tuple(getattr(key, 'request_capacity', key) for key in unexpected)}"
            )


# ---------------------------------------------------------------------------
# DSA shrink replay (B2c): module-level remap support helpers.
# Provenance: vllm-ascend-sparse@c7c4a4ac sfa_v1.py:236-241, :244-250,
# :253-363, :366-395, :704-727. The staged-graph consumers
# (_prepare_sfa_remap_boundary / _validate_dsa_scratch_capacity /
# _fixed_staged_decode_mtp) and _dsa_build_target_slot_mapping are P9/P11+
# and intentionally not ported in this slice.
# ---------------------------------------------------------------------------


def _dsa_topk_to_2d_indices(topk_indices: torch.Tensor) -> torch.Tensor:
    if topk_indices.dim() == 3 and topk_indices.shape[1] == 1:
        return topk_indices[:, 0, :]
    if topk_indices.dim() == 2:
        return topk_indices
    return topk_indices.reshape(topk_indices.shape[0], -1)


def _sync_compute_stream_after_lmcache_sparse_wait() -> None:
    """Fence the first selective sparse load once per worker process."""
    global _lmcache_sparse_wait_sync_once_done

    if (
        not envs.VLLM_ASCEND_LMCACHE_SPARSE_WAIT_SYNC_ONCE
        or _lmcache_sparse_wait_sync_once_done
    ):
        return
    with _lmcache_sparse_wait_sync_once_lock:
        if _lmcache_sparse_wait_sync_once_done:
            return
        if not (
            hasattr(torch, "npu")
            and hasattr(torch.npu, "current_stream")
        ):
            return
        torch.npu.current_stream().synchronize()
        _lmcache_sparse_wait_sync_once_done = True


def _dsa_build_decode_row_metadata(
    query_start_locs: Any,
    prompt_lens: Any,
    computed_tokens: Any,
    num_input_tokens: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Build the eager/mixed shrink row layout using CPU metadata only.

    Returns ``(boundaries, request_indices, row_offsets,
    num_decode_rows, num_decode_requests)``. Prefill/padding rows keep
    boundary=0/request=-1; decode rows carry the request prompt boundary,
    request owner and their offset within this step. This is the fork's
    general mixed-batch path (sfa_v1.py:1343-1405) without the P11+
    cold-compact resume special case; it naturally covers pure decode and
    MTP rows as well, so correctness does not depend on the staged/fixed
    layout optimization.
    """
    plens = np.asarray(prompt_lens, dtype=np.int32)
    computed = np.asarray(computed_tokens, dtype=np.int64)
    qsl = np.asarray(query_start_locs, dtype=np.int64)
    num_reqs = len(plens)
    if len(computed) < num_reqs or len(qsl) < num_reqs + 1:
        raise RuntimeError(
            "DSA shrink row metadata inputs do not cover all requests: "
            f"requests={num_reqs}, computed={len(computed)}, "
            f"query_starts={len(qsl)}."
        )
    if num_reqs and int(qsl[num_reqs]) > num_input_tokens:
        raise RuntimeError(
            "DSA shrink query rows exceed the active input view: "
            f"query_end={int(qsl[num_reqs])}, "
            f"num_input_tokens={num_input_tokens}."
        )

    boundaries = np.zeros(num_input_tokens, dtype=np.int32)
    request_indices = np.full(num_input_tokens, -1, dtype=np.int32)
    row_offsets = np.zeros(num_input_tokens, dtype=np.int32)
    num_decode_rows = 0
    num_decode_requests = 0
    for request_index in range(num_reqs):
        start, end = int(qsl[request_index]), int(qsl[request_index + 1])
        prompt_len = int(plens[request_index])
        first_decode = max(
            start,
            start + prompt_len - int(computed[request_index]),
        )
        if first_decode >= end:
            continue
        num_decode_requests += 1
        boundaries[first_decode:end] = prompt_len
        request_indices[first_decode:end] = request_index
        row_offsets[first_decode:end] = np.arange(
            end - first_decode, dtype=np.int32
        )
        num_decode_rows += end - first_decode
    return (
        boundaries,
        request_indices,
        row_offsets,
        num_decode_rows,
        num_decode_requests,
    )


@lru_cache(maxsize=1)
def _decode_window_save_window_size() -> int:
    value = os.environ.get("LMCACHE_DECODE_WINDOW_SAVE_WINDOW_SIZE", "0")
    try:
        return max(0, int(value or 0))
    except ValueError:
        return 0


def _update_dsa_split_boundary_in_place(
    attn_metadata: Any,
    cached_tokens: list[int] | None,
    decode_window_size: int,
) -> torch.Tensor:
    """Update the builder-owned row boundary without temporary device tensors.

    Each decode row's boundary is overwritten with the proven LMCache
    frontier (min with the decode-window start when window saving is on):
    rows below the boundary read through compact scratch, rows above keep
    their absolute positions. Provenance: fork sfa_v1.py:253-363.
    """
    split_boundary = attn_metadata.split_boundary
    boundary_cpu = attn_metadata.decode_split_boundary_cpu
    boundary_cpu_tensor = attn_metadata.decode_split_boundary_cpu_tensor
    row_req_indices_cpu = attn_metadata.decode_req_indices_cpu
    if split_boundary is None or boundary_cpu is None or boundary_cpu_tensor is None or row_req_indices_cpu is None:
        raise RuntimeError(
            "DSA sparse boundary backing storage is incomplete. Rebuild "
            "attention metadata with the configured max_num_batched_tokens."
        )

    num_rows = int(split_boundary.shape[0])
    if (
        len(boundary_cpu) < num_rows
        or int(boundary_cpu_tensor.shape[0]) < num_rows
        or len(row_req_indices_cpu) < num_rows
    ):
        raise RuntimeError(
            "DSA sparse boundary active view exceeds its backing storage: "
            f"num_rows={num_rows}, boundary_cpu={len(boundary_cpu)}, "
            f"boundary_tensor={int(boundary_cpu_tensor.shape[0])}, "
            f"row_req_indices={len(row_req_indices_cpu)}."
        )

    seq_lens_cpu = attn_metadata.seq_lens_cpu
    num_reqs = int(seq_lens_cpu.shape[0])
    row_req_indices = np.asarray(
        row_req_indices_cpu[:num_rows],
        dtype=np.int32,
    )
    valid_rows = row_req_indices >= 0
    if np.any(row_req_indices[valid_rows] >= num_reqs):
        bad_row = int(
            np.flatnonzero(
                valid_rows & (row_req_indices >= num_reqs)
            )[0]
        )
        raise RuntimeError(
            "DSA sparse row references a request outside seq_lens: "
            f"row={bad_row}, "
            f"request_index={int(row_req_indices[bad_row])}, "
            f"num_reqs={num_reqs}."
        )

    has_cached_frontier = cached_tokens is not None
    if (
        has_cached_frontier
        and decode_window_size <= 0
        and len(cached_tokens) == 0
        and attn_metadata.num_decode_tokens > 0
    ):
        raise RuntimeError(
            "LMCache sparse remap has decode rows but no request boundaries"
        )

    if np.any(valid_rows) and (
        has_cached_frontier or decode_window_size > 0
    ):
        if has_cached_frontier:
            cached_count = min(len(cached_tokens), num_reqs)
            if cached_count == num_reqs:
                request_boundaries = np.asarray(
                    cached_tokens[:cached_count],
                    dtype=np.int32,
                )
            else:
                request_boundaries = np.zeros(
                    num_reqs,
                    dtype=np.int32,
                )
                request_boundaries[:cached_count] = np.asarray(
                    cached_tokens[:cached_count],
                    dtype=np.int32,
                )
        else:
            request_boundaries = np.empty(
                num_reqs,
                dtype=np.int32,
            )
        if decode_window_size > 0:
            if isinstance(seq_lens_cpu, torch.Tensor):
                seq_lens = seq_lens_cpu.detach().numpy()
            else:
                seq_lens = np.asarray(seq_lens_cpu)
            current_positions = np.maximum(
                seq_lens[:num_reqs].astype(np.int64, copy=False) - 1,
                0,
            )
            window_starts = (
                current_positions // decode_window_size
                * decode_window_size
            )
            if has_cached_frontier:
                np.minimum(
                    window_starts,
                    request_boundaries,
                    out=request_boundaries,
                    casting="unsafe",
                )
            else:
                request_boundaries[:] = window_starts
        boundary_cpu[:num_rows][valid_rows] = request_boundaries[
            row_req_indices[valid_rows]
        ]

    split_boundary.copy_(boundary_cpu_tensor[:num_rows])
    attn_metadata.decode_split_boundary = split_boundary
    return split_boundary


def _resolve_sparse_cached_tokens_by_request(
    attn_metadata: Any,
    request_ids: Any,
) -> list[int]:
    """Resolve strict connector frontiers in the native request order.

    Provenance: fork sfa_v1.py:366-395.
    """
    row_req_indices = attn_metadata.decode_req_indices_cpu
    if row_req_indices is None:
        raise RuntimeError(
            "[SFA sparse remap] row/request mapping is unavailable."
        )
    decode_request_indices = sorted(
        {
            int(request_index)
            for request_index in row_req_indices
            if int(request_index) >= 0
        }
    )
    request_ids = list(request_ids) if request_ids is not None else []
    if decode_request_indices and decode_request_indices[-1] >= len(request_ids):
        raise RuntimeError(
            "[SFA sparse remap] active request IDs do not cover all decode "
            "rows."
        )
    decode_request_ids = [
        request_ids[request_index] for request_index in decode_request_indices
    ]
    resolved = get_lmcache_sparse_cached_tokens(decode_request_ids)
    cached_tokens = [0] * int(attn_metadata.seq_lens_cpu.shape[0])
    for request_index, committed_end in zip(
        decode_request_indices, resolved, strict=True
    ):
        cached_tokens[request_index] = int(committed_end)
    return cached_tokens


def _validate_dsa_scratch_capacity(
    boundary_rows: Any,
    row_req_indices: Any,
    scratch_base_rows: Any,
    index_topk: int,
    scratch_capacity: int | None = None,
) -> None:
    """Validate request-level union scratch cannot alias live KV positions.

    scratch_base_rows is kept in the signature for provenance parity with
    the fork; the row-specific base path is a dead pipe (P11+) and is
    never consumed here.

    Provenance: fork sfa_v1.py:534-577.
    """
    width = int(index_topk)
    if width <= 0:
        raise RuntimeError(
            f"DSA compact scratch requires a positive index_topk, got {width}."
        )
    boundaries = np.asarray(boundary_rows, dtype=np.int64).reshape(-1)
    request_rows = np.asarray(row_req_indices, dtype=np.int64).reshape(-1)
    if boundaries.size != request_rows.size:
        raise RuntimeError(
            "DSA compact scratch metadata shapes differ: "
            f"boundaries={boundaries.size}, request_rows={request_rows.size}."
        )
    if scratch_capacity is None or int(scratch_capacity) < width:
        raise RuntimeError(
            "DSA request-union scratch reservation is missing or too small: "
            f"scratch_capacity={scratch_capacity}, index_topk={width}."
        )
    capacity = int(scratch_capacity)
    for request_index in sorted(
        {int(value) for value in request_rows if int(value) >= 0}
    ):
        rows = np.flatnonzero(request_rows == request_index)
        if rows.size * width > capacity:
            raise RuntimeError(
                "DSA request-union scratch reservation is too small: "
                f"request={request_index}, rows={rows.size}, "
                f"index_topk={width}, scratch_capacity={capacity}."
            )
        request_boundaries = boundaries[rows]
        if np.any(
            (request_boundaries != 0)
            & (request_boundaries < capacity)
        ):
            raise RuntimeError(
                "DSA request-union scratch would alias live KV positions: "
                f"request={request_index}, boundaries="
                f"{request_boundaries.tolist()}, "
                f"scratch_capacity={capacity}."
            )


def _prepare_sfa_remap_boundary(
    attn_metadata: Any,
    request_ids: Any,
    *,
    is_dummy_run: bool,
    index_topk: int,
    cached_tokens: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Fill the stable graph-A remap-boundary input once per step.

    Connector metadata and request/row mapping are host objects that cannot
    be frozen into a captured runnable. Resolve them eagerly on CPU, then
    copy the final per-row boundary into the builder-owned NPU tensor whose
    address survives across decode steps.

    Provenance: fork sfa_v1.py:398-530.
    """
    boundary = attn_metadata.decode_remap_boundary
    if boundary is None:
        raise RuntimeError(
            "[SFA sparse remap] boundary storage is unavailable."
        )
    if attn_metadata.decode_remap_boundary_ready:
        return boundary

    prompt_rows = attn_metadata.prompt_lens_cpu_rows
    row_req_indices = attn_metadata.decode_req_indices_cpu
    seq_lens_cpu = attn_metadata.seq_lens_cpu
    if prompt_rows is None or row_req_indices is None or seq_lens_cpu is None:
        raise RuntimeError("[SFA sparse remap] CPU metadata is incomplete.")

    prompt_rows_np = np.asarray(prompt_rows, dtype=np.int32).reshape(-1)
    row_req_indices_np = np.asarray(
        row_req_indices,
        dtype=np.int64,
    ).reshape(-1)
    if isinstance(seq_lens_cpu, torch.Tensor):
        seq_lens = seq_lens_cpu.detach().numpy().reshape(-1)
    else:
        seq_lens = np.asarray(seq_lens_cpu).reshape(-1)
    if (
        int(boundary.numel()) != int(prompt_rows_np.size)
        or row_req_indices_np.size != prompt_rows_np.size
    ):
        raise RuntimeError(
            "[SFA sparse remap] boundary shapes differ: "
            f"boundary={tuple(boundary.shape)}, "
            f"prompt_rows={tuple(prompt_rows_np.shape)}, "
            f"row_req_indices={tuple(row_req_indices_np.shape)}."
        )

    valid_rows = row_req_indices_np >= 0
    decode_request_indices_np = np.unique(row_req_indices_np[valid_rows])
    if (
        decode_request_indices_np.size
        and int(decode_request_indices_np[-1]) >= len(seq_lens)
    ):
        request_index = int(decode_request_indices_np[-1])
        raise RuntimeError(
            "[SFA sparse remap] decode row references request "
            f"{request_index}, but only {len(seq_lens)} sequence lengths "
            "are available."
        )

    cached_tokens_by_request = np.zeros(len(seq_lens), dtype=np.int32)
    cached_request_mask = np.zeros(len(seq_lens), dtype=np.bool_)
    if not is_dummy_run:
        if cached_tokens is None:
            if decode_request_indices_np.size:
                if request_ids is None:
                    raise RuntimeError(
                        "[SFA sparse remap] active request IDs are unavailable."
                    )
                request_ids = list(request_ids)
                if (
                    decode_request_indices_np[-1]
                    >= len(request_ids)
                ):
                    raise RuntimeError(
                        "[SFA sparse remap] active request IDs do not cover "
                        "all decode rows."
                    )
                decode_request_ids = [
                    request_ids[int(index)]
                    for index in decode_request_indices_np
                ]
                resolved_tokens = get_lmcache_sparse_cached_tokens(
                    decode_request_ids
                )
                cached_tokens_by_request[
                    decode_request_indices_np
                ] = np.asarray(resolved_tokens, dtype=np.int32)
                cached_request_mask[decode_request_indices_np] = True
        else:
            if len(cached_tokens) != len(decode_request_indices_np):
                raise RuntimeError(
                    "[SFA sparse remap] cached-token count differs from the "
                    f"active request count: {len(cached_tokens)} vs "
                    f"{len(decode_request_indices_np)}."
                )
            cached_tokens_by_request[
                decode_request_indices_np
            ] = np.asarray(cached_tokens, dtype=np.int32)
            cached_request_mask[decode_request_indices_np] = True

    decode_window_size = _decode_window_save_window_size()
    boundary_rows = prompt_rows_np.copy()
    if decode_request_indices_np.size:
        request_boundaries = np.zeros(len(seq_lens), dtype=np.int32)
        if decode_window_size > 0:
            current_positions = np.maximum(
                seq_lens.astype(np.int64, copy=False) - 1,
                0,
            )
            request_boundaries[:] = (
                current_positions // decode_window_size * decode_window_size
            )
            np.minimum(
                request_boundaries,
                cached_tokens_by_request,
                out=request_boundaries,
                where=cached_request_mask,
            )
        else:
            request_boundaries[cached_request_mask] = (
                cached_tokens_by_request[cached_request_mask]
            )
        rows_with_dynamic_boundary = valid_rows.copy()
        if decode_window_size <= 0:
            rows_with_dynamic_boundary[valid_rows] = (
                cached_request_mask[row_req_indices_np[valid_rows]]
            )
        boundary_rows[rows_with_dynamic_boundary] = (
            request_boundaries[
                row_req_indices_np[rows_with_dynamic_boundary]
            ]
        )

    _validate_dsa_scratch_capacity(
        boundary_rows,
        row_req_indices_np,
        None,
        index_topk,
        getattr(attn_metadata, "decode_scratch_capacity", None),
    )

    boundary.copy_(torch.from_numpy(boundary_rows))
    attn_metadata.decode_remap_boundary_ready = True
    return boundary


def _dsa_mask_padding_sparse_rows(
    topk_indices: torch.Tensor,
    row_req_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep graph padding rows from referencing freed DSA logical blocks.

    Provenance: fork sfa_v1.py:704-727.
    """
    topk_2d = _dsa_topk_to_2d_indices(topk_indices)
    num_rows = int(topk_2d.shape[0])
    if row_req_indices is None:
        return topk_indices, topk_2d
    row_req_indices = row_req_indices[:num_rows].to(device=topk_indices.device)
    if int(row_req_indices.numel()) < num_rows:
        pad = torch.full(
            (num_rows - int(row_req_indices.numel()),),
            -1,
            dtype=row_req_indices.dtype,
            device=topk_indices.device,
        )
        row_req_indices = torch.cat((row_req_indices, pad), dim=0)
    padding_mask = row_req_indices < 0
    if not topk_indices.is_contiguous():
        topk_indices = topk_indices.contiguous()
        topk_2d = _dsa_topk_to_2d_indices(topk_indices)
    topk_2d.masked_fill_(padding_mask.reshape(-1, 1), 0)
    return topk_indices, topk_2d


class AscendSFAImpl(MLAAttentionImpl):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    # Supports forward using the all-gather o_proj weight for decode requests when Sharded CP is enabled.
    o_proj_full_pools: dict[tuple[str, int | None, torch.dtype, int, tuple[int, ...]], torch.Tensor] = {}

    # q_hadamard and k_hadamard tensor shared when dsa c8 enabled
    q_hadamard: torch.Tensor | None = None
    k_hadamard: torch.Tensor | None = None

    # One-shot gate for the VLLM_ASCEND_DSA_DEBUG runtime table evidence.
    _dsa_debug_logged: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        # MLA Args
        self.q_lora_rank = kwargs["q_lora_rank"]
        self.kv_lora_rank = kwargs["kv_lora_rank"]
        self.qk_nope_head_dim = kwargs["qk_nope_head_dim"]
        self.qk_rope_head_dim = kwargs["qk_rope_head_dim"]
        self.qk_head_dim = kwargs["qk_head_dim"]
        self.v_head_dim = kwargs["v_head_dim"]
        self.rotary_emb = kwargs["rotary_emb"]
        self.q_proj = kwargs["q_proj"] if self.q_lora_rank is None else kwargs["q_b_proj"]
        self.fused_qkv_a_proj = kwargs.get("fused_qkv_a_proj")
        self.kv_b_proj = kwargs["kv_b_proj"]
        self.o_proj = kwargs["o_proj"]
        self.indexer = kwargs["indexer"]
        self.kv_a_proj_with_mqa = kwargs.get("kv_a_proj_with_mqa")
        self.kv_a_layernorm = kwargs.get("kv_a_layernorm")
        self.q_a_layernorm = kwargs.get("q_a_layernorm")
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tp_group().rank_in_group
        self.q_b_proj = kwargs["q_b_proj"]
        self.skip_topk = kwargs.get("skip_topk", False)
        self.topk_indices_buffer = kwargs.get("topk_indices_buffer")

        ascend_config = get_ascend_config()
        self.vllm_config = get_current_vllm_config()
        kv_transfer_config = self.vllm_config.kv_transfer_config
        self.is_kv_producer = kv_transfer_config is not None and kv_transfer_config.is_kv_producer
        self.is_kv_consumer = kv_transfer_config is not None and kv_transfer_config.is_kv_consumer

        self.sfa_qsfa_tile_size = SFA_QSFA_TILE_SIZE
        self.sfa_qsfa_packed_kv_head_dim = 0

        # DSA shrink replay (B2d): impl-side gate (UNBUNDLE, fork :1698-1708)
        # and the MTP<=2 combination guard. The runner only injects the data
        # plane under two-groups, so the flag alone never enables a path.
        self.dsa_shrink_latent = (
            int(envs.VLLM_ASCEND_DSA_SHRINK_LATENT)
            if bool(envs.VLLM_ASCEND_DSA_UNBUNDLE)
            else 0
        )
        decode_threshold = 1 + int(
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config is not None
            else 0
        )
        # P9 batch 5: the staged custom-op call site reads the row width from
        # the implementation. Provenance: fork sfa_v1.py:1709-1714.
        self.decode_threshold = decode_threshold
        if self.dsa_shrink_latent:
            if decode_threshold > 2:
                raise ValueError(
                    "DSA shrink-latent compact-scratch decode supports at "
                    f"most MTP2 (decode_threshold={decode_threshold})."
                )
        self.dsa_index_topk = int(
            getattr(
                self.vllm_config.model_config.hf_text_config,
                "index_topk",
                0,
            )
            or 0
        )
        self.dsa_scratch_capacity = decode_threshold * self.dsa_index_topk
        # P9 batch 4: the impl consumes the SAME shared predicate and
        # validated capture-size list as the platform and the runner (token
        # capacities). An empty tuple when staged is not configured
        # (fail-closed for bridge allocation, leaves staged=0 untouched).
        # Provenance: fork sfa_v1.py:1709-1714.
        from vllm_ascend.utils import (
            staged_sfa_graph_capture_sizes,
            staged_sfa_graph_configured,
        )

        self.enable_staged_sfa_graph = staged_sfa_graph_configured(
            self.vllm_config
        )
        self._staged_sfa_graph_capture_sizes: tuple[int, ...] = (
            staged_sfa_graph_capture_sizes(self.vllm_config)
        )
        self._staged_sfa_capture_state = _StagedSFACaptureState()
        self._staged_sfa_bridge_buffers: tuple[torch.Tensor, ...] | None = None
        self.sfa_qsfa_k_nope_clip_alpha: torch.Tensor | None = None
        self.sfa_qsfa_kr_cache_dummy: torch.Tensor | None = None

        self.local_num_heads = self.num_heads
        self.layer_name = kwargs.get("layer_name")
        hf_config = self.vllm_config.model_config.hf_config
        hf_text_config = getattr(self.vllm_config.model_config, "hf_text_config", None)
        config_candidates = (hf_config, hf_text_config)
        self.index_cache_enabled = _get_config_bool(
            config_candidates,
            "use_index_cache",
        ) or _has_shared_indexer_layers(config_candidates)
        self.use_index_cache = self.skip_topk or self.index_cache_enabled
        self.has_indexer = self.indexer is not None
        if not self.has_indexer and not self.skip_topk:
            raise ValueError(
                "Indexer is required for DSA unless skip_topk is enabled. "
                f"Got indexer=None, skip_topk={self.skip_topk}, "
                f"layer_name={self.layer_name}."
            )
        if not self.has_indexer and self.topk_indices_buffer is None:
            raise ValueError(
                "topk_indices_buffer is required when indexer is None and "
                f"skip_topk is enabled. layer_name={self.layer_name}."
            )
        # indexer param
        if self.has_indexer:
            self.n_head: int = self.indexer.n_head  # 64
            self.head_dim: int = self.indexer.head_dim  # 128
            self.wq_b = self.indexer.wq_b
            self.wk_weights_proj = self.indexer.wk_weights_proj
            self.k_norm = self.indexer.k_norm
        else:
            self.n_head = getattr(hf_config, "index_n_heads", 0)
            self.head_dim = getattr(hf_config, "index_head_dim", 0)
            self.wq_b = None
            self.wk_weights_proj = None
            self.k_norm = None
        self.cp_size = 1
        self.is_rope_neox_style = True
        self.use_torch_npu_lightning_indexer = False
        if self.vllm_config.model_config.hf_config.model_type in ["glm_moe_dsa"]:
            self.is_rope_neox_style = False
            self.use_torch_npu_lightning_indexer = True

        # Sparse C8 has two independent meanings in SFA:
        # - SFA packed KV cache for npu_kv_quant_sparse_flash_attention.
        # - C8 indexer cache for lightning indexer.
        # The user-facing switches control these layouts independently, and
        # layers without an indexer only apply the SFA setting.
        self.enable_sparse_sfa_c8 = ascend_config.enable_sparse_sfa_c8
        self.enable_sparse_li_c8 = self.has_indexer and ascend_config.is_sparse_li_c8_layer(self.layer_name)
        if self.enable_sparse_sfa_c8 or self.enable_sparse_li_c8:
            if get_ascend_device_type() == AscendDeviceType.A5:
                self.c8_k_cache_dtype = torch.float8_e4m3fn
                self.c8_k_scale_cache_dtype = torch.float32
            else:
                self.c8_k_cache_dtype = torch.int8
                self.c8_k_scale_cache_dtype = torch.float16

        # DSA unbundle replay (Step 2 / A2b): the MLA layer owns a latent-only
        # 2-tuple while the sibling DeepseekV32IndexerCache layer owns the
        # indexer key as its own 1-tuple. Re-assemble a 3-tuple in forward so
        # the indexer read/write (kv_cache[2]) work unchanged (fork semantics
        # vllm-ascend-sparse@c7c4a4ac sfa_v1.py:3220-3238; official final has
        # no virtual_engine, so the sibling cache is taken as-is).
        self.dsa_unbundle = bool(envs.VLLM_ASCEND_DSA_UNBUNDLE)
        self._dsa_idx_cache_t = None

        if self.enable_sparse_sfa_c8:
            self.sfa_qsfa_packed_kv_head_dim = get_sfa_qsfa_packed_head_dim(
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                self.sfa_qsfa_tile_size,
            )
        # PD decode consumers with sparse C8 can use mla_prolog_v3 to write the packed KV cache.
        # TODO: Re-enable after the community CANN baseline upgrades from 9.0 to 9.1.
        # npu_mla_prolog_v3 depends on CANN 9.1 and is not available in the current community CANN 9.0.
        cann_version = getattr(torch.version, "cann", None)
        if cann_version is not None:
            from packaging.version import Version

            sfa_prolog_v3_supported = Version(cann_version) >= Version("9.1.0")
        else:
            sfa_prolog_v3_supported = False
        self.enable_sfa_prolog_v3 = (
            sfa_prolog_v3_supported
            and self.is_kv_consumer
            and self.enable_sparse_sfa_c8
            and get_ascend_device_type() != AscendDeviceType.A5
        )
        self.enable_mlapo = ascend_config.enable_mlapo and not (
            self.enable_sfa_prolog_v3 or (self.enable_sparse_sfa_c8 and get_ascend_device_type() != AscendDeviceType.A5)
        )

        # Effective in SFA when FlashComm is enabled.
        self.enable_dsa_cp = enable_dsa_cp()
        self.enable_sp = enable_sp()

        # Enable layer sharding via DSA-CP on the P node in the PD-disaggregated setup.
        self.enable_dsa_cp_with_layer_shard = enable_dsa_cp_with_layer_shard()

        # SFA DSA-CP mixed deployments keep o_proj in the existing TP layout.
        # Decode can use the TP-sharded o_proj directly after an activation
        # all-to-all, while prefill/mixed batches temporarily gather the TP
        # shards into a full-weight buffer because their SFA output is not
        # TP-sharded. This is part of the DSA-CP mixed-mode data path rather
        # than an independent user-facing feature switch.
        self.enable_dsa_cp_with_o_proj_tp = enable_dsa_cp_with_o_proj_tp()

        if self.enable_dsa_cp:
            self.local_num_heads = self.num_heads * self.tp_size
            if self.enable_dsa_cp_with_layer_shard:
                self.layer_sharding_kwargs = []
                for layer_name in get_ascend_config().layer_sharding or []:
                    if layer_name in kwargs:
                        self.layer_sharding_kwargs.append(kwargs[layer_name])
                    else:
                        logger.warning_once(
                            f"Layer '{layer_name}' not found in kwargs, skipping sharding. "
                            f"Check layer_sharding config and model layer names."
                        )
                register_all_layers_to_shard_weight_series(self.layer_sharding_kwargs)

    def _cross_layer_empty_outputs(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Return the fixed bridge tuple used by staged custom-op fakes."""
        return self._ensure_staged_sfa_bridge_buffers(hidden_states)

    def reset_staged_sfa_capture(self) -> None:
        """Discard one capture generation before eager warmup rebuilds it."""
        self._staged_sfa_capture_state = _StagedSFACaptureState()
        self._dsa_idx_cache_t = None
        self._staged_sfa_bridge_buffers = None

    def seal_staged_sfa_capture(
        self,
        graph_keys: tuple[Any, ...],
    ) -> None:
        """Seal all expected graph keys after capture completes."""
        self._staged_sfa_capture_state.seal(graph_keys)

    def _ensure_staged_sfa_bridge_buffers(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Allocate the six fixed-address staged bridge slots in warmup.

        Allocation is deliberately forbidden once graph capture/replay starts:
        Python allocation is not replayed, so a late tensor would either be
        absent from the graph or replace the address recorded by capture.
        """
        buffers = self._staged_sfa_bridge_buffers
        capture_sizes = self._staged_sfa_graph_capture_sizes
        if not capture_sizes:
            raise RuntimeError(
                "staged SFA graph capture sizes are unavailable; configure "
                "them before eager warmup allocates bridge storage"
            )
        max_tokens = int(capture_sizes[-1])
        decode_threshold = 1 + int(
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config is not None
            else 0
        )
        if decode_threshold <= 0 or max_tokens % decode_threshold != 0:
            raise RuntimeError(
                "staged SFA maximum token capacity must be divisible by the "
                "decode row width: "
                f"max_tokens={max_tokens}, rows={decode_threshold}"
            )
        max_requests = max_tokens // decode_threshold
        scratch_capacity = decode_threshold * self.dsa_index_topk
        if buffers is None:
            context = get_forward_context()
            if (
                getattr(context, "cudagraph_runtime_mode", CUDAGraphMode.NONE)
                != CUDAGraphMode.NONE
            ):
                raise RuntimeError(
                    "staged SFA bridge storage was not allocated by eager "
                    "warmup before graph capture/replay"
                )
            buffers = (
                hidden_states.new_empty(
                    (
                        max_tokens,
                        self.local_num_heads,
                        self.kv_lora_rank,
                    )
                ),
                hidden_states.new_empty(
                    (
                        max_tokens,
                        self.local_num_heads,
                        self.qk_rope_head_dim,
                    )
                ),
                torch.empty(
                    (max_tokens, 1, self.dsa_index_topk),
                    dtype=torch.int32,
                    device=hidden_states.device,
                ),
                torch.empty(
                    (max_requests, scratch_capacity),
                    dtype=torch.int32,
                    device=hidden_states.device,
                ),
                torch.empty(
                    (max_requests,),
                    dtype=torch.int32,
                    device=hidden_states.device,
                ),
                torch.empty(
                    (max_requests, scratch_capacity),
                    dtype=torch.long,
                    device=hidden_states.device,
                ),
            )
            self._staged_sfa_bridge_buffers = buffers
        if any(tensor.device != hidden_states.device for tensor in buffers):
            raise RuntimeError(
                "staged SFA bridge storage moved to a different device"
            )
        if buffers[0].dtype != hidden_states.dtype:
            raise RuntimeError(
                "staged SFA bridge storage dtype differs from hidden states"
            )
        return buffers

    def _copy_to_staged_sfa_bridge(
        self,
        hidden_states: torch.Tensor,
        outputs: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Copy graph-A outputs into their fixed bridge slots."""
        buffers = self._ensure_staged_sfa_bridge_buffers(hidden_states)
        if len(outputs) != len(buffers):
            raise RuntimeError(
                "staged SFA pre returned an unexpected bridge arity: "
                f"{len(outputs)}"
            )
        for source, destination in zip(outputs, buffers, strict=True):
            rows = int(source.shape[0])
            if (
                rows > int(destination.shape[0])
                or tuple(source.shape[1:])
                != tuple(destination.shape[1:])
            ):
                raise RuntimeError(
                    "staged SFA bridge output exceeds its fixed storage: "
                    f"source={tuple(source.shape)}, "
                    f"destination={tuple(destination.shape)}"
                )
            destination[:rows].copy_(source)
        return buffers

    def _cross_layer_kv_cache(
        self,
        layer_name: str,
        kv_cache: tuple[torch.Tensor, ...],
    ) -> tuple[tuple[torch.Tensor, ...], str | None, bool]:
        """Assemble the 3-tuple KV (latent-nope, latent-pe, indexer).

        Reuses the baseline's cached indexer lookup (``_dsa_idx_cache_t``)
        for the unbundle mode. Provenance: fork sfa_v1.py:2808-2827.
        """
        index_layer_name = (
            _dsa_indexer_layer_name(layer_name) if self.dsa_unbundle else None
        )
        index_enabled = bool(
            index_layer_name is not None and _dsa_index_lmcache_enabled()
        )
        if self.dsa_unbundle and len(kv_cache) < 3:
            index_cache = getattr(self, "_dsa_idx_cache_t", None)
            if index_cache is None:
                context = get_forward_context()
                assert index_layer_name is not None
                # Official ForwardContext has no virtual_engine (the fork
                # indexed kv_cache by it); take the sibling cache as-is —
                # the same resolution as the native unbundle replay path
                # (see _dsa_reassemble_kv_cache below).
                registered = (
                    context.no_compile_layers[index_layer_name].kv_cache
                )
                index_cache = (
                    registered[0]
                    if isinstance(registered, (tuple, list))
                    else registered
                )
                self._dsa_idx_cache_t = index_cache
            kv_cache = (*kv_cache, index_cache)
        return kv_cache, index_layer_name, index_enabled

    def _indexer_topk_for_staged(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        cos: torch.Tensor,
        sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        indexer_block_table: torch.Tensor,
    ) -> torch.Tensor:
        """Run the indexer top-k for the staged pre-compute path.

        .. deprecated::
            Historical helper that passed the key projection as the query
            with ``weights=None`` — semantically invalid. Kept only as a
            fail-loud tombstone; the staged pre-compute now calls the
            production :meth:`indexer_select_post_process` with the
            block-table override. Provenance: fork sfa_v1.py:2730-2740.
        """
        raise NotImplementedError(
            "the staged indexer top-k must go through "
            "indexer_select_post_process with indexer_block_table_override; "
            "this helper computed an invalid query and is retained only to "
            "fail loudly if anything still calls it"
        )

    def _cross_layer_pre_compute(
        self,
        hidden_states: torch.Tensor,
        kv_cache_nope: torch.Tensor,
        kv_cache_pe: torch.Tensor,
        indexer_cache: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        slot_mapping: torch.Tensor,
        indexer_slot_mapping: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        indexer_block_table: torch.Tensor,
        remap_boundary: torch.Tensor,
        row_req_indices: torch.Tensor,
        request_block_table: torch.Tensor,
        selected_packed: torch.Tensor,
        selected_counts: torch.Tensor,
        target_slot_mapping: torch.Tensor,
        local_to_union_workspace: torch.Tensor,
        attn_metadata: M,
    ) -> tuple[torch.Tensor, ...]:
        """Pre-retrieval compute captured by the outer PIECEWISE graph.

        Provenance: fork sfa_v1.py:2668-2771. Adaptations from the fork:
        exec_kv takes an extra attn_metadata parameter; indexer top-k goes
        through _indexer_topk_for_staged (attn_metadata positional arg);
        index_topk is self.dsa_index_topk; block_size is read from config.
        """
        assert self.fused_qkv_a_proj is not None
        assert self.q_a_layernorm is not None

        # Overlap the weight fetch with the preceding compute inside the
        # captured graph A. Provenance: fork sfa_v1.py:2694-2698.
        weight_prefetch_method = get_weight_prefetch_method()
        weight_prefetch_method.maybe_prefetch_mla_or_sla_weight_in_current_stream(
            inputs=self.fused_qkv_a_proj.weight,
            dependency=hidden_states,
        )
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_no_split = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            dim=-1,
        )
        q_c = self.q_a_layernorm(q_c)
        k_li, _ = self.indexer_select_pre_process(
            x=hidden_states,
            cos=cos,
            sin=sin,
        )
        kv_cache = (kv_cache_nope, kv_cache_pe, indexer_cache)
        self.exec_kv(
            kv_no_split,
            cos,
            sin,
            kv_cache,
            slot_mapping,
            attn_metadata,
        )

        ql_nope, q_pe = self._q_proj_and_k_up_proj(q_c)
        q_pe = self.rope_single(q_pe, cos, sin)

        torch_npu.npu_scatter_nd_update_(
            indexer_cache.view(-1, k_li.shape[-1]),
            indexer_slot_mapping.view(-1, 1),
            k_li.view(-1, k_li.shape[-1]),
        )
        # Production indexer top-k (weights from hidden states, multi-head
        # query from q_c, RoPE) with the fixed-layout block-table override
        # and the model's index_topk as the kernel width. The historical
        # bug passed the key projection as the query with weights=None.
        # Provenance: fork sfa_v1.py:2730-2740.
        topk_indices = self.indexer_select_post_process(
            x=hidden_states,
            q_c=q_c,
            kv_cache=kv_cache,
            attn_metadata=None,
            cos=cos,
            sin=sin,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            indexer_block_table_override=indexer_block_table,
            sparse_count=self.dsa_index_topk,
        )

        from vllm_ascend.distributed.kv_transfer.sparse_offload.prepare_sparse_indices import (
            prepare_sparse_indices,
        )

        row_count = int(topk_indices.shape[0])
        request_count = int(request_block_table.shape[0])
        staged_mtp = row_count // max(request_count, 1)
        block_size = int(self.vllm_config.cache_config.block_size)

        (
            topk_indices,
            selected_packed,
            selected_counts,
            target_slot_mapping,
        ) = prepare_sparse_indices(
            topk_indices,
            remap_boundary,
            request_block_table,
            block_size,
            hidden_states.device,
            row_req_indices=row_req_indices,
            selected_packed=selected_packed,
            selected_counts=selected_counts,
            target_slot_mapping=target_slot_mapping,
            local_to_union_workspace=local_to_union_workspace,
            staged_mtp=staged_mtp,
        )
        return (
            ql_nope,
            q_pe,
            topk_indices,
            selected_packed,
            selected_counts,
            target_slot_mapping,
        )

    def _cross_layer_post_compute(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        topk_indices: torch.Tensor,
        kv_cache_nope: torch.Tensor,
        kv_cache_pe: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        block_table: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: M,
    ) -> torch.Tensor:
        """Post-retrieval compute captured by the outer PIECEWISE graph.

        Provenance: fork sfa_v1.py:2773-2808. The FA call passes the
        caller-provided attn_metadata (baseline requires it; fork passed
        a lightweight stub — we keep the real metadata since the fixed
        layout still carries per-request block tables and seq lens).
        """
        kv_cache = (kv_cache_nope, kv_cache_pe)
        attn_output = self._execute_sparse_flash_attention_process(
            ql_nope,
            q_pe,
            kv_cache,
            topk_indices,
            attn_metadata,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
        )
        attn_output = self._v_up_proj(attn_output)
        # Overlap the o_proj weight fetch with the v_up_proj compute inside
        # the captured graph B. Provenance: fork sfa_v1.py:2800-2806.
        weight_prefetch_method = get_weight_prefetch_method()
        weight_prefetch_method.maybe_prefetch_mla_or_sla_weight_in_current_stream(
            inputs=self.o_proj.weight,
            dependency=attn_output,
            max_size=MAX_O_PROJ_PREFETCH_SIZE,
            linear_layer=self.o_proj,
        )
        output[...] = self.o_proj(attn_output)[0]
        return output

    def cross_layer_lmcache_retrieve(
        self,
        layer_name: str,
        next_layer_name: str,
        selected_packed: torch.Tensor,
        selected_counts: torch.Tensor,
        target_slots: torch.Tensor,
        attn_metadata: M | None,
        context: Any,
    ) -> None:
        """Eager retrieve window between graph A and graph B.

        Provenance: fork sfa_v1.py:3074-3129.
        """
        graph_key = getattr(context, "staged_sfa_graph_key", None)
        if attn_metadata is None or graph_key is None:
            return
        if getattr(context, "staged_sfa_graph_dummy_run", False):
            if next_layer_name:
                next_metadata = context.attn_metadata[next_layer_name]
                _prepare_sfa_remap_boundary(
                    next_metadata,
                    next_metadata.req_ids,
                    is_dummy_run=True,
                    index_topk=self.dsa_index_topk,
                )
            return

        route = context.staged_sfa_route
        state = self._staged_sfa_capture_state
        index_enabled = bool(state.runtime and state.runtime[3])
        producer_event = state.producer_event
        if producer_event is not None:
            attn_metadata.reshape_cache_event = producer_event

        request_ids = attn_metadata.decode_request_ids_compact
        if request_ids is None:
            raise RuntimeError("staged SFA request ids are unavailable")
        request_count = len(request_ids)
        wait_for_kv_layer_from_connector(
            layer_name,
            selected_tokens=selected_packed[:request_count],
            token_start_index=None,
            request_ids=request_ids,
            target_slot_mapping=target_slots[:request_count],
            selected_token_counts=selected_counts[:request_count],
            payload_event=producer_event,
        )
        if (
            _lmcache_sparse_wait_sync_once_enabled()
            and not _lmcache_sparse_wait_sync_once_done
        ):
            _sync_compute_stream_after_lmcache_sparse_wait()
        if next_layer_name:
            next_metadata = context.attn_metadata[next_layer_name]
            _prepare_sfa_remap_boundary(
                next_metadata,
                next_metadata.req_ids,
                is_dummy_run=False,
                index_topk=self.dsa_index_topk,
                # route.frontiers is already the immutable frontier tuple;
                # a nested lookup would silently discard it (route propagation
                # regression). Provenance: fork sfa_v1.py:3120-3127.
                cached_tokens=(
                    getattr(route, "frontiers", None)
                    if route is not None
                    else None
                ),
            )
            if index_enabled:
                wait_for_kv_layer_from_connector(
                    _dsa_indexer_layer_name(next_layer_name)
                )

    def bootstrap_cross_layer(self, layer_name: str) -> None:
        """Prepare layer zero before the first captured island is launched.

        Provenance: fork sfa_v1.py:3131-3151.
        """
        context = get_forward_context()
        metadata = context.attn_metadata[layer_name]
        is_dummy = bool(
            getattr(context, "staged_sfa_graph_dummy_run", False)
        )
        _prepare_sfa_remap_boundary(
            metadata,
            metadata.req_ids,
            is_dummy_run=is_dummy,
            index_topk=self.dsa_index_topk,
            # Two legitimate hops: context.staged_sfa_route is the decision
            # object and .frontiers is its tuple field (unlike the retrieve
            # path where route was already the decision).
            cached_tokens=(
                None
                if is_dummy
                else getattr(
                    getattr(context, "staged_sfa_route", None),
                    "frontiers",
                    None,
                )
            ),
        )
        runtime = self._staged_sfa_capture_state.runtime
        if (
            not is_dummy
            and runtime
            and runtime[2] is not None
            and runtime[3]
        ):
            wait_for_kv_layer_from_connector(runtime[2])

    def submit_cross_layer_save(self) -> None:
        """Trigger cross-layer save after graph B completes.

        Provenance: fork sfa_v1.py:3180-3190.
        """
        runtime = self._staged_sfa_capture_state.runtime
        if runtime is None:
            return
        layer_name, kv_cache, index_layer_name, index_enabled = runtime
        if (
            bool(self.dsa_shrink_latent)
            and _decode_window_save_window_size() == 0
        ):
            return
        maybe_save_kv_layer_to_connector(
            layer_name, [kv_cache[0], kv_cache[1]]
        )
        if index_layer_name is not None and index_enabled:
            maybe_save_kv_layer_to_connector(index_layer_name, [kv_cache[2]])

    def cross_layer_graph_pre(
        self,
        layer_name: str,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: M | None,
        need_gather_q_kv: bool,
        output: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Run graph A, or the complete native path outside staged replay.

        Provenance: fork sfa_v1.py:2936-3075. The eligibility/route checks
        (~50 lines in the fork) belong to the batch-5 routing layer and
        are intentionally not ported here; this wrapper runs the eager
        warmup orchestration that Batch 5 will gate behind its own checks.
        """
        context = get_forward_context()
        if attn_metadata is None:
            self.forward(
                layer_name,
                hidden_states,
                kv_cache,
                attn_metadata,
                need_gather_q_kv,
                output,
            )
            return self._cross_layer_empty_outputs(hidden_states)

        kv_cache, index_layer_name, index_enabled = (
            self._cross_layer_kv_cache(layer_name, kv_cache)
        )
        graph_key = getattr(context, "staged_sfa_graph_key", None)
        if graph_key is None:
            self.forward(
                layer_name,
                hidden_states,
                kv_cache,
                attn_metadata,
                need_gather_q_kv,
                output,
            )
            return self._cross_layer_empty_outputs(hidden_states)

        # P9 Batch 5: simplified eligibility check — the runner authorized
        # a graph key, so this step has the fixed decode layout; verify the
        # impl-side invariants that the runner cannot see. C8 (both SFA
        # packed-KV and indexer variants), MLAPO, and DSA-CP all build
        # layouts the staged fixed data plane does not represent and must
        # be rejected here — they exist on this baseline.
        # Provenance: fork sfa_v1.py:2390-2608 (~25 conditions → the
        # baseline-applicable subset).
        if self.dsa_shrink_latent != 2:
            raise RuntimeError(
                "[SFA cross-layer graph] staged path requires SHRINK_LATENT=2"
            )
        if self.enable_sparse_sfa_c8 or self.enable_sparse_li_c8:
            raise RuntimeError(
                "[SFA cross-layer graph] sparse C8 layouts are not "
                "supported by the staged path"
            )
        if self.enable_mlapo:
            raise RuntimeError(
                "[SFA cross-layer graph] MLAPO is not supported by the "
                "staged path"
            )
        if self.enable_dsa_cp:
            raise RuntimeError(
                "[SFA cross-layer graph] DSA-CP is not supported by the "
                "staged path"
            )
        if not staged_sfa_connector_supports_sparse_load():
            raise RuntimeError(
                "[SFA cross-layer graph] the active connector does not "
                "support staged sparse selective loads"
            )
        # Two-group indexer tables: the staged pre dereferences the indexer
        # group's own table/slots below; an unbundle-only configuration
        # leaves them None and must fail before graph execution.
        # Provenance: fork sfa_v1.py:2554-2605 (eligibility subset).
        if (
            getattr(attn_metadata, "indexer_block_table", None) is None
            or getattr(attn_metadata, "indexer_slot_mapping", None) is None
        ):
            raise RuntimeError(
                "[SFA cross-layer graph] requires two-group indexer tables "
                "(VLLM_ASCEND_DSA_TWO_GROUPS=1)"
            )
        # Scratch reservation must fit inside one request's logical table:
        # the fused remap kernel gathers through logical positions across
        # the whole scratch capacity.
        table_width = int(attn_metadata.block_table.shape[1])
        block_size = int(self.vllm_config.cache_config.block_size)
        if self.dsa_scratch_capacity > table_width * block_size:
            raise RuntimeError(
                "[SFA cross-layer graph] scratch capacity exceeds the "
                f"block-table logical capacity: scratch="
                f"{self.dsa_scratch_capacity}, capacity="
                f"{table_width * block_size}"
            )
        if len(kv_cache) != 3:
            raise RuntimeError(
                "[SFA cross-layer graph] requires exactly three KV tensors"
            )
        if any(cache.ndim != 4 for cache in kv_cache):
            raise RuntimeError(
                "[SFA cross-layer graph] requires rank-4 PA_BSND KV tensors"
            )
        expected_hidden_dims = (
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.head_dim,
        )
        if tuple(int(cache.shape[-1]) for cache in kv_cache) != tuple(
            int(dim) for dim in expected_hidden_dims
        ):
            raise RuntimeError(
                "[SFA cross-layer graph] KV cache hidden dimensions do "
                "not match SFA"
            )

        is_dummy = bool(
            getattr(context, "staged_sfa_graph_dummy_run", False)
        )
        state = self._staged_sfa_capture_state
        initialized_capacity = state.initialized_cache_capacity
        if (
            is_dummy
            and int(graph_key.request_capacity) > initialized_capacity
        ):
            for cache in kv_cache:
                cache[initialized_capacity : int(graph_key.request_capacity)].zero_()
            state.initialized_cache_capacity = int(graph_key.request_capacity)

        if (
            is_dummy
            and getattr(context, "cudagraph_runtime_mode", CUDAGraphMode.NONE)
            == CUDAGraphMode.PIECEWISE
        ):
            # ACL capture cannot include the host copy in boundary
            # preparation. The immediately preceding eager warmup filled
            # this stable buffer (03-4 §7 constraint ②).
            capture_boundary = state.remap_boundary
            boundary = attn_metadata.decode_remap_boundary
            if (
                capture_boundary is None
                or boundary is None
                or capture_boundary.data_ptr() != boundary.data_ptr()
                or capture_boundary.shape != boundary.shape
            ):
                raise RuntimeError(
                    "[SFA cross-layer graph] remap boundary was not "
                    "prepared in stable storage by eager warmup"
                )
            remap_boundary = capture_boundary
        else:
            remap_boundary = _prepare_sfa_remap_boundary(
                attn_metadata,
                attn_metadata.req_ids,
                is_dummy_run=is_dummy,
                index_topk=self.dsa_index_topk,
                cached_tokens=getattr(
                    getattr(context, "staged_sfa_route", None),
                    "frontiers",
                    None,
                ),
            )
            if is_dummy:
                state.remap_boundary = remap_boundary

        row_req_indices = attn_metadata.decode_req_indices
        selected_packed = attn_metadata.decode_selected_tokens
        selected_counts = attn_metadata.decode_selected_counts
        target_slots = attn_metadata.decode_target_slot_mapping
        local_to_union_workspace = (
            attn_metadata.decode_union_mapping_workspace
        )
        if any(
            value is None
            for value in (
                row_req_indices,
                selected_packed,
                selected_counts,
                target_slots,
                local_to_union_workspace,
            )
        ):
            raise RuntimeError(
                "staged SFA request-union buffers are unavailable"
            )

        outputs = self._cross_layer_pre_compute(
            hidden_states,
            kv_cache[0],
            kv_cache[1],
            kv_cache[2],
            attn_metadata.cos,
            attn_metadata.sin,
            attn_metadata.slot_mapping,
            attn_metadata.indexer_slot_mapping,
            attn_metadata.cum_query_lens,
            attn_metadata.seq_lens,
            attn_metadata.indexer_block_table,
            remap_boundary,
            row_req_indices,
            attn_metadata.block_table,
            selected_packed,
            selected_counts,
            target_slots,
            local_to_union_workspace,
            attn_metadata,
        )
        outputs = self._copy_to_staged_sfa_bridge(
            hidden_states,
            outputs,
        )
        producer_event = state.producer_event
        if producer_event is None:
            if (
                getattr(
                    context, "cudagraph_runtime_mode", CUDAGraphMode.NONE
                )
                != CUDAGraphMode.NONE
            ):
                raise RuntimeError(
                    "staged SFA producer event was not created by "
                    "eager warmup"
                )
            producer_event = torch.npu.Event()
            state.producer_event = producer_event
        attn_metadata.reshape_cache_event = producer_event
        producer_event.record()
        state.runtime = (
            layer_name,
            kv_cache,
            index_layer_name,
            index_enabled,
        )
        if (
            is_dummy
            and getattr(context, "cudagraph_runtime_mode", CUDAGraphMode.NONE)
            == CUDAGraphMode.PIECEWISE
        ):
            state.register(graph_key, outputs, kv_cache)
        return outputs

    def cross_layer_graph_post(
        self,
        layer_name: str,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        topk_indices: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: M | None,
        output: torch.Tensor,
    ) -> None:
        """Run graph B: sparse FA + projections into the fixed output.

        Provenance: fork sfa_v1.py:3152-3175.
        """
        graph_key = getattr(
            get_forward_context(), "staged_sfa_graph_key", None
        )
        if attn_metadata is None or graph_key is None:
            return
        rows = int(graph_key.token_capacity)
        kv_cache, _, _ = self._cross_layer_kv_cache(layer_name, kv_cache)
        self._cross_layer_post_compute(
            ql_nope[:rows],
            q_pe[:rows],
            topk_indices[:rows],
            kv_cache[0],
            kv_cache[1],
            attn_metadata.cum_query_lens,
            attn_metadata.seq_lens,
            attn_metadata.block_table,
            output,
            attn_metadata,
        )

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config=None,
        speculative_config=None,
        num_dcp_pcp_tokens=None,
        draft_attn_metadatas=None,
    ):
        # sfa does not need to update graph params
        pass

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        # NOTE: We currently do not support quant kv_b_proj.
        assert isinstance(self.kv_b_proj.quant_method, UnquantizedLinearMethod)
        # NOTE: Weight will be reshaped next, we need to revert and transpose it.
        kv_b_proj_weight = torch_npu.npu_format_cast(self.kv_b_proj.weight.data, ACL_FORMAT_FRACTAL_ND).T
        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.local_num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        ), (
            f"{kv_b_proj_weight.shape=}, "
            f"{self.kv_lora_rank=}, "
            f"{self.local_num_heads=}, "
            f"{self.qk_nope_head_dim=}, "
            f"{self.v_head_dim=}"
        )
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.local_num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # NOTE: When we make a incontiguous weight contiguous, a new address will be allocated for the weight,
        # in graph + RL scenario, we only capture the graph once, and the weight address is expected to be the same
        # across iterations, so we need to copy the weight to the original address after making it contiguous.
        if not hasattr(self, "W_UV"):
            # Convert from (L, N, V) to (N, L, V)
            self.W_UV = W_UV.transpose(0, 1).contiguous()
            # Convert from (L, N, P) to (N, P, L)
            self.W_UK_T = W_UK.permute(1, 2, 0).contiguous()
        else:
            self.W_UV.copy_(W_UV.transpose(0, 1).contiguous())
            self.W_UK_T.copy_(W_UK.permute(1, 2, 0).contiguous())

        # TODO(zzzzwwjj): Currently, torch.ops._C_ascend.batch_matmul_transpose cannot support weight nz
        # self.W_UV = maybe_trans_nz(self.W_UV)

        # Dispose kv_b_proj since it is replaced by W_UV and W_UK_T to save memory
        dispose_layer(self.kv_b_proj)
        if self.enable_dsa_cp:
            if self.enable_dsa_cp_with_layer_shard:
                for layer in self.layer_sharding_kwargs or []:
                    if is_hidden_layer(layer):
                        post_process_after_loading_for_shard_weight_series(layer)
            elif self.enable_dsa_cp_with_o_proj_tp:
                self._init_o_proj_tp_full_params()

        if self.enable_sfa_prolog_v3:
            reasons = self._get_sfa_prolog_v3_unsupported_reasons()
            if reasons:
                self.enable_sfa_prolog_v3 = False
                self.enable_mlapo = False
                for msg in reasons:
                    logger.warning_once(
                        f"{msg} Disable SFA mla_prolog_v3 for layer {self.layer_name}; "
                        "fallback to native preprocessing."
                    )
            else:
                self._process_weights_for_fused_prolog_v3()

        if not self.enable_sfa_prolog_v3 and self.enable_mlapo:
            quant_method = getattr(
                getattr(self.fused_qkv_a_proj, "quant_method", None),
                "quant_method",
                None,
            )
            reasons = []
            is_quantized = isinstance(quant_method, (AscendW8A8LinearMethod, AscendW8A8MXFP8DynamicLinearMethod))
            if self.fused_qkv_a_proj is None:
                reasons.append("fused_qkv_a_proj is None, mlapo is disabled.")
            if not is_quantized and get_ascend_device_type() != AscendDeviceType.A5:
                reasons.append(
                    "Currently mlapo only supports W8A8 quantization in SFA scenario on non-A5 devices."
                    "Some layers in your model are not quantized with W8A8,"
                    "thus mlapo is disabled for these layers."
                )
            if self.enable_dsa_cp:
                reasons.append("Currently mlapo does not support SFA with CP,thus mlapo is disabled for these layers.")
            if reasons:
                self.enable_mlapo = False
                for msg in reasons:
                    logger.warning_once(msg)
            else:
                self.mlapo_is_quantized = is_quantized
                if get_ascend_device_type() == AscendDeviceType.A5:
                    if is_quantized:
                        self._process_weights_for_fused_mlapo_a5(act_dtype)
                    else:
                        self._process_weights_for_fused_mlapo_a5_float(act_dtype)
                else:
                    self._process_weights_for_fused_mlapo(act_dtype)

        if self.enable_sparse_li_c8 and get_ascend_device_type() == AscendDeviceType.A5:
            if hasattr(self, "mlapo_is_quantized") and not self.mlapo_is_quantized:
                self.c8_k_cache_dtype = act_dtype
                self.c8_k_scale_cache_dtype = act_dtype

        if not self.enable_mlapo and not self.enable_sfa_prolog_v3:
            # if mlapo, W_UK_T can't trans nz
            self.W_UK_T = maybe_trans_nz(self.W_UK_T)

        if self.has_indexer and self.enable_sparse_li_c8 and AscendSFAImpl.q_hadamard is None:
            AscendSFAImpl.q_hadamard = torch.tensor(scipy.linalg.hadamard(128), dtype=torch.bfloat16, device="npu") / (
                128**0.5
            )
        if self.has_indexer and self.enable_sparse_li_c8 and AscendSFAImpl.k_hadamard is None:
            AscendSFAImpl.k_hadamard = torch.tensor(scipy.linalg.hadamard(128), dtype=torch.bfloat16, device="npu") / (
                128**0.5
            )

    def _dsa_reassemble_kv_cache(
        self, layer_name: str, kv_cache: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        """Re-assemble the unbundle MLA 2-tuple into a 3-tuple for indexer.

        Un-bundled (Step 2 / A2): the MLA layer owns only the latent 2-tuple
        while the sibling ``...self_attn.indexer.k_cache`` layer owns the
        indexer key as its own 1-tuple. Both groups share the same block id
        space in the unbundle-only slice, so ``attn_metadata.block_table`` /
        ``slot_mapping`` address both caches; reassembling
        ``(k_nope, k_pe, indexer_k)`` keeps the indexer read/write
        (``kv_cache[2]``) unchanged in both ``sfa_v1`` and ``device_op``.

        The indexer KV tensor is allocated once at startup, so the sibling
        reference is resolved lazily and cached to avoid a per-step
        ``no_compile_layers`` dict lookup + tuple rebuild on the decode path.
        """
        if not self.dsa_unbundle or len(kv_cache) >= 3:
            return kv_cache
        _idx_t = getattr(self, "_dsa_idx_cache_t", None)
        if _idx_t is None:
            _idx_cache = get_forward_context().no_compile_layers[
                _dsa_indexer_layer_name(layer_name)
            ].kv_cache
            _idx_t = _idx_cache[0] if isinstance(_idx_cache, (tuple, list)) else _idx_cache
            self._dsa_idx_cache_t = _idx_t
        return (kv_cache[0], kv_cache[1], _idx_t)

    def _dsa_maybe_wait_indexer_rows(
        self,
        layer_name: str,
        kv_cache: tuple[torch.Tensor, ...] | None,
    ) -> None:
        """Wait for remote indexer rows before top-k selection (PD decode).

        A cold shared-cache decode needs prompt index rows before top-k
        selection. The group-1 wait is a no-op when resident and does not
        advance the group-0 latent-layer cursor. Without it, the native
        path can run top-k while the prompt indexer rows are still in
        flight from the peer (PD cold start), reading uninitialized rows.
        Provenance: fork sfa_v1.py:3416-3421.
        """
        if (
            kv_cache is not None
            and self.dsa_unbundle
            and self.has_indexer
            and _dsa_index_lmcache_enabled()
        ):
            wait_for_kv_layer_from_connector(
                _dsa_indexer_layer_name(layer_name)
            )

    def _maybe_save_unbundled_kv_cache(
        self,
        layer_name: str,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: M | None = None,
    ) -> None:
        """Save the unbundled latent and indexer groups separately (A2d).

        Fork semantics (vllm-ascend-sparse sfa_v1.py:3187-3189): the latent
        group is dispatched under the MLA layer name and the indexer group
        under its sibling ``...self_attn.indexer.k_cache`` name so the
        connector routes kv_group=0 / kv_group=1 independently. Bundled
        (2-tuple) layers keep the single latent-only save.

        DSA shrink (B2d, fork :4220-4231 simplified): on a pure decode step
        (DecodeOnly / SpecDecoding — MTP verify steps included) the latent
        tail stays NPU-resident and a regular per-token store buys nothing
        (~88KB/token of PCIe for no consumer), so the regular save is
        skipped only while decode-window saving is disabled. When a window
        is configured, the same layer callback must remain active so the
        synthetic store-only metadata can persist the window and advance the
        committed frontier (fork :4220-4231 exact gate).
        """
        if (
            self.dsa_shrink_latent
            and attn_metadata is not None
            and attn_metadata.attn_state in (
                AscendAttentionState.DecodeOnly,
                AscendAttentionState.SpecDecoding,
            )
            and _decode_window_save_window_size() <= 0
        ):
            return
        maybe_save_kv_layer_to_connector(layer_name, [kv_cache[0], kv_cache[1]])
        if self.dsa_unbundle and len(kv_cache) >= 3:
            maybe_save_kv_layer_to_connector(
                _dsa_indexer_layer_name(layer_name),
                [kv_cache[2]],
            )

    @staticmethod
    def _is_w8a8_dynamic_linear(layer: torch.nn.Module | None) -> bool:
        quant_method = getattr(getattr(layer, "quant_method", None), "quant_method", None)
        return isinstance(quant_method, AscendW8A8DynamicLinearMethod)

    def _get_sfa_prolog_v3_unsupported_reasons(self) -> list[str]:
        reasons = []
        for name, layer in (
            ("fused_qkv_a_proj", self.fused_qkv_a_proj),
            ("q_proj", self.q_proj),
        ):
            if not self._is_w8a8_dynamic_linear(layer):
                reasons.append(f"Currently SFA mla_prolog_v3 only supports W8A8 dynamic quantization for {name}.")
        if self.kv_a_layernorm is None or self.q_a_layernorm is None:
            reasons.append("SFA mla_prolog_v3 requires q_a_layernorm and kv_a_layernorm.")
        if getattr(self.q_proj, "_chunk_size", 0):
            reasons.append("SFA mla_prolog_v3 does not support chunked q_proj weights yet.")
        if self.enable_dsa_cp:
            reasons.append("SFA mla_prolog_v3 does not support DSA-CP; DSA-CP takes precedence.")
        if self.is_kv_producer:
            reasons.append("SFA mla_prolog_v3 is disabled on KV producer workers.")
        return reasons

    def _process_weights_for_fused_prolog_v3(self) -> None:
        assert self.fused_qkv_a_proj is not None
        assert self.q_proj is not None

        fused_weight = self.fused_qkv_a_proj.weight.data
        weight_dq = fused_weight[..., : self.q_lora_rank].contiguous()
        weight_dkv_kr = fused_weight[..., self.q_lora_rank :].contiguous()
        weight_uq_qr = self.q_proj.weight.data.contiguous()
        self.weight_dq = torch_npu.npu_format_cast(weight_dq, ACL_FORMAT_FRACTAL_NZ)
        self.weight_dkv_kr = torch_npu.npu_format_cast(weight_dkv_kr, ACL_FORMAT_FRACTAL_NZ)
        self.weight_uq_qr = torch_npu.npu_format_cast(weight_uq_qr, ACL_FORMAT_FRACTAL_NZ)

        q_a_proj_deq_scl = self.fused_qkv_a_proj.weight_scale[: self.q_lora_rank].contiguous()
        kv_a_proj_deq_scl = self.fused_qkv_a_proj.weight_scale[self.q_lora_rank :].contiguous()
        self.dequant_scale_w_dq = q_a_proj_deq_scl.view(1, -1).to(torch.float)
        self.dequant_scale_w_dkv_kr = kv_a_proj_deq_scl.view(1, -1).to(torch.float)
        self.dequant_scale_w_uq_qr = self.q_proj.weight_scale.data.view(1, -1).to(torch.float)
        if self.enable_sparse_sfa_c8:
            self.sfa_qsfa_k_nope_clip_alpha = torch.ones(
                1,
                dtype=torch.float32,
                device=self.weight_dq.device,
            )
            if self.sfa_qsfa_kr_cache_dummy is None:
                # ckvkr_repo_mode=1 stores rope in the packed KV cache, but the
                # operator still requires kr_cache. Keep a stable, non-aliased
                # dummy so first-run tiling/graph capture cannot alias kv_cache.
                self.sfa_qsfa_kr_cache_dummy = torch.empty(
                    0,
                    dtype=torch.bfloat16,
                    device=self.weight_dq.device,
                )
        if self.is_kv_consumer:
            # Decode-only workers only execute Prolog. Drop the native Linear
            # weights after their Prolog layouts and scales have been copied.
            dispose_layer(self.fused_qkv_a_proj)
            dispose_layer(self.q_proj)
            torch.npu.empty_cache()

    # Processing the input parameters for MLAPO by reordering and transposing
    # QKV(and part of Q) weight, applying RoPE-related dimension transformations,
    # and handling quantization parameters.
    def _process_weights_for_fused_mlapo(self, act_dtype: torch.dtype):
        assert self.kv_a_proj_with_mqa is None
        assert self.fused_qkv_a_proj is not None

        kv_a_proj_wt = self.fused_qkv_a_proj.weight.data[..., self.q_lora_rank :].contiguous()
        q_a_proj_wt = self.fused_qkv_a_proj.weight.data[..., : self.q_lora_rank].contiguous()

        kv_a_proj_wt = kv_a_proj_wt.t().contiguous()
        kv_a_proj_wt = trans_rope_weight(kv_a_proj_wt, self.qk_rope_head_dim)
        kv_a_proj_wt = kv_a_proj_wt.t().contiguous()
        wd_qkv = torch.cat((kv_a_proj_wt, q_a_proj_wt), dim=-1)
        wd_qkv = wd_qkv.t().contiguous()
        wd_qkv = transdata(wd_qkv, block_size=(16, 32)).unsqueeze(0).contiguous()
        self.wd_qkv = torch_npu.npu_format_cast(wd_qkv, 29)

        kv_a_proj_deq_scl = self.fused_qkv_a_proj.deq_scale[self.q_lora_rank :].contiguous()
        q_a_proj_deq_scl = self.fused_qkv_a_proj.deq_scale[: self.q_lora_rank].contiguous()
        kv_a_proj_deq_scl = kv_a_proj_deq_scl.reshape(self.kv_lora_rank + self.qk_rope_head_dim, -1).contiguous()
        kv_a_proj_deq_scl = trans_rope_weight(kv_a_proj_deq_scl, self.qk_rope_head_dim)
        kv_a_proj_deq_scl = kv_a_proj_deq_scl.view(self.kv_lora_rank + self.qk_rope_head_dim).contiguous()
        self.deq_scale_qkv = torch.cat((kv_a_proj_deq_scl, q_a_proj_deq_scl), dim=-1).contiguous()

        kv_a_proj_qt_bias = self.fused_qkv_a_proj.quant_bias[self.q_lora_rank :].contiguous()
        q_a_proj_qt_bias = self.fused_qkv_a_proj.quant_bias[: self.q_lora_rank].contiguous()

        kv_a_proj_qt_bias = kv_a_proj_qt_bias.reshape(self.kv_lora_rank + self.qk_rope_head_dim, -1).contiguous()
        kv_a_proj_qt_bias = trans_rope_weight(kv_a_proj_qt_bias, self.qk_rope_head_dim)
        kv_a_proj_qt_bias = kv_a_proj_qt_bias.view(self.kv_lora_rank + self.qk_rope_head_dim).contiguous()
        self.quant_bias_qkv = torch.cat((kv_a_proj_qt_bias, q_a_proj_qt_bias), dim=-1).contiguous()

        wu_q = self.q_proj.weight.data
        wu_q = wu_q.t().reshape(self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim, -1)
        wu_q = trans_rope_weight(wu_q, self.qk_rope_head_dim)
        wu_q = wu_q.reshape(self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim), -1)
        wu_q = transdata(wu_q, block_size=(16, 32)).unsqueeze(0).contiguous()
        self.wu_q = torch_npu.npu_format_cast(wu_q, 29)

        qb_deq_scl = self.q_proj.deq_scale.data
        qb_deq_scl = qb_deq_scl.reshape(self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim, -1)
        qb_deq_scl = trans_rope_weight(qb_deq_scl, self.qk_rope_head_dim)
        self.qb_deq_scl = qb_deq_scl.reshape(self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim))

        qb_qt_bias = self.q_proj.quant_bias.data
        qb_qt_bias = qb_qt_bias.reshape(self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim, -1)
        qb_qt_bias = trans_rope_weight(qb_qt_bias, self.qk_rope_head_dim)
        self.qb_qt_bias = qb_qt_bias.reshape(self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim))

        device = self.q_proj.weight.device
        self.gamma1 = self.q_a_layernorm.weight.data  # type: ignore[union-attr]
        self.beta1 = self.q_a_layernorm.bias.data  # type: ignore[union-attr]
        self.gamma2 = self.kv_a_layernorm.weight.data  # type: ignore[union-attr]
        self.quant_scale0 = self.fused_qkv_a_proj.input_scale.data
        self.quant_offset0 = self.fused_qkv_a_proj.input_offset.data
        self.quant_scale1 = self.q_proj.input_scale.data
        self.quant_offset1 = self.q_proj.input_offset.data
        self.ctkv_scale = torch.tensor([1], dtype=act_dtype, device=device)
        self.q_nope_scale = torch.tensor([1], dtype=act_dtype, device=device)

        # On KV consumers (decode-only) MLAPO uses the transformed weights built above;
        # the original fused_qkv_a_proj/q_proj weights and quant params are no longer
        # referenced, so drop them to save memory.
        if (
            self.vllm_config.kv_transfer_config is not None
            and self.vllm_config.kv_transfer_config.is_kv_consumer
            and self.vllm_config.scheduler_config.max_num_batched_tokens <= MLAPO_MAX_SUPPORTED_TOKENS
        ):
            self.fused_qkv_a_proj.weight = None
            self.fused_qkv_a_proj.deq_scale = None
            self.fused_qkv_a_proj.quant_bias = None
            self.q_proj.weight = None
            self.q_proj.deq_scale = None
            self.q_proj.quant_bias = None
            torch.npu.empty_cache()

    def _process_weights_for_fused_mlapo_a5(self, act_dtype: torch.dtype):
        assert self.fused_qkv_a_proj is not None
        assert self.q_proj is not None
        weight_dq = self.fused_qkv_a_proj.weight.data[..., : self.q_lora_rank].contiguous()
        self.weight_dq = torch_npu.npu_format_cast(weight_dq, 29)

        weight_uq_qr = self.q_proj.weight.data.contiguous()
        self.weight_uq_qr_scale = self.q_proj.weight_scale.data.transpose(0, 1)
        self.weight_uq_qr_scale = self.weight_uq_qr_scale.reshape(
            -1, self.weight_uq_qr_scale.shape[1] * self.weight_uq_qr_scale.shape[2]
        )
        self.weight_uq_qr = torch_npu.npu_format_cast(weight_uq_qr, 29)

        weight_dkv_kr = self.fused_qkv_a_proj.weight.data[..., self.q_lora_rank :].contiguous()
        self.weight_dkv_kr = torch_npu.npu_format_cast(weight_dkv_kr, 29)

        weight_scale = self.fused_qkv_a_proj.weight_scale
        weight_scale = weight_scale.transpose(0, 1)
        weight_scale = weight_scale.reshape(-1, weight_scale.shape[1] * weight_scale.shape[2])
        self.weight_dq_scale = weight_scale[: self.q_lora_rank, ...]
        self.weight_dkv_kr_scale = weight_scale[self.q_lora_rank :, ...]

    def _process_weights_for_fused_mlapo_a5_float(self, act_dtype: torch.dtype):
        assert self.fused_qkv_a_proj is not None
        assert self.q_proj is not None
        self.fused_qkv_a_proj.weight.data = self.fused_qkv_a_proj.weight.data.T
        weight_dq = self.fused_qkv_a_proj.weight.data[..., : self.q_lora_rank].contiguous()
        self.weight_dq_cpu = weight_dq.cpu()
        self.weight_dq = torch_npu.npu_format_cast(weight_dq, 29)

        weight_uq_qr = self.q_proj.weight.data.T
        weight_uq_qr = weight_uq_qr.contiguous()
        self.weight_uq_qr_cpu = weight_uq_qr.cpu()
        self.weight_uq_qr = torch_npu.npu_format_cast(weight_uq_qr, 29)

        weight_dkv_kr = self.fused_qkv_a_proj.weight.data[..., self.q_lora_rank :].contiguous()
        self.weight_dkv_kr_cpu = weight_dkv_kr.cpu()
        self.weight_dkv_kr = torch_npu.npu_format_cast(weight_dkv_kr, 29)

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: M,
        k_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        raise NotImplementedError("forward_mha is not supported for SFA attention. Use forward() instead.")

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: M,
        layer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError("forward_mqa is not supported for SFA attention. Use forward() instead.")

    def rope_single(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = x.shape
        S = 1
        x = x.view(B, N, S, D)
        x = torch_npu.npu_interleave_rope(x, cos, sin)
        return x.view(B, N, D)

    def _init_o_proj_tp_full_params(self):
        """
        Initialize TP-mode aliases and Full-mode buffers for DSA-CP o_proj.

        In SFA DSA-CP mixed execution, the same model instance can run both
        decode-only and prefill/mixed batches:
        - Decode-only batches all-to-all the SFA output in the TP group, then
          run the original TP-sharded o_proj.
        - Prefill/mixed batches produce SFA output that is not directly
          compatible with TP-sharded o_proj, so each rank all-gathers the TP
          o_proj shards and input-sharded quant params before running o_proj.

        The original TP parameter storage remains the persistent source of
        truth. The o_proj_tp_* tensors below alias that storage, while the
        o_proj_full_* tensors are temporary gather destinations reused across
        forwards. They are not a second persistent copy of the TP weight.
        """
        sample = self.o_proj.weight
        self.o_proj_full_weight_gather_dim = 1 if self._is_o_proj_unquantized() else 0
        if self.o_proj_full_weight_gather_dim == 0:
            full_shape = (sample.shape[0] * self.tp_size, sample.shape[1])
            gather_shape = full_shape
        else:
            full_shape = (sample.shape[0], sample.shape[1] * self.tp_size)
            gather_shape = (sample.shape[1] * self.tp_size, sample.shape[0])
        # Main and MTP layers can use different quantized o_proj weight layouts,
        # so key the shared full-gather pool by gather dimension, dtype, and shape.
        pool_key = (
            sample.device.type,
            sample.device.index,
            sample.dtype,
            self.o_proj_full_weight_gather_dim,
            full_shape,
        )
        if pool_key not in AscendSFAImpl.o_proj_full_pools:
            AscendSFAImpl.o_proj_full_pools[pool_key] = torch.empty(
                gather_shape, dtype=sample.dtype, device=sample.device
            )
        self.o_proj_full_gather_pool = AscendSFAImpl.o_proj_full_pools[pool_key]
        if self.o_proj_full_weight_gather_dim == 0:
            self.o_proj_full_pool = self.o_proj_full_gather_pool
        else:
            self.o_proj_full_pool = self.o_proj_full_gather_pool.transpose(0, 1)

        # TP tensors alias the original parameter storage. The TP shard remains
        # the single source of truth; full-weight tensors below are temporary
        # gather destinations only.
        self.o_proj_tp_weight = self.o_proj.weight.detach()
        if self.o_proj_full_weight_gather_dim == 0:
            self.o_proj_tp_weight_gather_input = self.o_proj_tp_weight
        else:
            # Communication scratch only: all_gather_into_tensor concatenates on
            # dim0, while unquantized row-parallel o_proj is sharded on dim1.
            self.o_proj_tp_weight_gather_input = self.o_proj_tp_weight.transpose(0, 1).contiguous()
        self.o_proj_tp_aclnn_input_params = {}
        self.o_proj_full_aclnn_input_params = {}
        for param_name in O_PROJ_ACLNN_INPUT_PARAMS:
            param = getattr(self.o_proj, param_name, None)
            if param is None:
                continue
            self.o_proj_tp_aclnn_input_params[param_name] = param.detach()
            self.o_proj_full_aclnn_input_params[param_name] = param.repeat(self.tp_size)

        self.o_proj_tp_input_sharded_quant_params = {}
        self.o_proj_full_input_sharded_quant_params = {}
        for param_name, param in self._iter_o_proj_input_sharded_quant_params():
            self.o_proj_tp_input_sharded_quant_params[param_name] = param.detach()
            self.o_proj_full_input_sharded_quant_params[param_name] = torch.empty(
                (param.shape[0] * self.tp_size, *param.shape[1:]), dtype=param.dtype, device=param.device
            )

    def _iter_o_proj_input_sharded_quant_params(self):
        if not isinstance(self.o_proj, nn.Module):
            return
        for param_name, param in self.o_proj.named_parameters(recurse=False):
            if param_name == "weight" or param_name in O_PROJ_ACLNN_INPUT_PARAMS:
                continue
            if getattr(param, "input_dim", None) == 1:
                yield param_name, param

    def _switch_o_proj_params(self, params: dict[str, torch.Tensor]):
        for param_name, param in params.items():
            getattr(self.o_proj, param_name).set_(param)

    def _get_o_proj_linear_method(self):
        quant_method = self.o_proj.quant_method
        return getattr(quant_method, "quant_method", quant_method)

    def _is_o_proj_unquantized(self) -> bool:
        return isinstance(self._get_o_proj_linear_method(), UnquantizedLinearMethod)

    def _apply_o_proj_full_weight(self, attn_output: torch.Tensor) -> torch.Tensor:
        return self._get_o_proj_linear_method().apply(self.o_proj, attn_output)

    def _handle_o_proj_weight_switch_and_forward(
        self,
        attn_output: torch.Tensor,
        output: torch.Tensor,
        o_proj_full_handle: torch.distributed.Work | None,
        o_proj_full_param_handles: list[torch.distributed.Work | None] | None,
        should_shard_weight: bool,
    ) -> tuple[torch.Tensor, bool]:
        """
        Handle o_proj weight switching between TP-mode and Full-mode, and execute forward computation.
        """
        # Gather o_proj weight from all TP ranks for Full-mode computation
        if should_shard_weight:
            # Wait for the completion of o_proj weight all-gather operation
            if o_proj_full_handle is not None:
                o_proj_full_handle.wait()
            for handle in o_proj_full_param_handles or []:
                if handle is not None:
                    handle.wait()

            # Temporarily switch o_proj to the gathered full-weight view for
            # prefill/mixed DSA-CP, whose attention output is not TP-sharded.
            self.o_proj.weight.set_(self.o_proj_full_pool)
            self._switch_o_proj_params(self.o_proj_full_aclnn_input_params)
            self._switch_o_proj_params(self.o_proj_full_input_sharded_quant_params)
            output[...] = self._apply_o_proj_full_weight(attn_output)
            # Restore TP aliases so later decode batches keep using TP storage.
            self.o_proj.weight.set_(self.o_proj_tp_weight)
            self._switch_o_proj_params(self.o_proj_tp_aclnn_input_params)
            self._switch_o_proj_params(self.o_proj_tp_input_sharded_quant_params)

            return output, False
        else:
            # For decode scenario: perform all-to-all communication on o_proj input activations
            # Reshape for all-to-all: [batch * seq, tp_size, head_dim] -> [tp_size, batch * seq, head_dim]
            send = (
                attn_output.view(-1, self.tp_size, self.num_heads * self.v_head_dim)
                .permute(1, 0, 2)
                .reshape(-1, self.num_heads * self.v_head_dim)
            )

            attn_output = torch.empty_like(send)
            torch.distributed.all_to_all_single(attn_output, send, group=get_tp_group().device_group)

            return attn_output, True

    @staticmethod
    def _flatten_for_byte_gather(tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
        if tensor.dim() == 0:
            raise RuntimeError("Byte-packed all-gather requires tensors with a token dimension.")
        tensor = tensor.contiguous()
        num_rows = tensor.shape[0]
        num_bytes_per_row = tensor.element_size()
        for dim in tensor.shape[1:]:
            num_bytes_per_row *= dim
        return tensor.view(torch.int8).view(num_rows, num_bytes_per_row), num_bytes_per_row

    @classmethod
    def _all_gather_byte_packed_async(
        cls,
        parts: list[tuple[str, torch.Tensor]],
        async_op: bool,
    ) -> tuple[torch.Tensor, torch.distributed.Work | None, tuple[_ByteGatherPart, ...]]:
        if not parts:
            raise RuntimeError("Byte-packed all-gather requires at least one tensor.")

        packed_parts = []
        metadata = []
        expected_num_rows: int | None = None
        for name, tensor in parts:
            num_rows = tensor.shape[0]
            if expected_num_rows is None:
                expected_num_rows = num_rows
            elif num_rows != expected_num_rows:
                raise RuntimeError(
                    "Cannot byte-pack KV tensors with different token counts: "
                    f"expected {expected_num_rows}, got {num_rows} for {name}."
                )

            packed_tensor, num_bytes_per_row = cls._flatten_for_byte_gather(tensor)
            packed_parts.append(packed_tensor)
            metadata.append(
                _ByteGatherPart(
                    name=name,
                    shape=tuple(tensor.shape),
                    dtype=tensor.dtype,
                    num_bytes_per_row=num_bytes_per_row,
                )
            )

        packed_input = torch.cat(packed_parts, dim=1) if len(packed_parts) > 1 else packed_parts[0]
        gathered, handle = all_gather_async(
            packed_input,
            get_tp_group(),
            async_op=async_op,
        )
        return gathered, handle, tuple(metadata)

    @staticmethod
    def _restore_byte_gathered_tensors(
        gathered: torch.Tensor,
        metadata: tuple[_ByteGatherPart, ...],
    ) -> dict[str, torch.Tensor]:
        chunks = torch.split(gathered, [part.num_bytes_per_row for part in metadata], dim=1)
        num_rows = gathered.shape[0]
        restored = {}
        for part, chunk in zip(metadata, chunks):
            restored[part.name] = chunk.contiguous().view(part.dtype).view(num_rows, *part.shape[1:])
        return restored

    def _get_full_kv(self, k, attn_metadata):
        return k

    def exec_kv(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple,
        slots: torch.Tensor,
        attn_metadata: M,
    ):
        B = kv_no_split.shape[0]
        N = self.num_kv_heads
        S = 1
        # npu_kv_rmsnorm_rope_cache needs [B, N, S, D]
        kv_no_split = kv_no_split.view(B, N, S, self.kv_lora_rank + self.qk_rope_head_dim)
        cache_mode = "PA"

        use_custom_kv = self.enable_sparse_sfa_c8 and (
            get_ascend_device_type() != AscendDeviceType.A5 or self.enable_dsa_cp or not self.has_indexer
        )
        if use_custom_kv:
            assert self.kv_a_layernorm is not None
            return custom_kv_rmsnorm_rope(
                kv_no_split,
                self.kv_a_layernorm.weight,
                cos,
                sin,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                epsilon=self.kv_a_layernorm.variance_epsilon,
                dst_type=(torch.float8_e4m3fn if get_ascend_device_type() == AscendDeviceType.A5 else 1),
                tile_size=self.sfa_qsfa_tile_size,
            )

        if self.enable_dsa_cp:
            _, _, k_pe, k_nope = torch_npu.npu_kv_rmsnorm_rope_cache(
                kv_no_split,
                self.kv_a_layernorm.weight,  # type: ignore[union-attr]
                cos,
                sin,
                slots.to(torch.int64),
                kv_cache[1],
                kv_cache[0],
                epsilon=self.kv_a_layernorm.variance_epsilon,  # type: ignore[union-attr]
                cache_mode=cache_mode,
                is_output_kv=True,
            )
            return k_pe, k_nope, None
        else:
            torch_npu.npu_kv_rmsnorm_rope_cache(
                kv_no_split,
                self.kv_a_layernorm.weight,  # type: ignore[union-attr]
                cos,
                sin,
                slots.to(torch.int64),
                kv_cache[1],
                kv_cache[0],
                epsilon=self.kv_a_layernorm.variance_epsilon,  # type: ignore[union-attr]
                cache_mode=cache_mode,
            )
            return None, None

    # Return `ql_nope`, `q_pe`
    def _q_proj_and_k_up_proj(self, x):
        q_nope, q_pe = (
            self.q_proj(x)[0]
            .view(-1, self.local_num_heads, self.qk_head_dim)
            .split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        )

        # Convert from (B, N, P) to (N, B, P)
        q_nope = q_nope.transpose(0, 1)
        # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
        ql_nope = torch.bmm(q_nope, self.W_UK_T)
        # Convert from (N, B, L) to (B, N, L)
        return ql_nope.transpose(0, 1), q_pe

    def _v_up_proj(self, x):
        num_input_tokens, _, _ = x.shape
        if (
            x.dtype in [torch.float16, torch.bfloat16]
            and hasattr(torch.ops._C_ascend, "batch_matmul_transpose")
            and num_input_tokens <= BMM_TRANS_MAX_SUPPORTED_TOKENS
        ):
            x = x.view(-1, self.local_num_heads, self.kv_lora_rank)
            res = torch.empty((num_input_tokens, self.local_num_heads, self.v_head_dim), dtype=x.dtype, device=x.device)
            torch.ops._C_ascend.batch_matmul_transpose(x, self.W_UV, res)
            x = res.reshape(-1, self.local_num_heads * self.v_head_dim)
        elif hasattr(torch_npu, "npu_transpose_batchmatmul"):
            # Convert from (N, B, L)/(N, B, 1, L) to (N, B, L)
            x = x.view(-1, self.local_num_heads, self.kv_lora_rank)
            # Multiply (N, B, L) x (N, L, V) -> (B, N, V)
            x = torch_npu.npu_transpose_batchmatmul(x, self.W_UV, perm_x1=(1, 0, 2), perm_y=(1, 0, 2))
            # Convert from (N, B, V) to (B, N * V)
            x = x.reshape(-1, self.local_num_heads * self.v_head_dim)
        else:
            # Convert from (B, N, L) to (N, B, L)
            x = x.view(-1, self.local_num_heads, self.kv_lora_rank).transpose(0, 1)
            # # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
            x = torch.bmm(x, self.W_UV)
            # # Convert from (N, B, V) to (B, N * V)
            x = x.transpose(0, 1).reshape(-1, self.local_num_heads * self.v_head_dim)
        return x

    def _sfa_preprocess_with_mlapo(
        self,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        cos: torch.Tensor,
        sin: torch.Tensor,
        slot_mapping: torch.Tensor,
        num_input_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return DeviceOperator.sfa_preprocess_with_mlapo(
            self,
            hidden_states,
            kv_cache,
            cos,
            sin,
            slot_mapping,
            num_input_tokens,
        )

    def _sfa_preprocess_with_prolog_v3(
        self,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        cos: torch.Tensor,
        sin: torch.Tensor,
        slot_mapping: torch.Tensor,
        cache_mode: str,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        ql_nope, q_pe, _, q_c, q_c_scale = DeviceOperator.execute_sfa_mla_prolog_v3(
            self,
            hidden_states=hidden_states,
            rope_sin=sin,
            rope_cos=cos,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            cache_mode=cache_mode,
        )
        ql_nope = ql_nope.view(-1, self.local_num_heads, self.kv_lora_rank)
        q_pe = q_pe.view(-1, self.local_num_heads, self.qk_rope_head_dim)
        if self.has_indexer:
            if q_c is None:
                raise RuntimeError("npu_mla_prolog_v3 did not return query_norm for SFA indexer.")
            q_c = q_c.view(-1, self.q_lora_rank)
            if q_c_scale is not None and self.wq_b is not None and self._is_w8a8_dynamic_linear(self.wq_b):
                q_c = (q_c, q_c_scale.view(-1))
        else:
            q_c = None

        k_nope = kv_cache[0] if cache_mode == "TND" else None
        k_pe = kv_cache[1] if cache_mode == "TND" and not self.enable_sparse_sfa_c8 else None
        return hidden_states, ql_nope, q_pe, q_c, k_nope, k_pe

    def indexer_select_pre_process(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        if not self.has_indexer:
            raise RuntimeError(
                f"indexer_select_pre_process should not be called when indexer is None. layer_name={self.layer_name}."
            )

        assert self.wk_weights_proj is not None
        assert self.k_norm is not None

        kw, _ = self.wk_weights_proj(x)
        k_li = kw[:, : self.head_dim]
        k_li = self.k_norm(k_li).unsqueeze(1)
        k_li = k_li.view(-1, 1, self.head_dim)

        if HAS_TRITON:
            cos = cos.view(-1, self.qk_rope_head_dim)
            sin = sin.view(-1, self.qk_rope_head_dim)
            k_li = rope_forward_triton_siso(
                k_li, cos, sin, rope_dim=self.qk_rope_head_dim, is_neox_style=self.is_rope_neox_style
            )
        else:
            k_li_pe, k_li_nope = torch.split(
                k_li, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
            )

            cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
            sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)

            k_li_pe = k_li_pe.unsqueeze(2)
            k_li_pe = torch_npu.npu_rotary_mul(k_li_pe, cos, sin)
            k_li_pe = k_li_pe.squeeze(2)

            k_li = torch.cat([k_li_pe, k_li_nope], dim=-1)  # [b*s,128]

        if self.enable_sparse_li_c8:
            k_li = k_li @ AscendSFAImpl.k_hadamard
            k_li, k_li_scale = torch_npu.npu_dynamic_quant(k_li.view(-1, self.head_dim), dst_type=self.c8_k_cache_dtype)
            k_li_scale = k_li_scale.to(self.c8_k_scale_cache_dtype)  # [b*s,]
            k_li_scale = k_li_scale.unsqueeze(-1)  # [b*s,1]
        else:
            k_li_scale = None

        return k_li, k_li_scale

    def indexer_select_post_process(
        self,
        x: torch.Tensor,
        q_c: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: M | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        indexer_block_table_override: torch.Tensor | None = None,
        sparse_count: int = 2048,
    ):
        """Production indexer top-k (weights + multi-head query + RoPE).

        P9 staged SFA: the fixed layout passes ``attn_metadata=None`` with
        ``indexer_block_table_override`` carrying the indexer group's table
        (wrapped in the lightweight namespace the device operator reads);
        ``sparse_count`` defaults to the native 2048 (an explicit None would
        override the device adaptor's own default and crash the kernel) and
        the staged caller passes the model's index_topk. Provenance: fork
        sfa_v1.py:2209-2231 (override design).
        """
        if not self.has_indexer:
            raise RuntimeError(
                f"indexer_select_post_process should not be called when indexer is None. layer_name={self.layer_name}."
            )

        assert self.wk_weights_proj is not None
        assert self.wq_b is not None

        if indexer_block_table_override is not None:
            indexer_block_table = indexer_block_table_override
            # The device operator only reads the two block-table fields from
            # the metadata argument; supply the fixed-layout override table.
            attn_metadata = SimpleNamespace(
                indexer_block_table=indexer_block_table,
                block_table=indexer_block_table,
            )  # type: ignore[assignment]

        kw, _ = self.wk_weights_proj(x)
        weights = kw[:, self.head_dim :]
        if isinstance(q_c, tuple):
            q_c_tensor, q_c_scale = q_c
            q_c_tensor = q_c_tensor.view(-1, q_c_tensor.shape[-1])
            quant_matmul_kwargs = dict(
                bias=None,
                output_dtype=x.dtype,
            )
            if q_c_tensor.dtype == torch.float8_e4m3fn:
                if q_c_scale.dim() == 2:
                    q_c_scale = q_c_scale.view(q_c_scale.shape[0], -1, 2)
                quant_matmul_kwargs.update(
                    scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                    pertoken_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                    group_sizes=[1, 1, getattr(self.wq_b.quant_method.quant_method, "group_size", 32)],
                )
            elif q_c_scale.dim() > 1 and q_c_scale.shape[-1] == 1:
                q_c_scale = q_c_scale.squeeze(dim=-1)
            q_li = torch_npu.npu_quant_matmul(
                q_c_tensor,
                self.wq_b.weight,
                self.wq_b.weight_scale,
                pertoken_scale=q_c_scale,
                **quant_matmul_kwargs,
            )
        else:
            q_li, _ = self.wq_b(q_c)
        q_li = q_li.view(-1, self.n_head, self.head_dim)
        if HAS_TRITON:
            q_li = rope_forward_triton_siso(
                q_li, cos, sin, rope_dim=self.qk_rope_head_dim, is_neox_style=self.is_rope_neox_style
            )
        else:
            q_li_pe, q_li_nope = torch.split(
                q_li, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
            )

            q_li_pe = q_li_pe.unsqueeze(2)
            q_li_pe = torch_npu.npu_rotary_mul(q_li_pe, cos, sin)
            q_li_pe = q_li_pe.squeeze(2)
            q_li = torch.cat([q_li_pe, q_li_nope], dim=-1)

        q_li_scale = None
        q_li_shape_ori = None
        if self.enable_sparse_li_c8:
            q_li_shape_ori = q_li.shape
            q_li = q_li @ AscendSFAImpl.q_hadamard
            q_li, q_li_scale = torch_npu.npu_dynamic_quant(q_li.view(-1, self.head_dim), dst_type=self.c8_k_cache_dtype)
            q_li_scale = q_li_scale.to(self.c8_k_scale_cache_dtype)  # [b*s,]

        record_attention_compute_start()
        return DeviceOperator.indexer_select_post_process(
            self,
            q_li,
            q_li_scale,
            q_li_shape_ori,
            weights,
            kv_cache,
            attn_metadata,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            self.enable_sparse_li_c8,
            self.use_torch_npu_lightning_indexer,
            sparse_count=sparse_count,
        )

    def _get_indexcache_topk_indices(self, num_tokens: int) -> torch.Tensor:
        if self.topk_indices_buffer is None:
            raise RuntimeError("IndexCache requires topk_indices_buffer when skip_topk is enabled.")
        topk_indices = self.topk_indices_buffer[:num_tokens]
        if topk_indices.dim() == 2:
            topk_indices = topk_indices.unsqueeze(1)
        return topk_indices

    def _update_indexcache_topk_indices(self, topk_indices: torch.Tensor) -> None:
        if self.topk_indices_buffer is None:
            return
        num_tokens = topk_indices.shape[0]
        topk_tokens = topk_indices.shape[-1]
        topk_indices_to_cache = topk_indices
        topk_indices_buffer = self.topk_indices_buffer[:num_tokens, :topk_tokens]
        if topk_indices_to_cache.dim() == 3 and topk_indices_buffer.dim() == 2:
            assert topk_indices_to_cache.shape[1] == 1
            topk_indices_to_cache = topk_indices_to_cache.squeeze(1)
        topk_indices_buffer.copy_(topk_indices_to_cache)

    def _execute_sparse_flash_attention_process(
        self, ql_nope, q_pe, kv_cache, topk_indices, attn_metadata, actual_seq_lengths_query, actual_seq_lengths_key
    ):
        return DeviceOperator.execute_sparse_flash_attention_process(
            self,
            ql_nope,
            q_pe,
            kv_cache,
            topk_indices,
            attn_metadata,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
        )

    def _record_dcp_query_gather_context(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        attn_metadata: M,
    ) -> None:
        return

    def _dsa_skip_dense_layer_wait(self, attn_metadata: M) -> bool:
        """DSA shrink (B2d): skip the pre-attention dense layer wait.

        Under shrink with decode rows in the batch, the dense
        wait_for_kv_layer call would advance the layerwise retriever once
        more than the selective retrieve path expects (double advance ->
        desync). The selective wait driven by the remap outputs becomes the
        only retrieve driver for those steps. Provenance: fork
        sfa_v1.py:3299-3305.
        """
        return bool(
            self.dsa_shrink_latent
            and attn_metadata.num_decode_tokens > 0
            and (
                attn_metadata.need_sparse_lmcache_payload
                or self.dsa_shrink_latent == 3
            )
        )

    def _record_dcp_kv_gather_context(
        self,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: M,
    ) -> None:
        """Start a DCP KV gather after this layer has populated its cache.

        The base implementation deliberately does nothing. The replicated-indexer
        DCP implementation overrides it for batches containing prefill requests.
        """
        return

    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: M,
        need_gather_q_kv: bool = False,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            # Profiling run.
            if self.enable_dsa_cp_with_layer_shard and not _EXTRA_CTX.in_profile_run:
                for layer in self.layer_sharding_kwargs or []:
                    if is_hidden_layer(layer):
                        reach_layer_for_shard_weight_series(layer)
            return output.fill_(0)

        kv_cache = self._dsa_reassemble_kv_cache(layer_name, kv_cache)

        cos = attn_metadata.cos
        sin = attn_metadata.sin
        slot_mapping = attn_metadata.slot_mapping
        slot_mapping_cp = None
        if self.enable_dsa_cp:
            assert attn_metadata.dsa_cp_context is not None
            slot_mapping_cp = attn_metadata.dsa_cp_context.slot_mapping_cp
            actual_seq_lengths_query = attn_metadata.dsa_cp_context.actual_seq_lengths_query
            actual_seq_lengths_key = attn_metadata.dsa_cp_context.actual_seq_lengths_key
        else:
            actual_seq_lengths_query = attn_metadata.cum_query_lens
            actual_seq_lengths_key = attn_metadata.seq_lens
        # DCP replicated indexer stores LI cache with the full/no-CP metadata, while
        # SFA KV remains stored with the DCP-sharded slot mapping.
        slot_mapping_sfa = (
            attn_metadata.dcp_context.slot_mapping
            if attn_metadata.dcp_context is not None
            else attn_metadata.slot_mapping
        )
        # DSA two-groups: the indexer key cache lives in the indexer group's
        # own block pool; its writes must use the indexer group's slot
        # mapping. Falls back to the shared (latent) slot mapping in
        # unbundle-only mode. Never use `or` on tensors (ambiguous truth
        # value). Fork semantics: F-ascend :3243-3247.
        idx_slot_mapping = (
            attn_metadata.indexer_slot_mapping
            if attn_metadata.indexer_slot_mapping is not None
            else slot_mapping
        )
        if envs.VLLM_ASCEND_DSA_DEBUG and not AscendSFAImpl._dsa_debug_logged:
            # One-shot runtime evidence that the two-group tables are live:
            # whether the indexer mirror exists, the first few block ids of
            # each table (distinct id spaces = independent pools), and which
            # slot mapping the indexer writes use.
            AscendSFAImpl._dsa_debug_logged = True
            indexer_bt = getattr(attn_metadata, "indexer_block_table", None)
            if indexer_bt is not None:
                logger.info(
                    "[DSA_DEBUG] layer=%s two_groups tables live: "
                    "latent_bt[:1]=%s indexer_bt[:1]=%s "
                    "idx_slot_is_shared=%s",
                    layer_name,
                    attn_metadata.block_table[:1, :4].tolist(),
                    indexer_bt[:1, :4].tolist(),
                    idx_slot_mapping is slot_mapping,
                )
            else:
                logger.info(
                    "[DSA_DEBUG] layer=%s indexer tables absent "
                    "(unbundle-only shared id space); idx_slot_is_shared=True",
                    layer_name,
                )

        # Inputs and outputs may be padded for CUDA graphs
        num_input_tokens = attn_metadata.num_input_tokens
        output_padded = output

        # all-gather o_proj weight for prefill stage of PD mix node
        o_proj_full_handle = None
        o_proj_full_param_handles = None
        # Prefill/mixed DSA-CP computes o_proj with a temporary full weight.
        # Decode keeps the original TP path and only exchanges activations.
        full_gather_o_proj_enabled = self.enable_dsa_cp_with_o_proj_tp and attn_metadata.attn_state not in {
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.SpecDecoding,
        }

        if self.enable_sfa_prolog_v3 and attn_metadata.attn_state in (
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.SpecDecoding,
        ):
            if self.enable_sp:
                hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                    hidden_states.contiguous(), need_gather_q_kv
                )
            assert slot_mapping.numel() == hidden_states.shape[0], (
                "SFA Prolog V3 requires one cache index per input token, "
                f"got token_x={hidden_states.shape[0]} and cache_index={slot_mapping.numel()}."
            )
            if self.has_indexer:
                k_li, k_li_scale = self.indexer_select_pre_process(x=hidden_states, cos=cos, sin=sin)
            else:
                k_li, k_li_scale = None, None

            # Prolog updates the paged KV cache in place. Wait for the prompt
            # blocks before writing the first Decode token into their tail block.
            if not self._dsa_skip_dense_layer_wait(attn_metadata):
                wait_for_kv_layer_from_connector(layer_name)
            hidden_states, ql_nope, q_pe, q_c, _, _ = self._sfa_preprocess_with_prolog_v3(
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                cos=cos,
                sin=sin,
                slot_mapping=slot_mapping,
                cache_mode="PA_BSND",
            )
        # run mlapo ops when dsa-cp is disabled, and ensure that num_tokens satisfies the count limitation
        elif self.enable_mlapo and (
            get_ascend_device_type() == AscendDeviceType.A5 or num_input_tokens <= MLAPO_MAX_SUPPORTED_TOKENS
        ):
            hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                hidden_states.contiguous(), need_gather_q_kv
            )
            hidden_states, ql_nope, q_pe, q_c = self._sfa_preprocess_with_mlapo(
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                cos=cos,
                sin=sin,
                slot_mapping=slot_mapping,
                num_input_tokens=num_input_tokens,
            )
            if self.has_indexer:
                k_li, k_li_scale = self.indexer_select_pre_process(
                    x=hidden_states,
                    cos=cos,
                    sin=sin,
                )
            else:
                k_li, k_li_scale = None, None
            if not self._dsa_skip_dense_layer_wait(attn_metadata):
                wait_for_kv_layer_from_connector(layer_name)
        # native
        else:
            assert self.fused_qkv_a_proj is not None, "q lora is required for DSA."
            weight_prefetch_method = get_weight_prefetch_method()
            weight_prefetch_method.maybe_prefetch_mla_or_sla_weight_in_current_stream(
                inputs=self.fused_qkv_a_proj.weight, dependency=hidden_states
            )
            if self.enable_sp and not self.enable_dsa_cp:
                hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                    hidden_states.contiguous(), need_gather_q_kv
                )
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_no_split = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            assert self.q_a_layernorm is not None, "q_a_layernorm must be initialized"
            q_c = self.q_a_layernorm(q_c)

            if self.has_indexer:
                k_li, k_li_scale = self.indexer_select_pre_process(
                    x=hidden_states,
                    cos=cos,
                    sin=sin,
                )
            else:
                k_li, k_li_scale = None, None

            if not self._dsa_skip_dense_layer_wait(attn_metadata):
                wait_for_kv_layer_from_connector(layer_name)

            if self.enable_dsa_cp:
                assert slot_mapping_cp is not None
                kv_slots = slot_mapping_cp
            else:
                kv_slots = slot_mapping_sfa
            kv_outputs = self.exec_kv(kv_no_split, cos, sin, kv_cache, kv_slots, attn_metadata)
            k_pe, k_nope = kv_outputs[:2]
            knope_scale = kv_outputs[2] if len(kv_outputs) == 3 else None

            if (
                self.enable_sparse_sfa_c8
                and not self.enable_dsa_cp
                and (get_ascend_device_type() != AscendDeviceType.A5 or not self.has_indexer)
            ):
                assert k_pe is not None
                assert k_nope is not None
                assert knope_scale is not None
                packed_kv = torch.cat([k_nope, k_pe, knope_scale], dim=-1)
                packed_head_dim = self.sfa_qsfa_packed_kv_head_dim
                assert packed_kv.shape[-1] == packed_head_dim
                torch_npu.npu_scatter_nd_update_(
                    kv_cache[0].view(-1, packed_head_dim),
                    slot_mapping_sfa.view(-1, 1),
                    packed_kv.view(-1, packed_head_dim),
                )

            if self.enable_dsa_cp:
                assert k_pe is not None
                assert k_nope is not None
                async_op = self.enable_dsa_cp_with_layer_shard or full_gather_o_proj_enabled
                kv_ag_handles = []
                # Pack all KV-related tensors into one byte stream so DSA-CP only
                # submits one KV all-gather while still preserving original dtypes.
                if self.enable_sparse_sfa_c8:
                    assert knope_scale is not None
                    fused_kv_parts = [
                        k_nope.view(-1, k_nope.shape[-1]),
                        k_pe.view(-1, k_pe.shape[-1]),
                        knope_scale.view(-1, knope_scale.shape[-1]),
                    ]
                else:
                    fused_kv_parts = [
                        k_pe.view(-1, k_pe.shape[-1]),
                        k_nope.view(-1, k_nope.shape[-1]),
                    ]

                fused_kv_input = torch.cat(fused_kv_parts, dim=1)
                kv_gather_parts = [("sfa_kv", fused_kv_input)]
                if self.has_indexer:
                    assert k_li is not None
                    k_li_gather_input = k_li
                    if not self.enable_sparse_sfa_c8 and not self.enable_sparse_li_c8:
                        k_li_gather_input = k_li.view(-1, k_li.shape[-1])
                    kv_gather_parts.append(("k_li", k_li_gather_input))
                if self.has_indexer and self.enable_sparse_li_c8:
                    assert k_li_scale is not None
                    kv_gather_parts.append(("k_li_scale", k_li_scale))

                kv_gathered_bytes, kv_ag_handle, kv_gather_metadata = self._all_gather_byte_packed_async(
                    kv_gather_parts,
                    async_op=async_op,
                )
                if kv_ag_handle is not None:
                    kv_ag_handles.append(kv_ag_handle)

            ql_nope, q_pe = self._q_proj_and_k_up_proj(q_c)
            q_pe = self.rope_single(q_pe, cos, sin)
            self._record_dcp_query_gather_context(ql_nope, q_pe, attn_metadata)

            if self.enable_dsa_cp:
                for kv_ag_handle in kv_ag_handles:
                    kv_ag_handle.wait()
                kv_gather_outputs = self._restore_byte_gathered_tensors(kv_gathered_bytes, kv_gather_metadata)
                fused_kv_no_split = kv_gather_outputs["sfa_kv"]
                if self.has_indexer:
                    k_li = kv_gather_outputs["k_li"]
                    if self.enable_sparse_li_c8:
                        k_li_scale = kv_gather_outputs["k_li_scale"]

                if self.enable_dsa_cp_with_layer_shard:
                    for layer in self.layer_sharding_kwargs or []:
                        if is_hidden_layer(layer):
                            reach_layer_for_shard_weight_series(layer)
                elif full_gather_o_proj_enabled:
                    _, o_proj_full_handle = all_gather_async(
                        self.o_proj_tp_weight_gather_input,
                        get_tp_group(),
                        output=self.o_proj_full_gather_pool,
                    )
                    o_proj_full_param_handles = []
                    for param_name, param in self.o_proj_tp_input_sharded_quant_params.items():
                        _, param_handle = all_gather_async(
                            param,
                            get_tp_group(),
                            output=self.o_proj_full_input_sharded_quant_params[param_name],
                        )
                        o_proj_full_param_handles.append(param_handle)

                if kv_cache is not None:
                    assert fused_kv_no_split is not None
                    if self.enable_sparse_sfa_c8:
                        torch_npu.npu_scatter_nd_update_(
                            kv_cache[0].view(-1, fused_kv_no_split.shape[-1]),
                            slot_mapping_sfa[: attn_metadata.num_actual_tokens].view(-1, 1),
                            fused_kv_no_split[: attn_metadata.num_actual_tokens],
                        )
                        k_pe = None
                        k_nope = None
                    else:
                        k_pe, k_nope = fused_kv_no_split.split(
                            [self.qk_rope_head_dim, self.kv_lora_rank],
                            dim=-1,
                        )
                    if not self.enable_sparse_sfa_c8:
                        assert k_pe is not None
                        assert k_nope is not None
                        k_nope = k_nope.view(k_nope.shape[0], 1, -1)
                        k_pe = k_pe.view(k_pe.shape[0], 1, -1)
                        DeviceOperator.reshape_and_cache(
                            key=k_nope[: attn_metadata.num_actual_tokens],
                            value=k_pe[: attn_metadata.num_actual_tokens],
                            key_cache=kv_cache[0],
                            value_cache=kv_cache[1],
                            slot_mapping=slot_mapping_sfa[: attn_metadata.num_actual_tokens],
                        )

            # DCP's prefill path may all-gather only the blocks referenced by
            # this batch. It must start after the current layer's SFA KV write,
            # but before the indexer/top-k work so communication can overlap it.
            if kv_cache is not None:
                self._record_dcp_kv_gather_context(kv_cache, attn_metadata)

            if self.has_indexer:
                assert k_li is not None
                k_li = self._get_full_kv(k_li, attn_metadata)

        self._dsa_maybe_wait_indexer_rows(layer_name, kv_cache)

        if kv_cache is not None and self.is_kv_producer:
            attn_metadata.reshape_cache_event = torch.npu.Event()

        if kv_cache is not None and self.has_indexer:
            assert k_li is not None
            if self.enable_sparse_sfa_c8:
                dsa_k_cache_idx = 1
                dsa_k_scale_cache_idx = 2
            else:
                dsa_k_cache_idx = 2
                dsa_k_scale_cache_idx = 3

            if get_ascend_config().c8_enable_reshape_optim:
                torch.ops._C_ascend.store_kv_block(
                    k_li,
                    kv_cache[dsa_k_cache_idx],
                    attn_metadata.group_len,
                    attn_metadata.group_key_idx,
                    attn_metadata.group_key_cache_idx,
                    attn_metadata.block_size,
                )
            else:
                torch_npu.npu_scatter_nd_update_(
                    kv_cache[dsa_k_cache_idx].view(-1, k_li.shape[-1]),
                    idx_slot_mapping.view(-1, 1),
                    k_li.view(-1, k_li.shape[-1]),
                )  # b, s, n, d
            if self.enable_sparse_li_c8:
                assert len(kv_cache) == (3 if self.enable_sparse_sfa_c8 else 4)
                if k_li_scale is not None:
                    if get_ascend_config().c8_enable_reshape_optim:
                        torch.ops._C_ascend.store_kv_block(
                            k_li_scale,
                            kv_cache[dsa_k_scale_cache_idx],
                            attn_metadata.group_len,
                            attn_metadata.group_key_idx,
                            attn_metadata.group_key_cache_idx,
                            attn_metadata.block_size,
                        )
                    else:
                        torch_npu.npu_scatter_nd_update_(
                            kv_cache[dsa_k_scale_cache_idx].view(-1, k_li_scale.shape[-1]),
                            idx_slot_mapping.view(-1, 1),
                            k_li_scale.view(-1, k_li_scale.shape[-1]),
                        )

        if kv_cache is not None and self.is_kv_producer:
            attn_metadata.reshape_cache_event.record()
            notify_kv_cache_written(self.layer_name or "")

        if self.enable_dsa_cp and attn_metadata.dsa_cp_context is not None:
            topk_num_tokens = attn_metadata.dsa_cp_context.local_end_with_pad - attn_metadata.dsa_cp_context.local_start
        else:
            topk_num_tokens = num_input_tokens or hidden_states.shape[0]
        if self.skip_topk:
            topk_indices = self._get_indexcache_topk_indices(topk_num_tokens)
        else:
            if not self.has_indexer:
                raise RuntimeError(f"skip_topk is False but indexer is None. layer_name={self.layer_name}.")
            assert q_c is not None
            topk_indices = self.indexer_select_post_process(
                x=hidden_states,
                q_c=q_c,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                cos=cos,
                sin=sin,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
            )
            if self.use_index_cache:
                self._update_indexcache_topk_indices(topk_indices)

        # DSA shrink replay (B2d): remap main chain. Resolve the proven
        # frontier, overwrite the per-row boundary in place, then dispatch
        # the remap kernel (B2e registers the NPU op; the dispatch module
        # imports lazily so the Python-only batches stay importable before
        # the csrc rebuild). Outputs: indices rewritten to compact scratch
        # rows below the boundary (>= boundary keeps absolute positions),
        # selected_packed / counts / target_slots for the selective
        # retrieve. Padding rows are zeroed so freed blocks are never
        # referenced. Provenance: fork sfa_v1.py:3451-3546.
        _sel_packed = None
        _target_slots = None
        _sel_counts = None
        if (
            self.dsa_shrink_latent
            and attn_metadata.split_boundary is not None
            and attn_metadata.num_decode_tokens > 0
            and (
                attn_metadata.need_sparse_lmcache_payload
                or self.dsa_shrink_latent == 3
            )
        ):
            from vllm_ascend.distributed.kv_transfer.sparse_offload.prepare_sparse_indices import (
                prepare_sparse_indices,
            )

            if self.dsa_shrink_latent != 3:
                _cached_split_boundary = attn_metadata.decode_split_boundary
                if (
                    _cached_split_boundary is not None
                    and _cached_split_boundary.shape == attn_metadata.split_boundary.shape
                    and _cached_split_boundary.device == attn_metadata.split_boundary.device
                    and _cached_split_boundary.dtype == attn_metadata.split_boundary.dtype
                ):
                    # Step metadata, not layer data: the request set and
                    # frontier resolution are identical for every SFA layer
                    # in this step, so reuse the tensor cached by the first
                    # SFA layer. Provenance: fork sfa_v1.py:3479-3496.
                    attn_metadata.split_boundary = _cached_split_boundary
                else:
                    _cached = _resolve_sparse_cached_tokens_by_request(
                        attn_metadata, attn_metadata.req_ids
                    )
                    _decode_window_size = _decode_window_save_window_size()
                    if _cached is not None or _decode_window_size > 0:
                        _update_dsa_split_boundary_in_place(
                            attn_metadata,
                            _cached,
                            _decode_window_size,
                        )
                    else:
                        # Cache the static boundary (no LMCache or decode
                        # window component) so the remaining SFA layers in
                        # this step skip the resolution entirely.
                        # Provenance: fork sfa_v1.py:3510-3511.
                        attn_metadata.decode_split_boundary = (
                            attn_metadata.split_boundary
                        )
            # Stage 3 is an isolation diagnostic by design: keep the
            # builder's prompt boundary, remap and run FA on uninitialized
            # scratch, but do NOT query/call LMCache. Crash => remap/FA;
            # clean (wrong output expected) => transfer path is implicated.
            _row_req_indices = attn_metadata.decode_req_indices
            _is_pure_decode = attn_metadata.attn_state in (
                AscendAttentionState.DecodeOnly,
                AscendAttentionState.SpecDecoding,
            )
            if _is_pure_decode:
                topk_indices, _topk_2d = _dsa_mask_padding_sparse_rows(
                    topk_indices,
                    _row_req_indices,
                )
            else:
                _topk_2d = _dsa_topk_to_2d_indices(topk_indices)
            _row_req_cpu = np.asarray(
                attn_metadata.decode_req_indices_cpu,
                dtype=np.int32,
            )[: int(_topk_2d.shape[0])]
            _valid_req = _row_req_cpu[_row_req_cpu >= 0]
            _rows_per_req = (
                int(np.bincount(_valid_req).max())
                if _valid_req.size
                else 0
            )
            _required_union_capacity = (
                _rows_per_req * int(_topk_2d.shape[1])
            )
            if _required_union_capacity > self.dsa_scratch_capacity:
                raise RuntimeError(
                    "DSA scratch capacity is smaller than the per-request "
                    "top-k union upper bound: "
                    f"rows_per_request={_rows_per_req}, "
                    f"topk_width={int(_topk_2d.shape[1])}, "
                    f"required={_required_union_capacity}, "
                    f"capacity={self.dsa_scratch_capacity}."
                )
            _table_capacity = (
                int(attn_metadata.block_table.shape[1])
                * int(attn_metadata.block_size)
            )
            if self.dsa_scratch_capacity > _table_capacity:
                raise RuntimeError(
                    "DSA scratch reservation exceeds the request block-table "
                    f"capacity: scratch={self.dsa_scratch_capacity}, "
                    f"table_capacity={_table_capacity}."
                )
            (
                topk_indices,
                _sel_packed,
                _sel_counts,
                _target_slots,
            ) = prepare_sparse_indices(
                topk_indices,
                attn_metadata.decode_split_boundary,
                attn_metadata.block_table,
                attn_metadata.block_size,
                kv_cache[0].device,
                row_req_indices=_row_req_indices,
                scratch_capacity=self.dsa_scratch_capacity,
                clear_invalid_rows=_is_pure_decode,
            )
            attn_metadata.decode_selected_tokens = _sel_packed
            attn_metadata.decode_selected_counts = _sel_counts
            attn_metadata.decode_target_slot_mapping = _target_slots

        # DSA shrink replay (B2d): selective retrieve. When the remap
        # produced a non-empty selected list, wait for exactly those tokens
        # and scatter them into the request's scratch slots (LMCache
        # selective load). Stage 3 (isolation diagnostics) skips the
        # retrieve on purpose. One-shot stream fence after the first
        # selective wait closes the first-hit race. Provenance: fork
        # sfa_v1.py:3915-3932.
        if (
            _sel_packed is not None
            and self.dsa_shrink_latent != 3
            and int(_sel_packed.numel()) > 0
        ):
            wait_for_kv_layer_from_connector(
                layer_name,
                selected_tokens=_sel_packed,
                request_ids=attn_metadata.req_ids,
                target_slot_mapping=_target_slots,
                selected_token_counts=_sel_counts,
            )
            _sync_compute_stream_after_lmcache_sparse_wait()

        # The selective retrieve must finish before FA consumes the remapped
        # scratch rows. Running FA first reads stale/uninitialized scratch.
        attn_output = self._execute_sparse_flash_attention_process(
            ql_nope,
            q_pe,
            kv_cache,
            topk_indices,
            attn_metadata,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
        )

        attn_output = self._v_up_proj(attn_output)
        weight_prefetch_method = get_weight_prefetch_method()
        weight_prefetch_method.maybe_prefetch_mla_or_sla_weight_in_current_stream(
            inputs=self.o_proj.weight,
            dependency=attn_output,
            max_size=MAX_O_PROJ_PREFETCH_SIZE,
            linear_layer=self.o_proj,
        )

        if self.enable_dsa_cp_with_o_proj_tp:
            # SFA DSA-CP mixed mode keeps o_proj weight sharded in the TP domain:
            # 1. prefill/mixed: gather TP shards into a temporary full weight.
            # 2. decode-only: all-to-all hidden states, then run TP o_proj.
            result, require_o_proj_forward = self._handle_o_proj_weight_switch_and_forward(
                attn_output=attn_output,
                output=output,
                o_proj_full_handle=o_proj_full_handle,
                o_proj_full_param_handles=o_proj_full_param_handles,
                should_shard_weight=full_gather_o_proj_enabled,
            )
            if not require_o_proj_forward:
                return result
            attn_output = result

        output[...] = self.o_proj(attn_output)[0]

        self._maybe_save_unbundled_kv_cache(layer_name, kv_cache, attn_metadata)

        return output_padded


def custom_kv_rmsnorm_rope(
    kv: torch.Tensor,
    gamma: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    *,
    epsilon: float = 1e-5,
    dst_type: torch.dtype | int = torch.float8_e4m3fn,
    tile_size: int = SFA_QSFA_TILE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rms_in, rope_in = kv.split([kv_lora_rank, qk_rope_head_dim], dim=-1)
    k_nope, _ = torch_npu.npu_rms_norm(rms_in, gamma, epsilon=epsilon)
    k_rope = torch_npu.npu_interleave_rope(rope_in, cos, sin)

    prefix_shape = k_nope.shape[:-1]
    k_nope, knope_scale = torch_npu.npu_dynamic_block_quant(
        k_nope.contiguous().view(-1, 1, kv_lora_rank),
        dst_type=dst_type,
        row_block_size=1,
        col_block_size=tile_size,
    )
    if dst_type == 1 or dst_type == torch.int8:
        # Return byte views so the caller can concatenate all three components.
        return (
            k_rope.contiguous().view(torch.int8),
            k_nope.view(*prefix_shape, kv_lora_rank),
            knope_scale.to(torch.float32).view(*prefix_shape, -1).contiguous().view(torch.int8),
        )

    # A5 transports the BF16 rope and scale bytes through FP8-typed tensors.
    return (
        k_rope.view(torch.float8_e4m3fn),
        k_nope,
        knope_scale.view(knope_scale.shape[0], -1).view(torch.float8_e4m3fn),
    )
