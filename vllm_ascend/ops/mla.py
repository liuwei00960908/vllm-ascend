# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionMetadata  # type: ignore

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.utils import is_vl_model, parse_layer_idx


class IndexerWrapper(nn.Module):
    """
    A wrapper of Indexer for Deepseek v3.2.
    This wrapper is currently used to solve the fp8 hard code issue of vllm's deepseek_v2.py.
    It wraps the original Indexer, inherits its module weights
    (including wq_b, wk_weights_proj or wk/weights_proj, k_norm)
    while deletes the unused topk_indices_buffer and k_cache to save memory.
    TODO: Will be removed once original Indexer supports different quantization methods.
    """

    def __init__(self, vllm_indexer: nn.Module) -> None:
        super().__init__()

        self.n_head: int = vllm_indexer.n_head  # 64
        self.head_dim: int = vllm_indexer.head_dim  # 128
        self.topk_tokens: int = vllm_indexer.topk_tokens  # 2048
        self.q_lora_rank: int = vllm_indexer.q_lora_rank  # 1536
        self.wq_b = vllm_indexer.wq_b
        self.wk_weights_proj = vllm_indexer.wk_weights_proj
        self.k_norm = vllm_indexer.k_norm
        self.softmax_scale = vllm_indexer.softmax_scale
        vllm_indexer.topk_indices_buffer = None  # delete topk_indices_buffer
        vllm_indexer.k_cache = None  # delete k_cache

    def forward(self):
        return


class AscendMultiHeadLatentAttention(MultiHeadLatentAttentionWrapper):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        skip_topk: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.prefix = prefix
        self.skip_topk = skip_topk
        hf_config = get_current_vllm_config().model_config.hf_text_config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.layers = hf_config.num_hidden_layers
        if mla_modules.indexer is not None:
            ascend_indexer = IndexerWrapper(mla_modules.indexer)
        else:
            ascend_indexer = None
        self.mla_attn = MLAAttention(
            num_heads=num_heads,
            scale=scale,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            kv_b_proj=mla_modules.kv_b_proj,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            use_sparse=mla_modules.is_sparse,
            indexer=ascend_indexer,
            skip_topk=skip_topk,
            topk_indices_buffer=getattr(mla_modules, "topk_indices_buffer", None),
            # extra args
            rotary_emb=mla_modules.rotary_emb,
            fused_qkv_a_proj=mla_modules.fused_qkv_a_proj,
            q_b_proj=mla_modules.q_b_proj,
            q_a_layernorm=mla_modules.q_a_layernorm,
            q_proj=mla_modules.q_proj,
            kv_a_proj_with_mqa=mla_modules.kv_a_proj_with_mqa,
            kv_a_layernorm=mla_modules.kv_a_layernorm,
            o_proj=mla_modules.o_proj,
            layer_name=f"{prefix}.attn",
        )

        original_process_weights = self.mla_attn.process_weights_after_loading

        def wrapped_process_weights(act_dtype: torch.dtype):
            from vllm_ascend.attention.sfa_v1 import AscendSFAImpl

            if not isinstance(self.mla_attn.impl, AscendSFAImpl):
                original_process_weights(act_dtype)
            self.mla_attn.impl.process_weights_after_loading(act_dtype)

        self.mla_attn.process_weights_after_loading = wrapped_process_weights

        # For VL models (e.g. Kimi K2.5), inputs_embeds at layer 0 comes from
        # the vision encoder as full [N, H] — it has NOT been reduce-scattered.
        # We detect this statically at init time (not at runtime via shape checks,
        # which break graph-mode compilation) so the branch is a constant to dynamo.
        vllm_config = get_current_vllm_config()
        _is_vl = is_vl_model(vllm_config)
        _layer_idx = parse_layer_idx(prefix)
        self.is_vl_first_layer = bool(_is_vl and _layer_idx == 0)

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor | None = None,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        hidden_dim = self.hidden_size

        if _EXTRA_CTX.flash_comm_v1_enabled and self.tp_size > 1 and self.is_vl_first_layer:
            need_gather_q_kv = False
            n_out = hidden_states.shape[0] // self.tp_size
            output = torch.empty((n_out, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
        else:
            need_gather_q_kv = _EXTRA_CTX.flash_comm_v1_enabled
            output = torch.empty(
                (hidden_states.shape[0], hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )

        torch.ops.vllm.mla_forward(hidden_states, need_gather_q_kv, output, self.prefix)
        output = output.view(-1, hidden_dim)
        return output


def mla_forward(
    hidden_states: torch.Tensor,
    need_gather_q_kv: bool,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    if forward_context.attn_metadata:
        attn_metadata = forward_context.attn_metadata[self.mla_attn.layer_name]
    else:
        attn_metadata = forward_context.attn_metadata
    kv_cache = self.mla_attn.kv_cache
    self.mla_attn.impl.forward(
        self.mla_attn.layer_name, hidden_states, kv_cache, attn_metadata, need_gather_q_kv, output
    )
    return


def mla_forward_fake(
    hidden_states: torch.Tensor,
    need_gather_q_kv: bool,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="mla_forward",
    op_func=mla_forward,
    mutates_args=["output"],
    fake_impl=mla_forward_fake,
    dispatch_key="PrivateUse1",
)


# ---------------------------------------------------------------------------
# P9 staged SFA graph (Batch 4): three custom ops that form the graph-A /
# retrieve-window / graph-B split boundaries. The compiler treats each op as
# an opaque unit; pre/post carry the bridge outputs as inputs/outputs (the
# data-flow anchor that prevents reordering — 03-4 §3.4), while the retrieve
# op's fake returns nothing and naturally becomes the eager hole between the
# two captured pieces. Provenance: fork ops/mla.py:245-445.
# ---------------------------------------------------------------------------

StagedSFABridge = tuple[
    torch.Tensor,  # 0: ql_nope (max_tokens, heads, kv_lora_rank)
    torch.Tensor,  # 1: q_pe (max_tokens, heads, qk_rope_head_dim)
    torch.Tensor,  # 2: topk_indices (max_tokens, 1, index_topk) int32
    torch.Tensor,  # 3: selected_packed (max_requests, scratch) int32
    torch.Tensor,  # 4: selected_counts (max_requests,) int32
    torch.Tensor,  # 5: target_slots (max_requests, scratch) int64
]


def _mla_runtime_metadata(layer_name: str):
    """Resolve forward_context → layer → attn_metadata."""
    forward_context = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    attn_metadata = (
        forward_context.attn_metadata[layer.mla_attn.layer_name]
        if forward_context.attn_metadata
        else forward_context.attn_metadata
    )
    return forward_context, layer, attn_metadata


def _mla_runtime_state(layer_name: str):
    """Resolve impl + kv_cache from the layer's runtime state."""
    forward_context, layer, attn_metadata = _mla_runtime_metadata(layer_name)
    return (
        layer.mla_attn.impl,
        layer.mla_attn.layer_name,
        layer.mla_attn.kv_cache,
        attn_metadata,
    )


def sfa_forward_pre(
    hidden_states: torch.Tensor,
    need_gather_q_kv: bool,
    output: torch.Tensor,
    layer_name: str,
    local_num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    index_topk: int,
    token_capacity: int,
    request_capacity: int,
    scratch_capacity: int,
) -> StagedSFABridge:
    impl, attn_layer_name, kv_cache, attn_metadata = _mla_runtime_state(layer_name)
    return impl.cross_layer_graph_pre(
        attn_layer_name,
        hidden_states,
        kv_cache,
        attn_metadata,
        need_gather_q_kv,
        output,
    )


def sfa_forward_pre_fake(
    hidden_states: torch.Tensor,
    need_gather_q_kv: bool,
    output: torch.Tensor,
    layer_name: str,
    local_num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    index_topk: int,
    token_capacity: int,
    request_capacity: int,
    scratch_capacity: int,
) -> StagedSFABridge:
    # Shape inference for the graph-A split: six empty tensors whose shapes
    # exactly match the bridge slots (03-4 §3.2). The scalar capacity
    # parameters are baked into the captured graph by value — this is why
    # one graph key serves one capacity and shape changes route to a
    # different entry.
    return (
        hidden_states.new_empty((token_capacity, local_num_heads, kv_lora_rank)),
        hidden_states.new_empty((token_capacity, local_num_heads, qk_rope_head_dim)),
        torch.empty(
            (token_capacity, 1, index_topk),
            dtype=torch.int32,
            device=hidden_states.device,
        ),
        torch.empty(
            (request_capacity, scratch_capacity),
            dtype=torch.int32,
            device=hidden_states.device,
        ),
        torch.empty(
            (request_capacity,),
            dtype=torch.int32,
            device=hidden_states.device,
        ),
        torch.empty(
            (request_capacity, scratch_capacity),
            dtype=torch.long,
            device=hidden_states.device,
        ),
    )


def sfa_lmcache_retrieve(
    selected_packed: torch.Tensor,
    selected_counts: torch.Tensor,
    target_slots: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    next_layer_name: str,
) -> None:
    context, layer, attn_metadata = _mla_runtime_metadata(layer_name)
    layer.mla_attn.impl.cross_layer_lmcache_retrieve(
        layer.mla_attn.layer_name,
        next_layer_name,
        selected_packed,
        selected_counts,
        target_slots,
        attn_metadata,
        context,
    )


def sfa_lmcache_retrieve_fake(
    selected_packed: torch.Tensor,
    selected_counts: torch.Tensor,
    target_slots: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    next_layer_name: str,
) -> None:
    return


def sfa_forward_post(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    topk_indices: torch.Tensor,
    selected_packed: torch.Tensor,
    selected_counts: torch.Tensor,
    target_slots: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    impl, attn_layer_name, kv_cache, attn_metadata = _mla_runtime_state(layer_name)
    impl.cross_layer_graph_post(
        attn_layer_name,
        ql_nope,
        q_pe,
        topk_indices,
        kv_cache,
        attn_metadata,
        output,
    )


def sfa_forward_post_fake(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    topk_indices: torch.Tensor,
    selected_packed: torch.Tensor,
    selected_counts: torch.Tensor,
    target_slots: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="sfa_forward_pre",
    op_func=sfa_forward_pre,
    mutates_args=["output"],
    fake_impl=sfa_forward_pre_fake,
    dispatch_key="PrivateUse1",
)
direct_register_custom_op(
    op_name="sfa_lmcache_retrieve",
    op_func=sfa_lmcache_retrieve,
    mutates_args=["output"],
    fake_impl=sfa_lmcache_retrieve_fake,
    dispatch_key="PrivateUse1",
)
direct_register_custom_op(
    op_name="sfa_forward_post",
    op_func=sfa_forward_post,
    mutates_args=["output"],
    fake_impl=sfa_forward_post_fake,
    dispatch_key="PrivateUse1",
)
