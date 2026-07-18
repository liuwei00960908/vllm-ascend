import sys
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import torch
from vllm.config import CUDAGraphMode
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.forward_context import BatchDescriptor

from tests.ut.attention.utils import patch_distributed_groups
from tests.ut.base import TestBase
from vllm_ascend.ascend_forward_context import StagedSFALiveParityState
from vllm_ascend.attention.attention_v1 import AscendAttentionState

if 'torch_npu._inductor' not in sys.modules:
    sys.modules['torch_npu._inductor'] = MagicMock()

import vllm_ascend.attention.sfa_v1 as sfa_v1
from vllm_ascend.attention.sfa_v1 import AscendSFABackend, AscendSFAImpl, AscendSFAMetadata, AscendSFAMetadataBuilder
from vllm_ascend.utils import enable_dsa_cp


class TestLMCacheSparseWaitSync(TestBase):

    def setUp(self):
        self.original_once_done = sfa_v1._lmcache_sparse_wait_sync_once_done
        sfa_v1._lmcache_sparse_wait_sync_once_done = False

    def tearDown(self):
        sfa_v1._lmcache_sparse_wait_sync_once_done = self.original_once_done

    def test_once_mode_synchronizes_only_first_sparse_wait(self):
        stream = MagicMock()
        with (
            patch.object(sfa_v1, "_LMCACHE_SPARSE_WAIT_SYNC_ONCE", True),
            patch.object(
                sfa_v1.torch.npu,
                "current_stream",
                return_value=stream,
            ) as current_stream,
        ):
            sfa_v1._sync_compute_stream_after_lmcache_sparse_wait()
            sfa_v1._sync_compute_stream_after_lmcache_sparse_wait()

        current_stream.assert_called_once_with()
        stream.synchronize.assert_called_once_with()
        self.assertTrue(sfa_v1._lmcache_sparse_wait_sync_once_done)

    def test_completed_mode_does_not_touch_npu_stream(self):
        sfa_v1._lmcache_sparse_wait_sync_once_done = True
        with patch.object(sfa_v1.torch.npu, "current_stream") as current_stream:
            sfa_v1._sync_compute_stream_after_lmcache_sparse_wait()

        current_stream.assert_not_called()

    def test_disabled_mode_does_not_synchronize(self):
        with (
            patch.object(sfa_v1, "_LMCACHE_SPARSE_WAIT_SYNC_ONCE", False),
            patch.object(sfa_v1.torch.npu, "current_stream") as current_stream,
        ):
            sfa_v1._sync_compute_stream_after_lmcache_sparse_wait()

        current_stream.assert_not_called()
        self.assertFalse(sfa_v1._lmcache_sparse_wait_sync_once_done)

    def test_sync_compute_stream_skips_when_npu_unavailable(self):
        with (
            patch.object(sfa_v1, "_LMCACHE_SPARSE_WAIT_SYNC_ONCE", True),
            patch.object(sfa_v1.torch, "npu", None),
        ):
            sfa_v1._sync_compute_stream_after_lmcache_sparse_wait()

        self.assertFalse(sfa_v1._lmcache_sparse_wait_sync_once_done)


class TestStagedSFAGraphPoc(TestBase):

    @staticmethod
    def _make_eligible_impl():
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        impl.dsa_shrink_latent = 2
        impl.num_kv_heads = 1
        impl.kv_lora_rank = 2
        impl.qk_rope_head_dim = 2
        impl.head_dim = 2
        impl.index_topk = 4
        impl.enable_mlapo = False
        impl.enable_dsa_cp = False
        impl.enable_dsa_cp_with_o_proj_tp = False
        impl.use_sparse_c8_indexer = False
        impl.dsa_offload_free_paged = False
        impl.q_lora_rank = 4
        impl.fused_qkv_a_proj = MagicMock()
        impl.q_a_layernorm = MagicMock()
        impl.vllm_config = MagicMock()
        impl.vllm_config.cache_config.block_size = 128
        impl.vllm_config.speculative_config = None
        impl.vllm_config.lora_config = None
        impl._staged_sfa_capture_phases = set()
        impl._staged_sfa_replay_proved = set()
        impl._staged_sfa_live_capture_validated = False
        impl._staged_sfa_live_validated_request_ids = None
        impl._staged_sfa_dummy_cache_initialized = False
        impl._staged_sfa_parity_output = None
        impl._staged_sfa_parity_latent_scratch = None
        return impl

    @staticmethod
    def _make_eligible_kv_cache(
        *,
        dtype=torch.bfloat16,
        device="cpu",
        block_size=128,
        num_blocks=2,
    ):
        return (
            torch.empty(
                num_blocks, block_size, 1, 2, dtype=dtype, device=device
            ),
            torch.empty(
                num_blocks, block_size, 1, 2, dtype=dtype, device=device
            ),
            torch.empty(
                num_blocks, block_size, 1, 2, dtype=dtype, device=device
            ),
        )

    @staticmethod
    def _make_decode_metadata():
        metadata = MagicMock()
        metadata.attn_state = AscendAttentionState.DecodeOnly
        metadata.num_input_tokens = 1
        metadata.num_actual_tokens = 1
        metadata.num_decode_tokens = 1
        metadata.cos = torch.ones(1, 2)
        metadata.sin = torch.zeros(1, 2)
        metadata.slot_mapping = torch.tensor([0])
        metadata.indexer_slot_mapping = torch.tensor([0])
        metadata.cum_query_lens = torch.tensor([1])
        metadata.seq_lens = torch.tensor([9])
        metadata.seq_lens_cpu = torch.tensor([9])
        metadata.block_table = torch.tensor([[0]])
        metadata.indexer_block_table = torch.tensor([[0]])
        metadata.prompt_lens = torch.tensor([8], dtype=torch.int32)
        metadata.prompt_lens_cpu_rows = [8]
        metadata.decode_req_indices = torch.tensor([0], dtype=torch.int32)
        metadata.decode_req_indices_cpu = [0]
        metadata.need_sparse_lmcache_payload = True
        metadata.decode_valid_rows_all = True
        metadata.decode_valid_row_indices = None
        metadata.decode_scratch_base = None
        metadata.decode_scratch_base_compact = None
        metadata.decode_target_slot_mapping = None
        metadata.decode_request_ids_compact = ["req-0"]
        metadata.decode_remap_boundary = torch.empty(1, dtype=torch.int32)
        metadata.decode_remap_boundary_ready = False
        return metadata

    def test_eager_pre_reference_has_no_cache_or_connector_side_effects(
        self,
    ):
        impl = self._make_eligible_impl()
        impl.kv_lora_rank = 2
        impl.qk_rope_head_dim = 2
        impl.num_kv_heads = 1
        hidden_states = torch.arange(
            4, dtype=torch.bfloat16
        ).view(1, 4)
        qkv_lora = torch.arange(
            8, dtype=torch.bfloat16
        ).view(1, 8)
        impl.fused_qkv_a_proj = MagicMock(return_value=(qkv_lora,))
        impl.q_a_layernorm = MagicMock(side_effect=lambda value: value)
        impl._q_proj_and_k_up_proj = MagicMock(
            side_effect=lambda value: (value[:, :2], value[:, 2:])
        )
        impl.rope_single = MagicMock(side_effect=lambda value, *_: value)
        impl.kv_a_layernorm = MagicMock()
        impl.kv_a_layernorm.weight = torch.ones(
            2, dtype=torch.bfloat16
        )
        impl.kv_a_layernorm.variance_epsilon = 1e-6
        expected_index = torch.tensor(
            [[[5.0, 6.0]]],
            dtype=torch.bfloat16,
        )
        impl.indexer_select_pre_process = MagicMock(
            return_value=(expected_index, None)
        )
        impl._get_full_kv = MagicMock(side_effect=lambda value, _: value)
        topk_indices = torch.tensor(
            [[[1, 2, 3, 4]]],
            dtype=torch.int32,
        )
        selected_packed = torch.tensor(
            [[1, 2, 3, 4]],
            dtype=torch.int32,
        )
        impl.indexer_select_post_process = MagicMock(
            return_value=topk_indices
        )
        impl.exec_kv = MagicMock()
        expected_nope = torch.tensor(
            [[[[1.0, 2.0]]]],
            dtype=torch.bfloat16,
        )
        expected_pe = torch.tensor(
            [[[[3.0, 4.0]]]],
            dtype=torch.bfloat16,
        )
        kv_cache = self._make_eligible_kv_cache()
        for cache in kv_cache:
            cache.zero_()
        live_cache_before = tuple(cache.clone() for cache in kv_cache)
        cos = torch.ones(
            1, 1, 1, 2, dtype=torch.bfloat16
        )
        sin = torch.zeros_like(cos)
        latent_slot = torch.tensor([129], dtype=torch.int32)
        indexer_slot = torch.tensor([130], dtype=torch.int32)

        with (
            patch.object(
                sfa_v1,
                "scratch_remap",
                return_value=(topk_indices, selected_packed),
            ) as remap,
            patch.object(
                sfa_v1.torch_npu,
                "npu_scatter_nd_update_",
            ) as scatter,
            patch.object(
                sfa_v1,
                "wait_for_kv_layer_from_connector",
            ) as wait_for_layer,
            patch.object(
                sfa_v1.torch_npu,
                "npu_kv_rmsnorm_rope_cache",
                return_value=(None, None, expected_pe, expected_nope),
            ) as latent_reference,
        ):
            first_result = impl._staged_sfa_eager_pre_reference_poc(
                hidden_states,
                kv_cache,
                cos,
                sin,
                latent_slot,
                indexer_slot,
                torch.tensor([1]),
                torch.tensor([9]),
                torch.tensor([[0]]),
                torch.tensor([8], dtype=torch.int32),
            )
            first_scratch = impl._staged_sfa_parity_latent_scratch
            second_result = impl._staged_sfa_eager_pre_reference_poc(
                hidden_states,
                kv_cache,
                cos,
                sin,
                latent_slot,
                indexer_slot,
                torch.tensor([1]),
                torch.tensor([9]),
                torch.tensor([[0]]),
                torch.tensor([8], dtype=torch.int32),
            )

        for result in (first_result, second_result):
            self.assertIs(result[2], topk_indices)
            self.assertIs(result[3], selected_packed)
            self.assertTrue(
                torch.equal(result[4], expected_nope.view(1, 2))
            )
            self.assertTrue(
                torch.equal(result[5], expected_pe.view(1, 2))
            )
            self.assertTrue(
                torch.equal(result[6], expected_index.view(1, 2))
            )
        for live, before in zip(kv_cache, live_cache_before):
            self.assertTrue(torch.equal(live, before))
        self.assertEqual(latent_reference.call_count, 2)
        impl.exec_kv.assert_not_called()
        scatter.assert_not_called()
        wait_for_layer.assert_not_called()
        self.assertEqual(remap.call_count, 2)
        second_scratch = impl._staged_sfa_parity_latent_scratch
        self.assertIsNotNone(first_scratch)
        self.assertIs(first_scratch[0], second_scratch[0])
        self.assertIs(first_scratch[1], second_scratch[1])
        self.assertNotEqual(
            first_scratch[0].data_ptr(),
            kv_cache[0].data_ptr(),
        )
        self.assertNotEqual(
            first_scratch[1].data_ptr(),
            kv_cache[1].data_ptr(),
        )

        for operator_call in latent_reference.call_args_list:
            args = operator_call.args
            kwargs = operator_call.kwargs
            self.assertEqual(
                tuple(args[0].shape),
                (1, 1, 1, 4),
            )
            self.assertEqual(args[0].dtype, torch.bfloat16)
            self.assertIs(args[1], impl.kv_a_layernorm.weight)
            self.assertIs(args[2], cos)
            self.assertIs(args[3], sin)
            self.assertEqual(args[4].dtype, torch.int64)
            self.assertEqual(args[4].tolist(), [0])
            self.assertIs(args[5], first_scratch[1])
            self.assertIs(args[6], first_scratch[0])
            self.assertEqual(
                tuple(args[5].shape),
                tuple(kv_cache[1][:1].shape),
            )
            self.assertEqual(
                tuple(args[6].shape),
                tuple(kv_cache[0][:1].shape),
            )
            self.assertEqual(args[5].dtype, kv_cache[1].dtype)
            self.assertEqual(args[6].dtype, kv_cache[0].dtype)
            self.assertEqual(args[5].device, kv_cache[1].device)
            self.assertEqual(args[6].device, kv_cache[0].device)
            self.assertEqual(
                kwargs["epsilon"],
                impl.kv_a_layernorm.variance_epsilon,
            )
            self.assertEqual(kwargs["cache_mode"], "PA")
            self.assertTrue(kwargs["is_output_kv"])

    def test_cache_write_parity_detects_wrong_physical_slot(self):
        impl = self._make_eligible_impl()
        cache = torch.tensor(
            [
                [[[1.0, 2.0]]],
                [[[3.0, 4.0]]],
            ]
        )
        wrong_slot = torch.tensor([1], dtype=torch.int64)
        actual_row = cache.reshape(-1, cache.shape[-1]).index_select(
            0,
            wrong_slot,
        )
        expected_current_value = torch.tensor([[1.0, 2.0]])

        flags = impl._staged_sfa_parity_flags(
            (
                (
                    "cache_nope",
                    actual_row,
                    expected_current_value,
                    False,
                ),
            )
        )

        self.assertEqual(flags[0][0], "cache_nope")
        self.assertFalse(bool(flags[0][1].item()))

    def test_save_submission_is_held_for_any_live_parity_candidate(
        self,
    ):
        impl = self._make_eligible_impl()
        parity_state = StagedSFALiveParityState(
            request_id="req-0",
            seq_len=9,
            expected_layers=1,
        )
        forward_context = MagicMock()
        forward_context.staged_sfa_live_parity_state = parity_state
        latent_cache = torch.tensor([1.0])
        index_cache = torch.tensor([2.0])
        operations = [
            ("layer-0", [latent_cache]),
            ("layer-0.index", [index_cache]),
        ]

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=forward_context,
            ),
            patch.object(
                sfa_v1,
                "maybe_save_kv_layer_to_connector",
            ) as save_layer,
        ):
            impl._submit_sfa_save_operations(operations)

        save_layer.assert_not_called()
        self.assertEqual(
            parity_state.pending_saves,
            operations,
        )

    def test_remap_boundary_is_resolved_once_per_step(self):
        metadata = self._make_decode_metadata()
        metadata.prompt_lens_cpu_rows = [1000]
        metadata.seq_lens_cpu = torch.tensor([1025])
        original_address = metadata.decode_remap_boundary.data_ptr()

        with (
            patch.object(
                sfa_v1,
                "get_lmcache_sparse_cached_tokens",
                return_value=[900],
            ) as get_cached_tokens,
            patch.object(
                sfa_v1,
                "_decode_window_save_window_size",
                return_value=256,
            ),
        ):
            first = sfa_v1._prepare_staged_sfa_remap_boundary(
                metadata,
                ["req-0"],
                is_dummy_run=False,
            )
            second = sfa_v1._prepare_staged_sfa_remap_boundary(
                metadata,
                ["req-0"],
                is_dummy_run=False,
            )

        self.assertIs(first, second)
        self.assertEqual(first.data_ptr(), original_address)
        self.assertEqual(first.tolist(), [900])
        get_cached_tokens.assert_called_once_with(["req-0"])

    def test_piecewise_dummy_requires_eager_cache_initialization(self):
        impl = self._make_eligible_impl()
        impl._get_staged_sfa_graph_wrappers = MagicMock(
            return_value=(MagicMock(), MagicMock())
        )
        forward_context = MagicMock()
        forward_context.staged_sfa_graph_dummy_run = True
        forward_context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=forward_context,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "eager dummy warmup",
            ),
        ):
            impl._forward_staged_sfa_graph_poc(
                layer_name="model.layers.0.self_attn.attn",
                index_layer_name=None,
                index_lmcache_enabled=False,
                hidden_states=torch.empty(1, 4),
                kv_cache=(
                    torch.empty(1),
                    torch.empty(1),
                    torch.empty(1),
                ),
                attn_metadata=self._make_decode_metadata(),
                output=torch.empty(1, 4),
            )

    def test_eligibility_accepts_single_native_piecewise_decode(self):
        impl = self._make_eligible_impl()
        metadata = self._make_decode_metadata()
        forward_context = MagicMock()
        forward_context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        forward_context.capturing = False
        forward_context.staged_sfa_graph_dummy_run = False
        forward_context.batch_descriptor = BatchDescriptor(
            num_tokens=1,
            num_reqs=None,
            uniform=False,
        )
        forward_context.dsa_offload_manager = None
        forward_context.dsa_adapter_cache = None

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=forward_context,
            ),
            patch.object(
                sfa_v1,
                "get_weight_prefetch_method",
                return_value=None,
            ),
            patch.object(
                sfa_v1.envs,
                "VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY",
                False,
            ),
        ):
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(dtype=dtype):
                    reason = impl._staged_sfa_graph_ineligible_reason(
                        torch.empty(1, 4, dtype=dtype),
                        self._make_eligible_kv_cache(dtype=dtype),
                        metadata,
                    )
                    self.assertIsNone(reason)

    def test_eligibility_rejects_invalid_cache_contract(self):
        impl = self._make_eligible_impl()
        metadata = self._make_decode_metadata()
        forward_context = MagicMock()
        forward_context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        forward_context.capturing = False
        forward_context.staged_sfa_graph_dummy_run = False
        forward_context.batch_descriptor = BatchDescriptor(
            num_tokens=1,
            num_reqs=None,
            uniform=False,
        )
        forward_context.dsa_offload_manager = None
        forward_context.dsa_adapter_cache = None
        valid = self._make_eligible_kv_cache()
        invalid_contracts = (
            (
                "rank",
                (
                    torch.empty(2, 128, 2, dtype=torch.bfloat16),
                    valid[1],
                    valid[2],
                ),
                "rank-4 PA_BSND",
            ),
            (
                "head_axis",
                (
                    torch.empty(
                        2, 128, 2, 2, dtype=torch.bfloat16
                    ),
                    valid[1],
                    valid[2],
                ),
                "one KV head",
            ),
            (
                "hidden_dim",
                (
                    torch.empty(
                        2, 128, 1, 3, dtype=torch.bfloat16
                    ),
                    valid[1],
                    valid[2],
                ),
                "hidden dimensions",
            ),
            (
                "different_block_sizes",
                (
                    valid[0],
                    valid[1],
                    torch.empty(
                        2, 64, 1, 2, dtype=torch.bfloat16
                    ),
                ),
                "block sizes do not agree",
            ),
            (
                "wrong_configured_block_size",
                self._make_eligible_kv_cache(block_size=64),
                "configured block size",
            ),
            (
                "different_devices",
                (
                    valid[0],
                    valid[1],
                    torch.empty(
                        2,
                        128,
                        1,
                        2,
                        dtype=torch.bfloat16,
                        device="meta",
                    ),
                ),
                "different devices",
            ),
            (
                "different_dtypes",
                (
                    valid[0],
                    valid[1],
                    torch.empty(
                        2, 128, 1, 2, dtype=torch.float16
                    ),
                ),
                "share one dtype",
            ),
            (
                "unsupported_dtype",
                self._make_eligible_kv_cache(dtype=torch.float32),
                "must be float16 or bfloat16",
            ),
        )

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=forward_context,
            ),
            patch.object(
                sfa_v1,
                "get_weight_prefetch_method",
                return_value=None,
            ),
            patch.object(
                sfa_v1.envs,
                "VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY",
                False,
            ),
        ):
            for name, kv_cache, expected_reason in invalid_contracts:
                with self.subTest(name=name):
                    reason = impl._staged_sfa_graph_ineligible_reason(
                        torch.empty(1, 4, dtype=torch.bfloat16),
                        kv_cache,
                        metadata,
                    )
                    self.assertIn(expected_reason, reason)

    def test_eligibility_rejects_weight_prefetch(self):
        impl = self._make_eligible_impl()
        metadata = self._make_decode_metadata()
        forward_context = MagicMock()
        forward_context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        forward_context.staged_sfa_graph_dummy_run = False
        forward_context.batch_descriptor = BatchDescriptor(
            num_tokens=1,
            num_reqs=None,
            uniform=False,
        )
        forward_context.dsa_offload_manager = None
        forward_context.dsa_adapter_cache = None
        weight_prefetch_method = MagicMock()
        weight_prefetch_method.mla_sfa_prefetch_enable = True

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=forward_context,
            ),
            patch.object(
                sfa_v1,
                "get_weight_prefetch_method",
                return_value=weight_prefetch_method,
            ),
            patch.object(
                sfa_v1.envs,
                "VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY",
                False,
            ),
        ):
            reason = impl._staged_sfa_graph_ineligible_reason(
                torch.empty(1, 4),
                self._make_eligible_kv_cache(),
                metadata,
            )

        self.assertEqual(reason, "weight prefetch is enabled")

    def test_eligibility_rejects_padded_or_batched_decode(self):
        impl = self._make_eligible_impl()
        metadata = self._make_decode_metadata()
        metadata.num_input_tokens = 2
        forward_context = MagicMock()
        forward_context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        forward_context.capturing = False
        forward_context.staged_sfa_graph_dummy_run = False
        forward_context.batch_descriptor = BatchDescriptor(
            num_tokens=1,
            num_reqs=None,
            uniform=False,
        )
        forward_context.dsa_offload_manager = None
        forward_context.dsa_adapter_cache = None

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=forward_context,
            ),
            patch.object(
                sfa_v1.envs,
                "VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY",
                False,
            ),
        ):
            reason = impl._staged_sfa_graph_ineligible_reason(
                torch.empty(2, 4),
                self._make_eligible_kv_cache(),
                metadata,
            )

        self.assertEqual(
            reason,
            "only a single, unpadded decode token is supported",
        )

    def test_eligibility_accepts_explicit_eager_dummy_warmup(self):
        impl = self._make_eligible_impl()
        metadata = self._make_decode_metadata()
        forward_context = MagicMock()
        forward_context.cudagraph_runtime_mode = CUDAGraphMode.NONE
        forward_context.capturing = False
        forward_context.staged_sfa_graph_dummy_run = True
        forward_context.batch_descriptor = BatchDescriptor(
            num_tokens=1,
            num_reqs=None,
            uniform=False,
        )
        forward_context.dsa_offload_manager = None
        forward_context.dsa_adapter_cache = None

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=forward_context,
            ),
            patch.object(
                sfa_v1,
                "get_weight_prefetch_method",
                return_value=None,
            ),
            patch.object(
                sfa_v1.envs,
                "VLLM_ASCEND_DSA_OFFLOAD_ASSERT_PARITY",
                False,
            ),
        ):
            reason = impl._staged_sfa_graph_ineligible_reason(
                torch.empty(1, 4),
                self._make_eligible_kv_cache(),
                metadata,
            )

        self.assertIsNone(reason)

    def test_wrapper_factory_creates_two_piecewise_graphs(self):
        impl = self._make_eligible_impl()
        impl.vllm_config = MagicMock()
        impl._staged_sfa_pre_graph = None
        impl._staged_sfa_post_graph = None
        pre_wrapper = MagicMock(name="pre_wrapper")
        post_wrapper = MagicMock(name="post_wrapper")

        with (
            patch(
                "vllm_ascend.compilation.acl_graph.ACLGraphWrapper",
                side_effect=[pre_wrapper, post_wrapper],
            ) as wrapper_cls,
            patch.object(sfa_v1.logger, "warning_once"),
        ):
            result = impl._get_staged_sfa_graph_wrappers()

        self.assertEqual(result, (pre_wrapper, post_wrapper))
        self.assertEqual(wrapper_cls.call_count, 2)
        for call in wrapper_cls.call_args_list:
            self.assertEqual(
                call.kwargs["runtime_mode"],
                CUDAGraphMode.PIECEWISE,
            )
            self.assertFalse(
                call.kwargs["cudagraph_options"].weak_ref_output
            )
            self.assertFalse(call.kwargs["synchronize_before_replay"])

    def test_live_validation_accepts_matching_graph_input_addresses(self):
        impl = self._make_eligible_impl()
        graph_inputs = (torch.empty(1), torch.empty(1))
        batch_descriptor = BatchDescriptor(
            num_tokens=1,
            num_reqs=None,
            uniform=False,
        )
        forward_context = MagicMock()
        forward_context.batch_descriptor = batch_descriptor
        pre_entry = MagicMock()
        pre_entry.aclgraph = object()
        pre_entry.input_addresses = [
            tensor.data_ptr() for tensor in graph_inputs
        ]
        pre_wrapper = MagicMock()
        pre_wrapper.concrete_aclgraph_entries = {
            batch_descriptor: pre_entry,
        }

        with patch.object(
            sfa_v1,
            "get_forward_context",
            return_value=forward_context,
        ):
            result = impl._validate_staged_sfa_graph_entry(
                "pre",
                pre_wrapper,
                graph_inputs,
            )

        self.assertIs(result, pre_entry)

    def test_live_validation_rejects_a_missing_post_graph(self):
        impl = self._make_eligible_impl()
        batch_descriptor = BatchDescriptor(
            num_tokens=1,
            num_reqs=None,
            uniform=False,
        )
        forward_context = MagicMock()
        forward_context.batch_descriptor = batch_descriptor
        post_wrapper = MagicMock()
        post_wrapper.concrete_aclgraph_entries = {}

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=forward_context,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "post region was not captured",
            ),
        ):
            impl._validate_staged_sfa_graph_entry(
                "post",
                post_wrapper,
            )

    def test_live_validation_rejects_changed_graph_input_addresses(self):
        impl = self._make_eligible_impl()
        batch_descriptor = BatchDescriptor(num_tokens=1)
        forward_context = MagicMock()
        forward_context.batch_descriptor = batch_descriptor
        graph_entry = MagicMock()
        graph_entry.aclgraph = object()
        graph_entry.input_addresses = [1, 2]
        graph_wrapper = MagicMock()
        graph_wrapper.concrete_aclgraph_entries = {
            batch_descriptor: graph_entry,
        }

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=forward_context,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "differing input indices",
            ),
        ):
            impl._validate_staged_sfa_graph_entry(
                "pre",
                graph_wrapper,
                (torch.empty(1), torch.empty(1)),
            )

    def test_live_validation_requires_recorded_graph_input_addresses(self):
        impl = self._make_eligible_impl()
        batch_descriptor = BatchDescriptor(num_tokens=1)
        forward_context = MagicMock()
        forward_context.batch_descriptor = batch_descriptor
        graph_entry = MagicMock()
        graph_entry.aclgraph = object()
        graph_entry.input_addresses = None
        graph_wrapper = MagicMock()
        graph_wrapper.concrete_aclgraph_entries = {
            batch_descriptor: graph_entry,
        }

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=forward_context,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "did not record its input addresses",
            ),
        ):
            impl._validate_staged_sfa_graph_entry(
                "pre",
                graph_wrapper,
                (torch.empty(1),),
            )
    def test_live_pointer_validation_runs_after_first_validated_call(self):
        impl = self._make_eligible_impl()
        impl._staged_sfa_capture_phases = {
            "pre:enter",
            "pre:exit",
            "post:enter",
            "post:exit",
        }
        impl._staged_sfa_replay_proved = {"pre", "post"}
        impl._staged_sfa_live_capture_validated = True
        impl._staged_sfa_live_validated_request_ids = ("req-0",)
        metadata = self._make_decode_metadata()
        hidden_states = torch.empty(1, 4)
        output = torch.empty_like(hidden_states)
        kv_cache = (
            torch.empty(1),
            torch.empty(1),
            torch.empty(1),
        )
        pre_graph = MagicMock()
        post_graph = MagicMock()
        impl._get_staged_sfa_graph_wrappers = MagicMock(
            return_value=(pre_graph, post_graph)
        )
        forward_context = MagicMock()
        forward_context.staged_sfa_graph_dummy_run = False
        forward_context.staged_sfa_live_parity_state = None
        forward_context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        forward_context.batch_descriptor = BatchDescriptor(num_tokens=1)

        pointer_validator = MagicMock(
            side_effect=RuntimeError("pointer drift")
        )
        impl._validate_staged_sfa_graph_entry = pointer_validator

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=forward_context,
            ),
            patch.object(
                sfa_v1.torch.profiler,
                "record_function",
                side_effect=lambda *args, **kwargs: nullcontext(),
            ),
            patch.object(
                sfa_v1,
                "_decode_window_save_window_size",
                return_value=256,
            ),
            self.assertRaisesRegex(RuntimeError, "pointer drift"),
        ):
            impl._forward_staged_sfa_graph_poc(
                layer_name="model.layers.0.self_attn.attn",
                index_layer_name=None,
                index_lmcache_enabled=False,
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                attn_metadata=metadata,
                output=output,
            )

        pointer_validator.assert_called_once()
        self.assertEqual(pointer_validator.call_args.args[0], "pre")
        pre_graph.assert_not_called()
        post_graph.assert_not_called()


    def test_capture_phase_requires_active_npu_stream_capture(self):
        impl = self._make_eligible_impl()
        impl._staged_sfa_capture_phases.clear()
        forward_context = MagicMock()
        forward_context.staged_sfa_graph_dummy_run = True
        forward_context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=forward_context,
            ),
            patch.object(
                sfa_v1.torch.npu,
                "is_current_stream_capturing",
                return_value=False,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "outside NPU stream capture",
            ),
        ):
            impl._observe_staged_sfa_capture_phase("pre", "enter")

    def test_startup_replay_proof_restores_poisoned_output(self):
        impl = self._make_eligible_impl()
        impl._staged_sfa_capture_phases = {"pre:enter", "pre:exit"}
        graph_input = torch.tensor([3.0])
        graph_output = torch.tensor([1.0, 2.0])
        reference = graph_output.clone()
        batch_descriptor = BatchDescriptor(num_tokens=1)
        forward_context = MagicMock()
        forward_context.batch_descriptor = batch_descriptor
        graph_entry = MagicMock()
        graph_entry.aclgraph = object()
        graph_entry.input_addresses = [graph_input.data_ptr()]
        graph_wrapper = MagicMock(
            side_effect=lambda *args: graph_output.copy_(reference)
        )
        graph_wrapper.concrete_aclgraph_entries = {
            batch_descriptor: graph_entry,
        }
        stream = MagicMock()

        with (
            patch.object(
                impl,
                "_staged_sfa_capture_dummy_active",
                return_value=True,
            ),
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=forward_context,
            ),
            patch.object(
                sfa_v1.torch.npu,
                "is_current_stream_capturing",
                return_value=False,
            ),
            patch.object(
                sfa_v1.torch.npu,
                "current_stream",
                return_value=stream,
            ),
        ):
            impl._prove_staged_sfa_graph_replay(
                "pre",
                graph_wrapper,
                (graph_input,),
                (graph_output,),
            )

        self.assertEqual(impl._staged_sfa_replay_proved, {"pre"})
        self.assertTrue(torch.equal(graph_output, reference))
        graph_wrapper.assert_called_once_with(graph_input)
        stream.synchronize.assert_called_once_with()

    def test_startup_replay_proof_requires_both_capture_phases(self):
        impl = self._make_eligible_impl()
        impl._staged_sfa_capture_phases = {"pre:enter"}

        with (
            patch.object(
                impl,
                "_staged_sfa_capture_dummy_active",
                return_value=True,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "capture-phase proof.*incomplete",
            ),
        ):
            impl._prove_staged_sfa_graph_replay(
                "pre",
                MagicMock(),
                (torch.tensor([3.0]),),
                (torch.tensor([1.0]),),
            )

    def test_startup_replay_proof_rejects_output_left_poisoned(self):
        impl = self._make_eligible_impl()
        impl._staged_sfa_capture_phases = {"pre:enter", "pre:exit"}
        graph_input = torch.tensor([3.0])
        graph_output = torch.tensor([1.0, 2.0])
        batch_descriptor = BatchDescriptor(num_tokens=1)
        forward_context = MagicMock()
        forward_context.batch_descriptor = batch_descriptor
        graph_entry = MagicMock()
        graph_entry.aclgraph = object()
        graph_entry.input_addresses = [graph_input.data_ptr()]
        graph_wrapper = MagicMock()
        graph_wrapper.concrete_aclgraph_entries = {
            batch_descriptor: graph_entry,
        }
        stream = MagicMock()

        with (
            patch.object(
                impl,
                "_staged_sfa_capture_dummy_active",
                return_value=True,
            ),
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=forward_context,
            ),
            patch.object(
                sfa_v1.torch.npu,
                "is_current_stream_capturing",
                return_value=False,
            ),
            patch.object(
                sfa_v1.torch.npu,
                "current_stream",
                return_value=stream,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "replay did not reproduce captured output",
            ),
        ):
            impl._prove_staged_sfa_graph_replay(
                "pre",
                graph_wrapper,
                (graph_input,),
                (graph_output,),
            )

        self.assertEqual(impl._staged_sfa_replay_proved, set())
        self.assertTrue(torch.isnan(graph_output).all())
        graph_wrapper.assert_called_once_with(graph_input)
        stream.synchronize.assert_called_once_with()

    def test_startup_replay_proof_rejects_partially_restored_output(self):
        impl = self._make_eligible_impl()
        impl._staged_sfa_capture_phases = {"pre:enter", "pre:exit"}
        graph_input = torch.tensor([3.0])
        graph_output = torch.arange(32, dtype=torch.float32)
        reference = graph_output.clone()
        batch_descriptor = BatchDescriptor(num_tokens=1)
        forward_context = MagicMock()
        forward_context.batch_descriptor = batch_descriptor
        graph_entry = MagicMock()
        graph_entry.aclgraph = object()
        graph_entry.input_addresses = [graph_input.data_ptr()]

        def restore_only_former_probe(*args):
            graph_output[:16].copy_(reference[:16])

        graph_wrapper = MagicMock(
            side_effect=restore_only_former_probe
        )
        graph_wrapper.concrete_aclgraph_entries = {
            batch_descriptor: graph_entry,
        }
        stream = MagicMock()

        with (
            patch.object(
                impl,
                "_staged_sfa_capture_dummy_active",
                return_value=True,
            ),
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=forward_context,
            ),
            patch.object(
                sfa_v1.torch.npu,
                "is_current_stream_capturing",
                return_value=False,
            ),
            patch.object(
                sfa_v1.torch.npu,
                "current_stream",
                return_value=stream,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "replay did not reproduce captured output",
            ),
        ):
            impl._prove_staged_sfa_graph_replay(
                "pre",
                graph_wrapper,
                (graph_input,),
                (graph_output,),
            )

        self.assertEqual(impl._staged_sfa_replay_proved, set())
        self.assertTrue(torch.equal(graph_output[:16], reference[:16]))
        self.assertTrue(torch.isnan(graph_output[16:]).all())
        graph_wrapper.assert_called_once_with(graph_input)
        stream.synchronize.assert_called_once_with()

    def test_live_path_rejects_an_incomplete_startup_proof(self):
        impl = self._make_eligible_impl()

        with self.assertRaisesRegex(
            RuntimeError,
            "startup replay smoke test is incomplete",
        ):
            impl._require_staged_sfa_startup_proof()

    def test_retrieve_barrier_runs_between_pre_and_post_graphs(self):
        impl = self._make_eligible_impl()
        impl._staged_sfa_capture_phases = {
            "pre:enter",
            "pre:exit",
            "post:enter",
            "post:exit",
        }
        impl._staged_sfa_replay_proved = {"pre", "post"}
        impl.is_kv_producer = True
        impl.dsa_offload_unbundle = True
        metadata = self._make_decode_metadata()
        hidden_states = torch.empty(1, 4)
        output = torch.zeros_like(hidden_states)
        kv_cache = (
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
        )
        ql_nope = torch.tensor([[1.0, 2.0]])
        q_pe = torch.tensor([[3.0, 4.0]])
        topk_indices = torch.tensor([[[1, 2, 3, 4]]], dtype=torch.int32)
        selected_packed = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
        order = []
        reshape_event = MagicMock()
        reshape_event.record.side_effect = lambda: order.append("event")

        def pre_graph(*args):
            order.append("pre")
            return ql_nope, q_pe, topk_indices, selected_packed

        def post_graph(*args):
            order.append("post")
            return output

        batch_descriptor = BatchDescriptor(num_tokens=1)
        pre_entry = MagicMock()
        pre_entry.aclgraph = object()
        pre_inputs = (
            hidden_states,
            *kv_cache,
            metadata.cos,
            metadata.sin,
            metadata.slot_mapping,
            metadata.indexer_slot_mapping,
            metadata.cum_query_lens,
            metadata.seq_lens,
            metadata.indexer_block_table,
            metadata.decode_remap_boundary,
        )
        pre_entry.input_addresses = [
            tensor.data_ptr() for tensor in pre_inputs
        ]
        post_entry = MagicMock()
        post_entry.aclgraph = object()
        post_inputs = (
            ql_nope,
            q_pe,
            topk_indices,
            kv_cache[0],
            kv_cache[1],
            metadata.cum_query_lens,
            metadata.seq_lens,
            metadata.block_table,
            output,
        )
        post_entry.input_addresses = [
            tensor.data_ptr() for tensor in post_inputs
        ]
        pre_graph.concrete_aclgraph_entries = {
            batch_descriptor: pre_entry,
        }
        post_graph.concrete_aclgraph_entries = {
            batch_descriptor: post_entry,
        }
        impl._get_staged_sfa_graph_wrappers = MagicMock(
            return_value=(pre_graph, post_graph)
        )
        forward_context = MagicMock()
        forward_context.capturing = False
        forward_context.staged_sfa_graph_dummy_run = False
        forward_context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        forward_context.batch_descriptor = batch_descriptor
        parity_state = StagedSFALiveParityState(
            request_id="req-0",
            seq_len=9,
            expected_layers=1,
        )
        forward_context.staged_sfa_live_parity_state = parity_state

        def wait_for_layer_side_effect(layer_name, *args, **kwargs):
            if kwargs.get("selected_tokens") is None:
                order.append("index-retrieve")
            else:
                order.append("latent-retrieve")

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=forward_context,
            ),
            patch.object(
                sfa_v1,
                "wait_for_kv_layer_from_connector",
                side_effect=wait_for_layer_side_effect,
            ) as wait_for_layer,
            patch.object(
                sfa_v1,
                "maybe_save_kv_layer_to_connector",
                side_effect=lambda *args, **kwargs: order.append("save"),
            ) as save_layer,
            patch.object(
                sfa_v1.torch.npu,
                "Event",
                return_value=reshape_event,
            ),
            patch.object(
                sfa_v1.torch.profiler,
                "record_function",
                side_effect=lambda *args, **kwargs: nullcontext(),
            ),
            patch.object(
                impl,
                "_staged_sfa_eager_pre_reference_poc",
                return_value=(
                    ql_nope.clone(),
                    q_pe.clone(),
                    topk_indices.clone(),
                    selected_packed.clone(),
                    kv_cache[0].view(1, 1).clone(),
                    kv_cache[1].view(1, 1).clone(),
                    kv_cache[2].view(1, 1).clone(),
                ),
            ) as eager_pre_reference,
            patch.object(
                impl,
                "_staged_sfa_eager_post_reference_poc",
                return_value=output.clone(),
            ) as eager_post_reference,
            patch.object(sfa_v1._dsa_prof, "step"),
            patch.object(sfa_v1.logger, "info_once") as info_once,
            patch.object(
                sfa_v1,
                "get_lmcache_sparse_cached_tokens",
                return_value=[8],
            ),
            patch.object(
                sfa_v1,
                "_decode_window_save_window_size",
                return_value=256,
            ),
            patch.object(
                sfa_v1,
                "_LMCACHE_SPARSE_WAIT_SYNC_ONCE",
                False,
            ),
        ):
            result = impl._forward_staged_sfa_graph_poc(
                layer_name="model.layers.0.self_attn.attn",
                index_layer_name="model.layers.0.self_attn.indexer.k_cache",
                index_lmcache_enabled=True,
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                attn_metadata=metadata,
                output=output,
            )

        self.assertIs(result, output)
        self.assertTrue(impl._staged_sfa_live_capture_validated)
        self.assertEqual(
            impl._staged_sfa_live_validated_request_ids, ("req-0",)
        )
        info_once.assert_called_once()
        self.assertEqual(
            order,
            [
                "index-retrieve",
                "pre",
                "event",
                "latent-retrieve",
                "post",
            ],
        )
        self.assertEqual(wait_for_layer.call_count, 2)
        self.assertEqual(
            wait_for_layer.call_args_list[0].args,
            ("model.layers.0.self_attn.indexer.k_cache",),
        )
        latent_call = wait_for_layer.call_args_list[1]
        self.assertEqual(
            latent_call.args,
            ("model.layers.0.self_attn.attn",),
        )
        self.assertIs(latent_call.kwargs["selected_tokens"], selected_packed)
        self.assertIsNone(latent_call.kwargs["token_start_index"])
        self.assertEqual(latent_call.kwargs["request_ids"], ["req-0"])
        self.assertIsNone(latent_call.kwargs["target_slot_mapping"])
        self.assertIs(metadata.reshape_cache_event, reshape_event)
        eager_pre_reference.assert_called_once()
        eager_post_reference.assert_called_once()
        self.assertEqual(parity_state.checked_impl_ids, {id(impl)})
        self.assertEqual(
            parity_state.checked_layer_names,
            ["model.layers.0.self_attn.attn"],
        )
        self.assertEqual(len(parity_state.match_flags), 8)
        self.assertEqual(parity_state.failures, [])
        save_layer.assert_not_called()
        self.assertEqual(len(parity_state.pending_saves), 2)
        latent_save_name, latent_save_tensors = (
            parity_state.pending_saves[0]
        )
        self.assertEqual(
            latent_save_name,
            "model.layers.0.self_attn.attn",
        )
        self.assertIs(latent_save_tensors[0], kv_cache[0])
        self.assertIs(latent_save_tensors[1], kv_cache[1])
        index_save_name, index_save_tensors = (
            parity_state.pending_saves[1]
        )
        self.assertEqual(
            index_save_name,
            "model.layers.0.self_attn.indexer.k_cache",
        )
        self.assertIs(index_save_tensors[0], kv_cache[2])

    def test_piecewise_dummy_proves_both_graphs_without_advancing_lmcache(
        self,
    ):
        impl = self._make_eligible_impl()
        impl._staged_sfa_dummy_cache_initialized = True
        impl.is_kv_producer = True
        impl.dsa_offload_unbundle = True
        metadata = self._make_decode_metadata()
        hidden_states = torch.ones(1, 4)
        output = torch.full_like(hidden_states, 7.0)
        kv_cache = (
            torch.empty(1),
            torch.empty(1),
            torch.empty(1),
        )
        ql_nope = torch.tensor([[1.0, 2.0]])
        q_pe = torch.tensor([[3.0, 4.0]])
        topk_indices = torch.tensor([[[1, 2, 3, 4]]], dtype=torch.int32)
        selected_packed = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
        pre_reference = tuple(
            tensor.clone()
            for tensor in (
                ql_nope,
                q_pe,
                topk_indices,
                selected_packed,
            )
        )
        post_reference = output.clone()
        batch_descriptor = BatchDescriptor(num_tokens=1)
        forward_context = MagicMock()
        forward_context.capturing = True
        forward_context.staged_sfa_graph_dummy_run = True
        forward_context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        forward_context.batch_descriptor = batch_descriptor
        capture_state = {"active": False}
        call_count = {"pre": 0, "post": 0}
        order = []

        def pre_graph(*args):
            if call_count["pre"] == 0:
                capture_state["active"] = True
                try:
                    impl._observe_staged_sfa_capture_phase("pre", "enter")
                    impl._observe_staged_sfa_capture_phase("pre", "exit")
                finally:
                    capture_state["active"] = False
                order.append("pre-capture")
            else:
                for graph_output, reference in zip(
                    (ql_nope, q_pe, topk_indices, selected_packed),
                    pre_reference,
                ):
                    graph_output.copy_(reference)
                order.append("pre-replay")
            call_count["pre"] += 1
            return ql_nope, q_pe, topk_indices, selected_packed

        def post_graph(*args):
            if call_count["post"] == 0:
                capture_state["active"] = True
                try:
                    impl._observe_staged_sfa_capture_phase("post", "enter")
                    output.copy_(post_reference)
                    impl._observe_staged_sfa_capture_phase("post", "exit")
                finally:
                    capture_state["active"] = False
                order.append("post-capture")
            else:
                output.copy_(post_reference)
                order.append("post-replay")
            call_count["post"] += 1
            return output

        pre_entry = MagicMock()
        pre_entry.aclgraph = object()
        pre_inputs = (
            hidden_states,
            *kv_cache,
            metadata.cos,
            metadata.sin,
            metadata.slot_mapping,
            metadata.indexer_slot_mapping,
            metadata.cum_query_lens,
            metadata.seq_lens,
            metadata.indexer_block_table,
            metadata.decode_remap_boundary,
        )
        pre_entry.input_addresses = [
            tensor.data_ptr() for tensor in pre_inputs
        ]
        pre_graph.concrete_aclgraph_entries = {
            batch_descriptor: pre_entry,
        }
        post_inputs = (
            ql_nope,
            q_pe,
            topk_indices,
            kv_cache[0],
            kv_cache[1],
            metadata.cum_query_lens,
            metadata.seq_lens,
            metadata.block_table,
            output,
        )
        post_entry = MagicMock()
        post_entry.aclgraph = object()
        post_entry.input_addresses = [
            tensor.data_ptr() for tensor in post_inputs
        ]
        post_graph.concrete_aclgraph_entries = {
            batch_descriptor: post_entry,
        }
        impl._get_staged_sfa_graph_wrappers = MagicMock(
            return_value=(pre_graph, post_graph)
        )
        stream = MagicMock()

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=forward_context,
            ),
            patch.object(
                sfa_v1.torch.npu,
                "is_current_stream_capturing",
                side_effect=lambda: capture_state["active"],
            ),
            patch.object(
                sfa_v1.torch.npu,
                "current_stream",
                return_value=stream,
            ),
            patch.object(
                sfa_v1,
                "wait_for_kv_layer_from_connector",
            ) as wait_for_layer,
            patch.object(
                sfa_v1,
                "maybe_save_kv_layer_to_connector",
            ) as save_layer,
            patch.object(
                sfa_v1.torch.profiler,
                "record_function",
                side_effect=lambda *args, **kwargs: nullcontext(),
            ),
            patch.object(sfa_v1._dsa_prof, "step") as profile_step,
            patch.object(sfa_v1.logger, "info_once") as info_once,
        ):
            result = impl._forward_staged_sfa_graph_poc(
                layer_name="model.layers.0.self_attn.attn",
                index_layer_name="model.layers.0.self_attn.indexer.k_cache",
                index_lmcache_enabled=True,
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                attn_metadata=metadata,
                output=output,
            )

        self.assertIs(result, output)
        self.assertEqual(
            impl._staged_sfa_capture_phases,
            {"pre:enter", "pre:exit", "post:enter", "post:exit"},
        )
        self.assertEqual(impl._staged_sfa_replay_proved, {"pre", "post"})
        self.assertEqual(
            order,
            ["pre-capture", "pre-replay", "post-capture", "post-replay"],
        )
        self.assertTrue(torch.equal(output, post_reference))
        self.assertEqual(stream.synchronize.call_count, 2)
        wait_for_layer.assert_not_called()
        save_layer.assert_not_called()
        profile_step.assert_not_called()
        self.assertTrue(
            any(
                "startup graph-replay output-write smoke passed"
                in call.args[0]
                for call in info_once.call_args_list
            )
        )

    def test_eager_dummy_runs_both_regions_without_advancing_lmcache(self):
        impl = self._make_eligible_impl()
        impl.is_kv_producer = True
        impl.dsa_offload_unbundle = True
        metadata = self._make_decode_metadata()
        hidden_states = torch.empty(1, 4)
        output = torch.empty_like(hidden_states)
        kv_cache = (
            torch.empty(1),
            torch.empty(1),
            torch.empty(1),
        )
        order = []
        topk_indices = torch.tensor([[[1, 2, 3, 4]]], dtype=torch.int32)
        selected_packed = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)

        def pre_graph(*args):
            order.append("pre")
            return (
                hidden_states,
                hidden_states,
                topk_indices,
                selected_packed,
            )

        def post_graph(*args):
            order.append("post")
            return output

        impl._get_staged_sfa_graph_wrappers = MagicMock(
            return_value=(pre_graph, post_graph)
        )
        forward_context = MagicMock()
        forward_context.capturing = False
        forward_context.staged_sfa_graph_dummy_run = True
        forward_context.cudagraph_runtime_mode = CUDAGraphMode.NONE

        with (
            patch.object(
                sfa_v1,
                "get_forward_context",
                return_value=forward_context,
            ),
            patch.object(
                sfa_v1,
                "wait_for_kv_layer_from_connector",
            ) as wait_for_layer,
            patch.object(
                sfa_v1,
                "maybe_save_kv_layer_to_connector",
            ) as save_layer,
            patch.object(
                sfa_v1.torch.profiler,
                "record_function",
                side_effect=lambda *args, **kwargs: nullcontext(),
            ),
            patch.object(sfa_v1._dsa_prof, "step") as profile_step,
            patch.object(
                sfa_v1,
                "_decode_window_save_window_size",
                return_value=0,
            ),
        ):
            result = impl._forward_staged_sfa_graph_poc(
                layer_name="model.layers.0.self_attn.attn",
                index_layer_name="model.layers.0.self_attn.indexer.k_cache",
                index_lmcache_enabled=True,
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                attn_metadata=metadata,
                output=output,
            )

        self.assertIs(result, output)
        self.assertEqual(order, ["pre", "post"])
        self.assertTrue(impl._staged_sfa_dummy_cache_initialized)
        for cache in kv_cache:
            self.assertTrue(torch.equal(cache, torch.zeros_like(cache)))
        wait_for_layer.assert_not_called()
        save_layer.assert_not_called()
        profile_step.assert_not_called()


class TestAscendSFABackend(TestBase):

    def test_get_name(self):
        self.assertEqual(AscendSFABackend.get_name(), "ASCEND_SFA")

    def test_get_builder_cls(self):
        self.assertEqual(AscendSFABackend.get_builder_cls(),
                         AscendSFAMetadataBuilder)

    def test_get_kv_cache_shape(self):
        result = AscendSFABackend.get_kv_cache_shape(2, 4, 8, 128)
        self.assertEqual(result, (2, 4, 8, 128))

    def test_get_impl_cls(self):
        result = AscendSFABackend.get_impl_cls()
        self.assertEqual(result, AscendSFAImpl)


class TestAscendSFAMetadata(TestBase):

    def test_ascend_sfa_metadata_default(self):
        num_actual_tokens = 100
        slot_mapping = torch.randn(100, 4, 1024)
        seq_lens = torch.tensor([30, 50])
        cum_query_lens = torch.tensor([0, 30, 80])
        block_table = torch.randint(0, 100, (100, 4))

        rope_dim = 32
        max_seq_len = int(seq_lens.max().item())
        sin = torch.randn(max_seq_len, rope_dim)
        cos = torch.randn(max_seq_len, rope_dim)

        num_input_tokens = 2
        head_dim = None
        attn_mask = None
        attn_state = AscendAttentionState.ChunkedPrefill

        metadata = AscendSFAMetadata(
            num_actual_tokens=num_actual_tokens,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            cum_query_lens=cum_query_lens,
            block_table=block_table,
            sin=sin,
            cos=cos,
            num_input_tokens=num_input_tokens,
            head_dim=head_dim,
            attn_mask=attn_mask,
            attn_state=attn_state,
        )

        self.assertEqual(metadata.num_actual_tokens, num_actual_tokens)
        self.assertIs(metadata.slot_mapping, slot_mapping)
        self.assertTrue(torch.equal(metadata.seq_lens, seq_lens))
        self.assertTrue(torch.equal(metadata.cum_query_lens, cum_query_lens))
        self.assertIs(metadata.block_table, block_table)
        self.assertIs(metadata.sin, sin)
        self.assertIs(metadata.cos, cos)
        self.assertEqual(metadata.num_input_tokens, num_input_tokens)
        self.assertIs(metadata.head_dim, head_dim)
        self.assertIs(metadata.attn_mask, attn_mask)
        self.assertEqual(metadata.attn_state, attn_state)


class TestAscendSFAMetadataBuilder(TestBase):

    @patch('vllm.distributed.parallel_state._TP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    def setUp(self, mock_tp):
        mock_tp.world_size = 2
        mock_tp.rank_in_group = MagicMock()
        mock_tp.device_group = MagicMock()

        self.mock_cfg = MagicMock()

        self.mock_cfg.parallel_config = MagicMock()
        self.mock_cfg.parallel_config.tensor_parallel_size = 1
        self.mock_cfg.parallel_config.prefill_context_parallel_size = 1
        self.mock_cfg.parallel_config.decode_context_parallel_size = 1

        self.mock_cfg.compilation_config = MagicMock()
        self.mock_cfg.compilation_config.pass_config = MagicMock()
        self.mock_cfg.compilation_config.pass_config.enable_sp = False

        self.mock_cfg.speculative_config.num_speculative_tokens = 0

        self.patcher = patch("vllm.config.get_current_vllm_config",
                             return_value=self.mock_cfg)
        self.patcher.start()

        # Mock parent class __init__ to avoid complex initialization,
        # but still set the essential attributes that child class needs
        def mock_parent_init(self, kv_cache_spec, layer_names, vllm_config,
                             device, metadata_cls, supports_dcp_with_varlen):
            self.metadata_cls = metadata_cls
            self.kv_cache_spec = kv_cache_spec
            self.model_config = vllm_config.model_config
            self.vllm_config = vllm_config
            self.device = device
            self.chunked_prefill_workspace_size = 128 * 1024
            self.chunked_prefill_workspace = torch.empty(
                (self.chunked_prefill_workspace_size,
                 vllm_config.model_config.get_head_size()),
                dtype=vllm_config.model_config.dtype,
                device=device,
            )

        self.parent_init_patcher = patch(
            "vllm.model_executor.layers.attention.mla_attention.MLACommonMetadataBuilder.__init__",
            mock_parent_init)
        self.parent_init_patcher.start()

        if hasattr(enable_dsa_cp, "cache_clear"):
            enable_dsa_cp.cache_clear()

    def tearDown(self):
        self.patcher.stop()
        self.parent_init_patcher.stop()

    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_default(self):
        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(kv_cache_spec=kv_cache_spec,
                                           layer_names=layer_names,
                                           vllm_config=vllm_config,
                                           device=device)

        assert builder.device == device
        assert builder.vllm_config == vllm_config

    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_build(
        self,
        mock_enable_dsa_cp,
        mock_get_cos_and_sin_mla,
        mock_get_current_vllm_config,
    ):
        mock_enable_dsa_cp.return_value = False

        cfg = MagicMock()
        cfg.model_config = MagicMock()
        cfg.model_config.hf_text_config = MagicMock()

        mock_get_current_vllm_config.return_value = cfg
        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(kv_cache_spec=kv_cache_spec,
                                           layer_names=layer_names,
                                           vllm_config=vllm_config,
                                           device=device)

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 10
        common_attn_metadata.num_actual_tokens = 100
        common_attn_metadata.query_start_loc = torch.tensor(
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.query_start_loc_cpu = torch.tensor(
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.slot_mapping = torch.randn(100, 4, 1024)
        common_attn_metadata.seq_lens_cpu = torch.tensor([2] * 10)
        common_attn_metadata.positions = torch.randn(100)
        common_attn_metadata.attn_mask = None
        common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        common_attn_metadata.block_table_tensor = torch.randn(100, 4)
        common_attn_metadata.cos = None
        common_attn_metadata.sin = None
        common_attn_metadata.num_input_tokens = 100

        mock_get_cos_and_sin_mla.return_value = (torch.randn(100),
                                                 torch.randn(100))

        metadata = builder.build(
            common_prefix_len=10,
            common_attn_metadata=common_attn_metadata,
        )

        assert isinstance(metadata, AscendSFAMetadata)
        assert metadata.num_actual_tokens == common_attn_metadata.num_actual_tokens
        assert metadata.slot_mapping.shape == (100, 4, 1024)

    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_build_for_graph_capture(
            self, mock_get_cos_and_sin_mla, mock_get_current_vllm_config):
        cfg = MagicMock()
        cfg.model_config = MagicMock()
        cfg.model_config.hf_text_config = MagicMock()

        mock_get_current_vllm_config.return_value = cfg

        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(kv_cache_spec=kv_cache_spec,
                                           layer_names=layer_names,
                                           vllm_config=vllm_config,
                                           device=device)

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 10
        common_attn_metadata.num_actual_tokens = 100
        common_attn_metadata.query_start_loc = torch.tensor(
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.query_start_loc_cpu = torch.tensor(
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.slot_mapping = torch.randn(100, 4, 1024)
        common_attn_metadata.seq_lens_cpu = torch.tensor([2] * 10)
        common_attn_metadata.positions = torch.randn(100)
        common_attn_metadata.attn_mask = None
        common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        common_attn_metadata.block_table_tensor = torch.randn(100, 4)
        common_attn_metadata.cos = None
        common_attn_metadata.sin = None
        common_attn_metadata.num_input_tokens = 100

        mock_get_cos_and_sin_mla.return_value = (torch.randn(100),
                                                 torch.randn(100))

        attn_metadata = builder.build_for_graph_capture(
            common_attn_metadata=common_attn_metadata,
            attn_state=AscendAttentionState.DecodeOnly,
        )

        assert isinstance(attn_metadata, AscendSFAMetadata)
        assert attn_metadata.attn_state == AscendAttentionState.DecodeOnly
